"""Multi-coin stream manager: real closed candles in, calibrated trade probabilities out.

Every model input comes from fully closed 1m and 15m candles of the Binance perpetual
behind the CoinDCX pair (BinancePublicClient.fetch_candle_history). New candles are pulled
when one closes; between candles only the displayed Binance price and the order book refresh.
Downloaded candles are kept on disk (<home>/cache/candles), so a retrain or restart only
fetches the candles that closed since.

- 1m candles: residual-alpha models, microstructure, and the 5m / 15m / 1h win-probability
  and trade-risk horizons.
- 15m candles: the 4h / 8h win-probability and trade-risk horizons (more history per candle).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from futurecast.config import AppConfig, AssetsConfig, load_config
from futurecast.engine.forecaster import PureForecastEngine
from futurecast.features.orderbook import compute_orderbook_metrics
from futurecast.features.pipeline import FeaturePipeline
from futurecast.features.shitcoin import is_high_beta_shitcoin
from futurecast.ingestion.binance import BinancePublicClient
from futurecast.ingestion.cache import CandleCache
from futurecast.ingestion.factory import (
    detect_exchange,
    extract_base_and_quote,
    get_symbol_sector,
    resolve_benchmarks,
    sanitize_symbol_for_filename,
)
from futurecast.ingestion.storage import align_cross_asset_data
from futurecast.labels.residual_alpha import compute_residual_alpha_target
from futurecast.models.ensemble import MultiModelEnsemble
from futurecast.models.trade_probability import (
    HORIZONS_BY_INTERVAL,
    TradeModelConfig,
    TradeProbabilityModel,
    trade_model_path,
)
from futurecast.models.trade_risk import TradeRiskConfig, TradeRiskModel, risk_model_path
from futurecast.paths import resolve
from futurecast.schema import ForecastEvent

logger = logging.getLogger(__name__)

CANDLE_MINUTES: Dict[str, int] = {"1m": 1, "15m": 15}
ENGINE_INTERVAL = "1m"
RESIDUAL_HORIZON_MINUTES = 15
HISTORY_BARS: Dict[str, int] = {"1m": 100_000, "15m": 20_000}  # training: ~69 days of 1m, ~208 days of 15m
MIN_TRAIN_BARS: Dict[str, int] = {"1m": 15_000, "15m": 3_000}  # ~10 and ~31 days
# Per interval kept live: above the longest feature window (1,440 candles) and, at 1m, the clock-time
# funding mean (3 days = 4,320 candles), so live features match the ones the models were trained on
LIVE_BARS = 6_000
CACHE_BARS = 120_000  # candles kept on disk per market and interval: training history plus newer candles


def _synthetic_candles(symbol: str, minutes: int, bars: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random-walk closed candles, used only in offline mode."""
    end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min") - pd.Timedelta(minutes=minutes)
    ts = pd.date_range(end=end, periods=bars, freq=f"{minutes}min")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002 * np.sqrt(minutes), bars)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.001 * np.sqrt(minutes), bars))
    volume = rng.uniform(1e5, 5e5, bars)
    taker = rng.uniform(0.4, 0.6, bars)
    return pd.DataFrame({
        "timestamp": ts, "symbol": symbol, "open": open_,
        "high": np.maximum(open_, close) * (1 + wick), "low": np.minimum(open_, close) * (1 - wick),
        "close": close, "volume": volume, "quote_volume": volume * close, "count": 100,
        "taker_buy_volume": volume * taker, "taker_buy_quote_volume": volume * close * taker,
    })


def _binance_client(assets: AssetsConfig) -> BinancePublicClient:
    source, storage = assets.data_source, assets.storage
    cache = CandleCache(Path(storage.cache_dir) / "candles", max_rows=CACHE_BARS) if storage.use_cache else None
    return BinancePublicClient(base_url=source.base_url, timeout=source.request_timeout_sec,
                               max_retries=source.max_retries, candle_cache=cache)


def _base(symbol: str) -> str:
    return extract_base_and_quote(symbol)[0]


def _json_safe(obj: Any) -> Any:
    """Plain JSON types only: numpy scalars unwrapped, NaN/inf become None."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _event_payload(ev: ForecastEvent) -> Dict[str, Any]:
    """15m residual-alpha model output and order book for the latest 5m candle."""
    return {
        "timestamp": ev.timestamp.isoformat(),
        "benchmark_price": ev.benchmark_price,
        "expected_residual_alpha_bps": ev.expected_residual_alpha_15m_bps,
        "z_score": ev.z_score,
        "beta_hedge_ratio": ev.beta_hedge_ratio,
        "signal": getattr(ev.signal, "value", str(ev.signal)),
        "confidence_percentile": ev.confidence_percentile,
        "microstructure_context": ev.microstructure_context,
        "orderbook_depth": ev.orderbook_depth,
        "model_predictions": {name: item.model_dump(mode="json") for name, item in ev.model_predictions.items()},
        "cross_verification": ev.cross_verification.model_dump(mode="json") if ev.cross_verification else None,
    }


class SymbolWorker:
    """Candle buffers, models and latest outputs for one coin."""

    def __init__(
        self,
        symbol: str,
        exchange: str,
        benchmarks: List[str],
        sector: str,
        config: AppConfig,
        models_dir: Path,
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange.lower()
        self.benchmarks = benchmarks
        self.sector = sector
        self.config = config
        self.models_dir = Path(models_dir)
        self.clean_symbol = sanitize_symbol_for_filename(symbol)
        self._lock = threading.RLock()

        secondary = benchmarks[1] if len(benchmarks) > 1 else benchmarks[0]
        self.pipes = {
            iv: FeaturePipeline(config=config.features, target_symbol=symbol,
                                primary_benchmark=benchmarks[0], secondary_benchmark=secondary)
            for iv in CANDLE_MINUTES
        }

        self.ensemble = MultiModelEnsemble(config.model)
        self.ensemble.load_models(self.models_dir, self.clean_symbol, bar_interval=ENGINE_INTERVAL)
        self.trade_models: Dict[str, Optional[TradeProbabilityModel]] = {
            iv: TradeProbabilityModel.load(self._trade_path(iv)) for iv in CANDLE_MINUTES
        }
        self.risk_models: Dict[str, Optional[TradeRiskModel]] = {
            iv: TradeRiskModel.load(self._risk_path(iv)) for iv in CANDLE_MINUTES
        }
        # Coins trained before the risk model existed count as untrained, so they get retrained
        self.is_custom_trained = all(self.trade_models.values()) and all(self.risk_models.values())
        self.training_status = "trained" if self.is_custom_trained else "untrained"
        self.training_error: Optional[str] = None
        self.training_progress: Optional[Dict[str, Any]] = None  # what training is doing right now
        self.training_report: Dict[str, Any] = {}

        self.engine = self._make_engine()
        self.candles: Dict[str, pd.DataFrame] = {iv: pd.DataFrame() for iv in CANDLE_MINUTES}
        self.data_source: Optional[str] = None
        self.trade_probabilities: Dict[str, Any] = {}
        self.trade_risk: Dict[str, Any] = {}
        self.live_price: Optional[float] = None
        self.orderbook: Dict[str, Any] = {}
        self.latest_event: Optional[ForecastEvent] = None
        self.last_bar_time: Optional[datetime] = None

    # ───────── setup ─────────

    def _trade_path(self, interval: str) -> Path:
        return trade_model_path(self.models_dir, f"{self.clean_symbol}_{interval}")

    def _risk_path(self, interval: str) -> Path:
        return risk_model_path(self.models_dir, f"{self.clean_symbol}_{interval}")

    def _make_engine(self) -> PureForecastEngine:
        from futurecast.publisher.broadcaster import SignalBroadcaster

        cfg = self.config.model_copy(deep=True)
        cfg.assets.target_asset = self.symbol
        cfg.assets.benchmark_assets = self.benchmarks
        return PureForecastEngine(
            config=cfg,
            trainer=self.ensemble.models["LightGBM"],
            ensemble=self.ensemble,
            feature_pipeline=self.pipes[ENGINE_INTERVAL],
            broadcaster=SignalBroadcaster(output_dir=self.config.forecasting.output_dir, render_terminal=False),
            min_buffer_size=60,
            max_buffer_size=LIVE_BARS,
            bar_minutes=CANDLE_MINUTES[ENGINE_INTERVAL],
        )

    @property
    def aligned_buffer(self) -> pd.DataFrame:
        return self.candles[ENGINE_INTERVAL]

    # ───────── candles ─────────

    def _fetch_aligned(
        self,
        client: BinancePublicClient,
        interval: str,
        bars: int,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Closed candles with each coin's settled funding rate, target and benchmarks on one time grid.

        `progress(symbol, done, total)` reports candles downloaded per symbol (0, 0 as each starts).
        """
        def fetch(sym: str) -> Tuple[pd.DataFrame, Optional[str]]:
            if progress is None:
                return client.fetch_candle_history(sym, interval, bars, funding=True)
            progress(sym, 0, 0)
            return client.fetch_candle_history(sym, interval, bars, funding=True,
                                               progress=lambda done, total: progress(sym, done, total))

        target, source = fetch(self.symbol)
        if target.empty:
            return pd.DataFrame(), None
        benches = {}
        for bench in self.benchmarks:
            bench_df, _ = fetch(bench)
            if bench_df.empty:
                return pd.DataFrame(), None
            benches[bench] = bench_df
        aligned = align_cross_asset_data(target, benches, freq=f"{CANDLE_MINUTES[interval]}min")
        return aligned, source

    def refresh_candles(self, client: BinancePublicClient, now: Optional[pd.Timestamp] = None) -> List[str]:
        """Append newly closed candles per interval and rerun models; returns intervals updated."""
        now = now or pd.Timestamp.now(tz="UTC")
        updated = []
        for iv, minutes in CANDLE_MINUTES.items():
            have = self.candles[iv]
            last_closed = now.floor(f"{minutes}min") - pd.Timedelta(minutes=minutes)
            if not have.empty and have.index[-1] >= last_closed:
                continue
            missing = LIVE_BARS if have.empty else int((last_closed - have.index[-1]) / pd.Timedelta(minutes=minutes))
            fresh, source = self._fetch_aligned(client, iv, min(LIVE_BARS, missing + 3))
            if fresh.empty:
                continue

            with self._lock:
                self.data_source = source
                if have.empty or missing >= LIVE_BARS - 3:  # first load, or a gap too long to bridge
                    self.candles[iv] = fresh.tail(LIVE_BARS)
                    if iv == ENGINE_INTERVAL:
                        self._load_engine(self.candles[iv])
                else:
                    new_rows = fresh[fresh.index > have.index[-1]]
                    if new_rows.empty:
                        continue
                    self.candles[iv] = pd.concat([have, new_rows]).tail(LIVE_BARS)
                    if iv == ENGINE_INTERVAL:
                        self._feed_engine(new_rows)
                self._update_trade_models(iv)
            updated.append(iv)
        return updated

    def load_synthetic(self, seed: int = 7) -> None:
        """Offline mode only: random-walk candles so the page has something to render."""
        rng = np.random.default_rng(seed)
        with self._lock:
            for iv, minutes in CANDLE_MINUTES.items():
                target = _synthetic_candles(self.symbol, minutes, 400, rng)
                benches = {b: _synthetic_candles(b, minutes, 400, rng) for b in self.benchmarks}
                self.candles[iv] = align_cross_asset_data(target, benches, freq=f"{minutes}min")
            self.data_source = "synthetic (offline mode)"
            self._load_engine(self.candles[ENGINE_INTERVAL])
            for iv in CANDLE_MINUTES:
                self._update_trade_models(iv)

    # ───────── models ─────────

    def _feed_engine(self, rows: pd.DataFrame) -> None:
        event = self.engine.feed_bars(rows, orderbook_depth=self.orderbook)
        if event is not None:
            self.latest_event = event
            self.last_bar_time = event.timestamp

    def _load_engine(self, candles: pd.DataFrame, warm_rows: int = 300) -> None:
        """Restart the residual engine on a candle history, pre-filling per-model z-score history."""
        self.engine.reset()
        if len(candles) < 2:
            return
        self._feed_engine(candles.iloc[:-1])
        feats = self.engine.last_features
        if feats is not None and self.ensemble.is_loaded:
            tail = feats.iloc[-warm_rows:]
            for name, model in self.ensemble.models.items():
                if model.model is None:
                    continue
                try:
                    for pred in model.predict(tail):
                        self.engine.calibrator.calibrate(float(pred), 1.0, key=name)
                except Exception as err:
                    logger.debug(f"Z-score warmup failed for {self.symbol} {name}: {err}")
        self._feed_engine(candles.iloc[[-1]])

    def _update_trade_models(self, interval: str) -> None:
        """Win probabilities and trade risk for this interval's horizons from its trained models."""
        for horizon in HORIZONS_BY_INTERVAL[interval]:
            self.trade_probabilities.pop(horizon, None)
            self.trade_risk.pop(horizon, None)
        win_model, risk_model = self.trade_models.get(interval), self.risk_models.get(interval)
        candles = self.candles[interval]
        if (win_model is None and risk_model is None) or candles.empty:
            return
        feats = self.engine.last_features
        if interval != ENGINE_INTERVAL or feats is None or feats.index[-1] != candles.index[-1]:
            feats = self.pipes[interval].generate_features(candles)
        close = candles[f"{self.symbol}_close"]
        if win_model is not None:
            try:
                self.trade_probabilities.update(win_model.predict_latest(feats, close))
            except Exception as err:
                logger.warning(f"Trade probability failed for {self.symbol} {interval}: {err}")
        if risk_model is not None:
            try:
                self.trade_risk.update(risk_model.predict_latest(feats, close))
            except Exception as err:
                logger.warning(f"Trade risk failed for {self.symbol} {interval}: {err}")

    def refresh_orderbook(self, client: BinancePublicClient) -> None:
        raw = client.fetch_orderbook_depth(self.symbol, 20)
        metrics = compute_orderbook_metrics(raw)
        with self._lock:
            self.orderbook = metrics
            self.engine.refresh_orderbook(metrics)

    def _train_residual(self, aligned: pd.DataFrame, feats: pd.DataFrame) -> Tuple[MultiModelEnsemble, Dict[str, Any]]:
        """15m residual-alpha ensemble on 5m candles: purged 70/15/15 split, scored on the last slice."""
        horizon = RESIDUAL_HORIZON_MINUTES // CANDLE_MINUTES[ENGINE_INTERVAL]
        target = compute_residual_alpha_target(
            aligned_df=aligned, features_df=feats, target_symbol=self.symbol,
            benchmark_symbol=self.benchmarks[0], horizon_minutes=horizon,
        )
        valid = target.notna()
        X, y = feats.loc[valid], target.loc[valid]
        n = len(y)
        train_end, es_end = int(n * 0.70), int(n * 0.85)
        ensemble = MultiModelEnsemble(self.config.model)
        metrics = ensemble.train_all(
            X.iloc[:train_end - horizon], y.iloc[:train_end - horizon],
            X.iloc[train_end:es_end - horizon], y.iloc[train_end:es_end - horizon],
            X.iloc[es_end:], y.iloc[es_end:],
        )
        return ensemble, metrics

    def _set_progress(self, stage: str, label: str, step: int, steps: int, done: int = 0, total: int = 0) -> None:
        within = done / total if total else 0.0
        self.training_progress = {
            "stage": stage, "label": label, "step": step, "steps": steps, "done": done, "total": total,
            "fraction": round((step - 1 + within) / steps, 4),
        }

    def train_sync(self, days: int = 0) -> Dict[str, Any]:
        """Fetch real candle history, validate and fit all models, then hot-swap them in.

        `days` is accepted for API compatibility; history depth is HISTORY_BARS per interval.
        Every interval is downloaded first (status "downloading", candles already in the disk
        cache are not fetched again), then the models are fitted (status "training"), with
        `training_progress` describing the current step. Never trains on synthetic data: if
        the exchange cannot supply enough candles the coin stays untrained and the error is reported.
        """
        self.training_status = "downloading"
        self.training_error = None
        client = _binance_client(self.config.assets)
        report: Dict[str, Any] = {"data": {}, "trade": {}, "risk": {}}
        try:
            if client.resolve(self.symbol) is None:
                raise RuntimeError(f"no trading Binance perpetual found for {self.symbol}")

            downloads = [(iv, sym) for iv in CANDLE_MINUTES for sym in (self.symbol, *self.benchmarks)]
            aligned_by_iv: Dict[str, pd.DataFrame] = {}
            for iv in CANDLE_MINUTES:
                def downloaded(sym: str, done: int, total: int, iv: str = iv) -> None:
                    self._set_progress("download", f"{iv} candles · {_base(sym)}",
                                       downloads.index((iv, sym)) + 1, len(downloads), done, total)

                aligned, source = self._fetch_aligned(client, iv, HISTORY_BARS[iv], progress=downloaded)
                if len(aligned) < MIN_TRAIN_BARS[iv]:
                    raise RuntimeError(f"only {len(aligned)} closed {iv} candles available (need {MIN_TRAIN_BARS[iv]})")
                aligned_by_iv[iv] = aligned
                report["data"][iv] = {"source": source, "candles": len(aligned),
                                      "start": str(aligned.index[0]), "end": str(aligned.index[-1])}

            self.training_status = "training"
            steps = sum(1 + 4 * len(HORIZONS_BY_INTERVAL[iv]) for iv in CANDLE_MINUTES) + 1
            done_steps = 0

            def advance(label: str) -> None:
                nonlocal done_steps
                done_steps += 1
                self._set_progress("train", label, done_steps, steps)

            trade_models: Dict[str, TradeProbabilityModel] = {}
            risk_models: Dict[str, TradeRiskModel] = {}
            history: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
            for iv, aligned in aligned_by_iv.items():
                advance(f"{iv} candle features")
                feats = self.pipes[iv].generate_features(aligned)
                ohlc = (aligned[f"{self.symbol}_close"], aligned[f"{self.symbol}_high"], aligned[f"{self.symbol}_low"])
                model = TradeProbabilityModel(TradeModelConfig(interval=iv, horizons=dict(HORIZONS_BY_INTERVAL[iv])))
                report["trade"][iv] = model.fit(feats, *ohlc, on_step=lambda h, side: advance(f"{h} {side} win odds"))
                risk = TradeRiskModel(TradeRiskConfig(interval=iv, horizons=dict(HORIZONS_BY_INTERVAL[iv])))
                report["risk"][iv] = risk.fit(feats, *ohlc, on_step=lambda h, side: advance(f"{h} {side} worst move"))
                trade_models[iv] = model
                risk_models[iv] = risk
                history[iv] = (aligned, feats)

            advance("residual alpha")
            aligned1, feats1 = history[ENGINE_INTERVAL]
            ensemble, report["residual"] = self._train_residual(aligned1, feats1)
        except Exception as err:
            self.training_status = "failed"
            self.training_error = str(err)
            self.training_progress = None
            logger.error(f"Training failed for {self.symbol}: {err}")
            raise
        finally:
            client.close()

        for iv, model in trade_models.items():
            model.save(self._trade_path(iv))
        for iv, risk in risk_models.items():
            risk.save(self._risk_path(iv))
        ensemble.save_models(self.models_dir, self.clean_symbol, meta={
            "bar_interval": ENGINE_INTERVAL,
            "horizon_minutes": RESIDUAL_HORIZON_MINUTES,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            **{f"history_{k}": v for k, v in report["data"][ENGINE_INTERVAL].items()},
        })

        with self._lock:
            self.trade_models = dict(trade_models)
            self.risk_models = dict(risk_models)
            self.ensemble = ensemble
            self.engine.ensemble = ensemble
            self.engine.trainer = ensemble.models["LightGBM"]
            for iv, (aligned, _) in history.items():
                self.candles[iv] = aligned.tail(LIVE_BARS)
            self.data_source = report["data"][ENGINE_INTERVAL]["source"]
            self._load_engine(self.candles[ENGINE_INTERVAL])
            for iv in CANDLE_MINUTES:
                self._update_trade_models(iv)
            self.is_custom_trained = True
            self.training_status = "trained"
            self.training_progress = None
            self.training_report = report

        logger.info(f"Trained and hot-reloaded models for {self.symbol}")
        return report

    # ───────── output ─────────

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            ev = self.latest_event
            candles = self.candles[ENGINE_INTERVAL]
            price = self.live_price
            if price is None and ev is not None and ":1000" not in (self.data_source or ""):
                price = ev.current_price  # underlying close; skipped for 1000x-quoted contracts
            data: Dict[str, Any] = {
                "symbol": self.symbol,
                "exchange": self.exchange.upper(),
                "sector": self.sector,
                "benchmarks": self.benchmarks,
                "is_shitcoin": is_high_beta_shitcoin(self.symbol, self.sector),
                "has_data": not candles.empty,
                "is_custom_trained": self.is_custom_trained,
                "training_status": self.training_status,
                "training_progress": self.training_progress,
                "training_error": self.training_error,
                "data_source": self.data_source,
                "candle_time": str(candles.index[-1]) if not candles.empty else None,
                "current_price": price,
                "trade_probabilities": self.trade_probabilities,
                "trade_risk": self.trade_risk,
                "execution_disabled": True,
            }
            if ev is not None:
                data.update(_event_payload(ev))
            return _json_safe(data)


class MultiCoinStreamManager:
    """Central supervisor coordinating real-time candles, models and WebSocket clients."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        models_dir: str | Path = "models",
        persistence_path: Optional[str | Path] = "tracked_symbols.json",
        live: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.models_dir = resolve(models_dir)
        self.persistence_path = resolve(persistence_path) if persistence_path is not None else None
        self.live = live
        self.workers: Dict[str, SymbolWorker] = {}
        self.ws_clients: Set[Any] = set()
        self._running = False
        self._train_lock: Optional[asyncio.Lock] = None
        self._training_tasks: Dict[str, asyncio.Task] = {}

    def load_persisted_symbols(self) -> List[str]:
        """Load list of tracked symbols from disk if previously saved."""
        if self.persistence_path is None:
            return []
        try:
            if self.persistence_path.exists():
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    symbols = data.get("symbols", [])
                    if isinstance(symbols, list) and symbols:
                        cleaned = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
                        logger.info(f"Loaded {len(cleaned)} persisted symbols from {self.persistence_path}: {cleaned}")
                        return cleaned
        except Exception as e:
            logger.warning(f"Could not load tracked symbols from {self.persistence_path}: {e}")
        return []

    def persist_symbols(self) -> None:
        """Save current list of tracked symbols to disk."""
        if self.persistence_path is None:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"symbols": list(self.workers.keys()), "updated_at": datetime.now(timezone.utc).isoformat()}
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist tracked symbols to {self.persistence_path}: {e}")

    def get_tracked_symbols(self) -> List[str]:
        return list(self.workers.keys())

    def add_symbol(self, symbol: str, exchange: Optional[str] = None, persist: bool = True) -> SymbolWorker:
        """Start tracking a coin. In live mode its candles load on the next stream tick."""
        sym = symbol.strip().upper()
        if sym in self.workers:
            return self.workers[sym]

        ex = (exchange or detect_exchange(sym)).lower()
        benchmarks = resolve_benchmarks(sym, ex, sector_aware=True)
        sector = get_symbol_sector(sym)
        logger.info(f"Adding coin: {sym} (Exchange: {ex}, Sector: {sector}, Benchmarks: {benchmarks})")
        worker = SymbolWorker(sym, ex, benchmarks, sector, self.config, self.models_dir)
        if not self.live:
            worker.load_synthetic(seed=sum(map(ord, sym)))
        self.workers[sym] = worker
        if persist:
            self.persist_symbols()
        return worker

    def remove_symbol(self, symbol: str, persist: bool = True) -> bool:
        sym = symbol.strip().upper()
        if sym not in self.workers:
            return False
        del self.workers[sym]
        if persist:
            self.persist_symbols()
        logger.info(f"Removed coin: {sym}")
        return True

    # ───────── training ─────────

    def schedule_training(self, symbol: str) -> None:
        """Queue background training (one coin at a time) unless already queued or running."""
        sym = symbol.strip().upper()
        task = self._training_tasks.get(sym)
        if task is not None and not task.done():
            return
        self._training_tasks[sym] = asyncio.create_task(self.train_symbol_models(sym))

    def ensure_trained(self) -> None:
        for sym, worker in self.workers.items():
            if not worker.is_custom_trained:
                self.schedule_training(sym)

    async def train_symbol_models(self, symbol: str, days: int = 0) -> Dict[str, Any]:
        sym = symbol.strip().upper()
        if sym not in self.workers:
            raise KeyError(f"Symbol {sym} is not currently tracked")
        worker = self.workers[sym]
        if self._train_lock is None:
            self._train_lock = asyncio.Lock()

        worker.training_status = "queued"
        await self.broadcast({"type": "FORECAST_UPDATE", **worker.snapshot()})
        async with self._train_lock:
            worker.training_status = "downloading"
            await self.broadcast({"type": "FORECAST_UPDATE", **worker.snapshot()})
            loop = asyncio.get_running_loop()
            try:
                report = await loop.run_in_executor(None, worker.train_sync, days)
            except Exception as err:
                report = {"status": "failed", "error": str(err)}
        if sym in self.workers:
            await self.broadcast({"type": "FORECAST_UPDATE", **worker.snapshot()})
        return report

    # ───────── output ─────────

    def coin_snapshot(self, worker: SymbolWorker) -> Dict[str, Any]:
        return worker.snapshot()

    def get_snapshot(self, timeframe: str = "15m") -> List[Dict[str, Any]]:
        """Current state of every tracked coin (all horizons are included)."""
        return [w.snapshot() for w in list(self.workers.values())]

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Push JSON message to all connected WebSocket clients."""
        if not self.ws_clients:
            return
        payload = json.dumps(message, default=str)
        disconnected = set()
        for ws in self.ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.ws_clients.discard(ws)

    # ───────── streaming ─────────

    def _refresh_worker(self, worker: SymbolWorker, client: BinancePublicClient) -> None:
        try:
            worker.refresh_candles(client)
        except Exception as err:
            logger.warning(f"Candle refresh failed for {worker.symbol}: {err}")
        try:
            worker.refresh_orderbook(client)
        except Exception as err:
            logger.debug(f"Order book refresh failed for {worker.symbol}: {err}")

    async def start_streaming(self, poll_interval_sec: float = 3.0, live: bool = True) -> None:
        """Refresh prices and order books every poll; candles and models when a candle closes."""
        self._running = True
        self.live = live
        logger.info(f"Starting stream loop (interval: {poll_interval_sec}s, live: {live})...")
        client = _binance_client(self.config.assets)
        if live:
            self.ensure_trained()

        while self._running:
            try:
                workers = list(self.workers.values())
                if not workers:
                    await asyncio.sleep(1.0)
                    continue
                if live:
                    prices = await asyncio.to_thread(client.fetch_live_prices, [w.symbol for w in workers])
                    for w in workers:
                        if w.symbol in prices:
                            w.live_price = prices[w.symbol]
                    await asyncio.gather(*(asyncio.to_thread(self._refresh_worker, w, client) for w in workers))
                for w in workers:
                    if w.symbol in self.workers:
                        await self.broadcast({"type": "FORECAST_UPDATE", **w.snapshot()})
            except Exception as e:
                logger.error(f"Error in stream loop: {e}", exc_info=True)
            await asyncio.sleep(poll_interval_sec)

    def stop(self) -> None:
        self._running = False
