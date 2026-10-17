"""How badly a long or a short opened now can go against the trader within a window, in the
trader's money: the stop, the leverage and the 1-in-10 bad case.

For a window of h candles (regime.WINDOWS), entry at the close of candle t:

- Adverse move: for a long (close_t - lowest low of candles t+1..t+h) / close_t, for a short
  (highest high of t+1..t+h - close_t) / close_t, at least 0. Highs and lows, so a dip inside
  the window counts. The last h candles have none yet.
- Swing unit: a HAR volatility forecast for the next h candles (Corsi 2009: a linear regression
  of the log of their mean Parkinson variance on the log realized volatility of the last 1, 4,
  24 and 96 windows, close-to-close and high-low), as a price fraction over the window. Targets
  are adverse moves in swing units; that is what lets thousands of past windows at every
  volatility level say something about a 0.25% or a 3% move.
- Two models of the adverse move in swing units, per side:
  `har`: the empirical distribution of the training windows' adverse moves (one for all rows;
  the HAR forecast alone moves it), and `lgbm`: LightGBM quantile regressions (`alphas`) on the
  inputs (risk_inputs), which can see skew, crowded funding and trend. Its stop-hit chance
  interpolates the predicted quantiles (exceed_chance).
- Bad case: the predicted `bad_case` (90%) quantile. Stop-hit chance: P(adverse move ≥ the stop's
  price move). Liquidation chance: the same at the isolated-margin liquidation distance.
- Risk score, 0-100: the stop-hit chance in percent. (100 x bad case / stop distance was the
  first idea; at 20x with a -5% margin stop it read 100 for every window from 1h up.)

The stop is on margin (the PnL % CoinDCX shows): a -5% stop at 20x is a 0.25% price move.

walk_forward is the honesty check: expanding training windows, test blocks after the first
`train_share` of history, a gap of h + `gap_hours` between them so overlapping targets don't
leak. LightGBM is used for a window and side only if its pinball loss beats the HAR baseline's by
`min_gain`, and a reading counts as checked only if the chosen model's bad case was exceeded in
`coverage_band` of the test windows.

Every input at a candle uses that candle's close and earlier ones only. Positions count candles,
not clock time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from deltarisk.ingestion.factory import extract_base_and_quote, get_symbol_sector
from deltarisk.features.setups import SetupSpec
from deltarisk.stats.regime import RegimeSpec, Window, regime_frame, trend_record

logger = logging.getLogger(__name__)

SIDES = ("long", "short")
HAR_LAGS = (1, 4, 24, 96)  # windows
HAR_COLUMNS = tuple(f"{kind}_{k}" for kind in ("rv", "pk") for k in HAR_LAGS)
FUNDING_HOURS = 8  # Binance settles most perpetuals at 00, 08 and 16 UTC
LOG_FLOOR = 1e-12  # variance floor before taking logs (flat candles)
ROUND_TRIP_PCT = 2 * SetupSpec.taker_bps / 100.0  # market in and out on CoinDCX USDT-M, GST included


# ───────── tiers, stops and the score ─────────

@dataclass(frozen=True)
class Tier:
    name: str
    leverage: float
    mmr: float  # maintenance margin rate (fixed: Binance's brackets need an authenticated call)


TIERS: Dict[str, Tier] = {
    "megacap": Tier("megacap", 20.0, 0.004),
    "altcoin": Tier("altcoin", 10.0, 0.005),
    "shitcoin": Tier("shitcoin", 3.0, 0.01),
}
MEGACAPS = ("BTC", "ETH", "SOL", "BNB", "XRP")
SHITCOIN_VOLUME_USD = 50e6  # Binance 24h quote volume below this: shitcoin


def coin_tier(symbol: str, quote_volume_24h: Optional[float] = None) -> str:
    """megacap (a fixed list), shitcoin (meme sector, or under $50M traded in 24h), else altcoin."""
    if extract_base_and_quote(symbol)[0] in MEGACAPS:
        return "megacap"
    if get_symbol_sector(symbol) == "MEME":
        return "shitcoin"
    if quote_volume_24h is not None and quote_volume_24h < SHITCOIN_VOLUME_USD:
        return "shitcoin"
    return "altcoin"


def stop_move(stop_margin_pct: float, leverage: float) -> float:
    """Price move (fraction) that loses `stop_margin_pct` % of the margin at `leverage`."""
    return abs(stop_margin_pct) / 100.0 / leverage


def liquidation_move(leverage: float, mmr: float) -> float:
    """Isolated-margin liquidation distance as a price fraction: about 1/L minus the maintenance rate."""
    return max(1.0 / leverage - mmr, 0.0)


def risk_score(stop_hit_chance: float) -> Optional[int]:
    """The stop-hit chance in percent, rounded."""
    if not np.isfinite(stop_hit_chance):
        return None
    return int(round(100.0 * min(max(stop_hit_chance, 0.0), 1.0)))


# ───────── inputs and targets ─────────

def _col(frame: pd.DataFrame, symbol: str, name: str) -> pd.Series:
    return frame[f"{symbol}_{name}"].astype(float)


def _log_vol(variance: pd.Series) -> pd.Series:
    return 0.5 * np.log(variance.clip(lower=LOG_FLOOR))


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """One candle's Parkinson variance estimate from its high-low range."""
    return np.log(high / low) ** 2 / (4.0 * math.log(2.0))


def _funding_inputs(index: pd.DatetimeIndex, minutes: int, hourly: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """The 1h chart's funding rate (settled before that hour closed) and its 90-day percentile,
    as known at each candle's close."""
    rate = _col(hourly, symbol, "funding_rate")
    pct = 100.0 * rate.rolling(90 * 24, min_periods=7 * 24).rank(pct=True)
    known = pd.DataFrame({"known_at": hourly.index + pd.Timedelta(hours=1),
                          "funding": rate.to_numpy(), "funding_pct": pct.to_numpy()})
    closes = pd.DataFrame({"known_at": index + pd.Timedelta(minutes=minutes)})
    known["known_at"] = known["known_at"].astype(closes["known_at"].dtype)
    merged = pd.merge_asof(closes, known, on="known_at", direction="backward")
    return merged[["funding", "funding_pct"]].set_index(index)


def risk_inputs(
    frame: pd.DataFrame,
    symbol: str,
    window: Window,
    benchmark: Optional[str] = None,
    hourly: Optional[pd.DataFrame] = None,
    regime_spec: RegimeSpec = RegimeSpec(),
    regime: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Model inputs at every candle of the window's chart (aligned frame, `{symbol}_close` etc.).

    HAR columns (HAR_COLUMNS): log per-candle volatility over the last 1, 4, 24 and 96 windows,
    close-to-close (rv) and high-low (pk). Then momentum, efficiency and volatility percentile
    (regime_frame); the last window's move in typical moves; taker-buy share of the last 1 and 4
    windows; position in the last 24 windows' range; UTC hour of the close and hours to the next
    funding settlement; the funding rate and its percentile when `hourly` carries it; and the
    benchmark's volatility when `benchmark` columns are present."""
    h = window.candles
    minutes = window.minutes // h
    close, high, low = (_col(frame, symbol, c) for c in ("close", "high", "low"))
    r = np.log(close).diff()
    pk = parkinson(high, low)
    X = pd.DataFrame(index=frame.index)
    for k in HAR_LAGS:
        X[f"rv_{k}"] = _log_vol((r ** 2).rolling(k * h).mean())
    for k in HAR_LAGS:
        X[f"pk_{k}"] = _log_vol(pk.rolling(k * h).mean())

    if regime is None:
        regime = regime_frame(frame, symbol, window, regime_spec)
    X["momentum"], X["efficiency"], X["vol_pct"] = regime["momentum"], regime["efficiency"], regime["vol_pct"]
    X["move"] = np.log(close / close.shift(h)) / (np.exp(X[f"rv_{HAR_LAGS[-1]}"]) * math.sqrt(h))

    if f"{symbol}_taker_buy_quote_volume" in frame and f"{symbol}_quote_volume" in frame:
        buy, total = _col(frame, symbol, "taker_buy_quote_volume"), _col(frame, symbol, "quote_volume")
        for k in (1, 4):
            sums = total.rolling(k * h).sum()
            X[f"taker_{k}"] = (buy.rolling(k * h).sum() / sums).where(sums > 0)

    n = 24 * h
    lo, hi = low.rolling(n).min(), high.rolling(n).max()
    X["range_pos"] = ((close - lo) / (hi - lo)).where(hi > lo, 0.5)

    close_time = frame.index + pd.Timedelta(minutes=minutes)
    hours = close_time.hour + close_time.minute / 60.0
    X["hour"] = np.asarray(hours, dtype=float)
    X["to_funding"] = FUNDING_HOURS - np.mod(np.asarray(hours, dtype=float), FUNDING_HOURS)

    if hourly is not None and not hourly.empty and f"{symbol}_funding_rate" in hourly:
        X = X.join(_funding_inputs(frame.index, minutes, hourly, symbol))

    if benchmark and f"{benchmark}_close" in frame:
        rb = np.log(_col(frame, benchmark, "close")).diff()
        X["market_rv"] = _log_vol((rb ** 2).rolling(24 * h).mean())
    return X


def adverse_moves(frame: pd.DataFrame, symbol: str, h: int) -> pd.DataFrame:
    """Long and short adverse move over the next h candles from each close (NaN for the last h)."""
    close, high, low = (_col(frame, symbol, c) for c in ("close", "high", "low"))
    lowest, highest = low.rolling(h).min().shift(-h), high.rolling(h).max().shift(-h)
    return pd.DataFrame({"long": ((close - lowest) / close).clip(lower=0.0),
                         "short": ((highest - close) / close).clip(lower=0.0)}, index=frame.index)


def future_vol(frame: pd.DataFrame, symbol: str, h: int) -> pd.Series:
    """The HAR target: log per-candle Parkinson volatility over the next h candles."""
    pk = parkinson(_col(frame, symbol, "high"), _col(frame, symbol, "low"))
    return _log_vol(pk.rolling(h).mean().shift(-h))


@dataclass
class WindowData:
    """Everything the models need for one window, row for row with the chart."""

    window: Window
    inputs: pd.DataFrame
    adverse: pd.DataFrame
    future_vol: pd.Series
    regime: pd.DataFrame

    @property
    def usable(self) -> np.ndarray:
        """Positions with a HAR forecast (finite HAR inputs)."""
        return np.flatnonzero(np.isfinite(self.inputs[list(HAR_COLUMNS)].to_numpy()).all(axis=1))

    @property
    def trainable(self) -> np.ndarray:
        """Usable positions whose targets are known."""
        known = (np.isfinite(self.adverse.to_numpy()).all(axis=1) & np.isfinite(self.future_vol.to_numpy()))
        return self.usable[known[self.usable]]


def window_data(
    candles: Dict[str, pd.DataFrame],
    symbol: str,
    window: Window,
    benchmark: Optional[str] = None,
    regime_spec: RegimeSpec = RegimeSpec(),
) -> WindowData:
    frame = candles[window.interval]
    regime = regime_frame(frame, symbol, window, regime_spec)
    return WindowData(window, risk_inputs(frame, symbol, window, benchmark, candles.get("1h"), regime_spec, regime),
                      adverse_moves(frame, symbol, window.candles), future_vol(frame, symbol, window.candles), regime)


# ───────── models ─────────

@dataclass(frozen=True)
class RiskSpec:
    alphas: Tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.97)
    bad_case: float = 0.9
    train_share: float = 0.6
    folds: int = 4
    gap_hours: float = 8.0
    min_gain: float = 0.02  # LightGBM must cut the pinball loss by this share
    coverage_band: Tuple[float, float] = (0.07, 0.13)
    # Small on purpose: 200 rounds of 15 leaves overfit (SOL, Oct 2026: bad case exceeded in 13-16%
    # of test windows instead of 10%, pinball 3-7% worse than the baseline).
    rounds: int = 60
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 0.05, "num_leaves": 4, "min_data_in_leaf": 1000, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 7,
        # One thread per model, many models at once (fit_pool): models this small spend more time
        # synchronising threads than boosting (36 s a coin on 10 threads each, see fit_pool)
        "num_threads": 1,
    })

    @property
    def bad_index(self) -> int:
        return self.alphas.index(self.bad_case)

    def gap(self, window: Window) -> int:
        """Candles between the last training reading and the first test reading."""
        return window.candles + int(math.ceil(self.gap_hours * 60 * window.candles / window.minutes))


def exceed_chance(quantiles: np.ndarray, alphas: Tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """P(move ≥ x) from predicted quantiles (rows ascending): linear between (0, 0) and the
    quantiles, an exponential tail fitted to the last two beyond the last. Vectorised over rows."""
    q = np.sort(np.atleast_2d(np.asarray(quantiles, dtype=float)), axis=1)
    a = np.asarray(alphas, dtype=float)
    n = len(q)
    x = np.broadcast_to(np.asarray(x, dtype=float), (n,))
    knots = np.c_[np.zeros(n), q]  # (0, 0) first
    levels = np.r_[0.0, a]
    j = (knots < x[:, None]).sum(axis=1)  # knots[j - 1] < x <= knots[j]
    inside = (j > 0) & (x < q[:, -1])
    rows = np.flatnonzero(inside)
    lo, hi = knots[rows, j[rows] - 1], knots[rows, j[rows]]
    cdf = np.zeros(n)
    cdf[rows] = levels[j[rows] - 1] + (levels[j[rows]] - levels[j[rows] - 1]) * (x[rows] - lo) / (hi - lo)
    out = 1.0 - cdf
    tail = x >= q[:, -1]
    step = q[:, -1] - q[:, -2]
    scale = np.where(step > 0, step / math.log((1 - a[-2]) / (1 - a[-1])), 1e-12)
    out[tail] = (1 - a[-1]) * np.exp(-(x[tail] - q[tail, -1]) / scale[tail])
    return out


class Har:
    """Linear regression of the log future volatility on the HAR inputs."""

    def __init__(self, coef: np.ndarray) -> None:
        self.coef = coef

    @staticmethod
    def _design(inputs: pd.DataFrame) -> np.ndarray:
        X = inputs[list(HAR_COLUMNS)].to_numpy(dtype=float)
        return np.c_[np.ones(len(X)), X]

    @classmethod
    def fit(cls, inputs: pd.DataFrame, target: np.ndarray) -> "Har":
        coef, *_ = np.linalg.lstsq(cls._design(inputs), target, rcond=None)
        return cls(coef)

    def swing(self, inputs: pd.DataFrame, h: int) -> np.ndarray:
        """Forecast volatility over the next h candles as a price fraction."""
        return np.exp(self._design(inputs) @ self.coef) * math.sqrt(h)


@dataclass
class Prediction:
    swing: np.ndarray  # HAR forecast per row (price fraction over the window)
    quantiles: Dict[str, Dict[str, np.ndarray]]  # model -> side -> (rows, alphas), in swing units

    def price(self, model: str, side: str) -> np.ndarray:
        return self.quantiles[model][side] * self.swing[:, None]


_POOL: Optional[ThreadPoolExecutor] = None
_POOL_LOCK = threading.Lock()


def fit_pool() -> ThreadPoolExecutor:
    """Threads for LightGBM fits, one per CPU, shared by every coin: LightGBM releases the GIL while
    it boosts, so single-threaded models run truly in parallel. On 10 cores this took one coin's
    400 models from 36 s (each on all threads) to a few seconds."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4, thread_name_prefix="lgbm")
        return _POOL


class WindowRiskModel:
    """HAR baseline and LightGBM quantiles of the adverse move, both sides, for one window."""

    def __init__(self, spec: RiskSpec = RiskSpec()) -> None:
        self.spec = spec
        self.har: Optional[Har] = None
        self.past: Dict[str, np.ndarray] = {}  # sorted training adverse moves in swing units
        self.boosters: Dict[str, List[Any]] = {}
        self.columns: List[str] = []
        self.typical_swing = float("nan")  # median HAR forecast over the training rows

    def fit(self, data: WindowData, rows: np.ndarray, lgbm: bool = True) -> "WindowRiskModel":
        inputs = data.inputs.iloc[rows]
        self.har = Har.fit(inputs, data.future_vol.to_numpy()[rows])
        swing = self.har.swing(inputs, data.window.candles)
        self.typical_swing = float(np.median(swing))
        self.columns = list(data.inputs.columns)
        X = inputs[self.columns].to_numpy(dtype=float)
        units = {side: data.adverse[side].to_numpy()[rows] / swing for side in SIDES}
        for side in SIDES:
            self.past[side] = np.sort(units[side])
        if lgbm:
            jobs = {(side, a): fit_pool().submit(self._boost, X, units[side], a) for side in SIDES for a in self.spec.alphas}
            self.boosters = {side: [jobs[(side, a)].result() for a in self.spec.alphas] for side in SIDES}
        return self

    def _boost(self, X: np.ndarray, label: np.ndarray, alpha: float) -> Any:
        """One quantile model. Each job builds its own Dataset, so threads share no LightGBM state."""
        import lightgbm as lgb

        params = {**self.spec.lgbm_params, "objective": "quantile", "alpha": alpha}
        return lgb.train(params, lgb.Dataset(X, label, params=params), num_boost_round=self.spec.rounds)

    def predict(self, data: WindowData, rows: np.ndarray) -> Prediction:
        inputs = data.inputs.iloc[rows]
        swing = self.har.swing(inputs, data.window.candles)
        X = inputs[self.columns].to_numpy(dtype=float)
        quantiles: Dict[str, Dict[str, np.ndarray]] = {"har": {}, "lgbm": {}}
        for side in SIDES:
            quantiles["har"][side] = np.tile(np.quantile(self.past[side], self.spec.alphas), (len(rows), 1))
            if side in self.boosters:
                quantiles["lgbm"][side] = np.sort(np.column_stack([b.predict(X) for b in self.boosters[side]]), axis=1)
        return Prediction(swing, quantiles)

    def chance(self, pred: Prediction, model: str, side: str, move: float) -> np.ndarray:
        """P(adverse move ≥ `move`, a price fraction) per row."""
        x = move / pred.swing
        if model == "har":
            past = self.past[side]
            return 1.0 - np.searchsorted(past, x, side="left") / len(past)
        return exceed_chance(pred.quantiles[model][side], self.spec.alphas, x)


# ───────── the honesty check ─────────

def pinball(actual: np.ndarray, quantiles: np.ndarray, alphas: Tuple[float, ...]) -> float:
    """Mean pinball loss over rows and quantiles."""
    diff = actual[:, None] - quantiles
    a = np.asarray(alphas)[None, :]
    return float(np.mean(np.maximum(a * diff, (a - 1.0) * diff)))


def calibration(predicted: np.ndarray, happened: np.ndarray, edges=tuple(np.linspace(0, 1, 11))) -> List[Dict[str, Any]]:
    """Predicted chance against how often it happened, per bucket of the prediction."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (predicted >= lo) & ((predicted < hi) | (hi >= 1.0))
        if inside.any():
            out.append({"from": float(lo), "to": float(hi), "n": int(inside.sum()),
                        "predicted": float(predicted[inside].mean()), "actual": float(happened[inside].mean())})
    return out


def split_positions(trainable: np.ndarray, spec: RiskSpec, gap: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(train, test) positions per fold: test blocks after the first `train_share`, each trained
    on every earlier position at least `gap` candles before the block starts."""
    n = len(trainable)
    start = int(n * spec.train_share)
    block = (n - start) // spec.folds
    folds = []
    for i in range(spec.folds):
        test = trainable[start + i * block: start + (i + 1) * block]
        if not len(test):
            continue
        train = trainable[trainable < test[0] - gap]
        if len(train):
            folds.append((train, test))
    return folds


def walk_forward(data: WindowData, stop: float, liquidation: float, spec: RiskSpec = RiskSpec(),
                 lgbm: bool = True) -> Dict[str, Any]:
    """Out-of-sample check per side: coverage of the bad case, pinball loss of both models, the
    stop-hit chance against what happened, and the model that ships. Without `lgbm` only the
    baseline is fitted and checked (offline mode)."""
    models = ("har", "lgbm") if lgbm else ("har",)
    folds = split_positions(data.trainable, spec, spec.gap(data.window))
    if not folds:
        return {"tests": 0, "folds": 0}
    actual: Dict[str, List[np.ndarray]] = {s: [] for s in SIDES}
    quant: Dict[Tuple[str, str], List[np.ndarray]] = {}
    hit: Dict[Tuple[str, str], List[np.ndarray]] = {}
    liq: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for train, test in folds:
        model = WindowRiskModel(spec).fit(data, train, lgbm)
        pred = model.predict(data, test)
        for side in SIDES:
            actual[side].append(data.adverse[side].to_numpy()[test])
            for m in models:
                quant.setdefault((m, side), []).append(pred.price(m, side))
                hit.setdefault((m, side), []).append(model.chance(pred, m, side, stop))
                liq.setdefault((m, side), []).append(model.chance(pred, m, side, liquidation))

    out: Dict[str, Any] = {"tests": int(sum(len(t) for _, t in folds)), "folds": len(folds),
                           "train_from": len(folds[0][0])}
    lo, hi = spec.coverage_band
    for side in SIDES:
        y = np.concatenate(actual[side])
        side_out: Dict[str, Any] = {}
        for m in models:
            q = np.concatenate(quant[(m, side)])
            p = np.concatenate(hit[(m, side)])
            side_out[m] = {
                "coverage": float(np.mean(y > q[:, spec.bad_index])),
                "pinball_pct": 100.0 * pinball(y, q, spec.alphas),
                "stop_hit_predicted": float(p.mean()),
                "brier": float(np.mean((p - (y >= stop)) ** 2)),
                "liquidation_predicted": float(np.concatenate(liq[(m, side)]).mean()),
            }
        gain = 1.0 - side_out["lgbm"]["pinball_pct"] / side_out["har"]["pinball_pct"] if lgbm else None
        chosen = "lgbm" if gain is not None and gain >= spec.min_gain else "har"
        side_out.update({
            "gain": gain, "model": chosen,
            "checked": lo <= side_out[chosen]["coverage"] <= hi,
            "stop_hit_rate": float(np.mean(y >= stop)),
            "liquidation_rate": float(np.mean(y >= liquidation)),
            "calibration": calibration(np.concatenate(hit[(chosen, side)]), y >= stop),
        })
        out[side] = side_out
    return out


def latest_reading(
    model: WindowRiskModel,
    data: WindowData,
    choice: Dict[str, str],
    stop: float,
    liquidation: float,
    leverage: float,
) -> Dict[str, Any]:
    """Today's numbers at the chart's last closed candle, per side, from the chosen model."""
    pos = data.usable[-1:]
    pred = model.predict(data, pos)
    out: Dict[str, Any] = {"at": data.inputs.index[pos[0]], "swing_pct": 100.0 * float(pred.swing[0])}
    for side in SIDES:
        m = choice.get(side, "har")
        q = pred.price(m, side)[0]
        bad = float(q[model.spec.bad_index])
        out[side] = {
            "model": m, "quantiles_pct": [100.0 * float(v) for v in q],
            "bad_case_pct": 100.0 * bad, "bad_case_margin_pct": 100.0 * bad * leverage,
            "stop_hit_chance": float(model.chance(pred, m, side, stop)[0]),
            "liquidation_chance": float(model.chance(pred, m, side, liquidation)[0]),
            "score": risk_score(float(model.chance(pred, m, side, stop)[0])),
        }
    return out


# ───────── one coin, every window: what the dashboard serves ─────────

# Quantiles sent to the page for the baseline, dense enough that interpolating them (exceed_chance)
# matches its empirical stop-hit chance; LightGBM sends its own `alphas`.
PAGE_ALPHAS: Tuple[float, ...] = tuple(round(float(a), 3) for a in np.r_[np.arange(0.02, 0.985, 0.02), 0.99, 0.995, 0.999])
MIN_TRAINING = 1000  # readings with known targets a window needs before it gets a model


def risk_reason(side: str, vol_ratio: float, funding_pct: Optional[float], regime: pd.Series,
                other_bad_ratio: Optional[float] = None) -> str:
    """The main drivers of one side's risk in a few words."""
    parts = []
    if np.isfinite(vol_ratio) and (vol_ratio >= 1.3 or vol_ratio <= 0.75):
        parts.append(f"volatility {vol_ratio:.1f}× normal")
    if regime.get("state") == "wild":
        parts.append("market wild")
    if funding_pct is not None and np.isfinite(funding_pct):
        if funding_pct >= 90 and side == "long":
            parts.append("funding crowded long")
        elif funding_pct <= 10 and side == "short":
            parts.append("funding crowded short")
    against = {"long": "bearish", "short": "bullish"}[side]
    if regime.get("momentum_label") == against:
        parts.append(f"momentum {against}")
    if other_bad_ratio is not None and np.isfinite(other_bad_ratio) and other_bad_ratio >= 1.15:
        parts.append(f"{other_bad_ratio:.1f}× the {'short' if side == 'long' else 'long'} side's bad case")
    return ", ".join(parts) or "normal conditions"


@dataclass
class WindowState:
    """A window's fitted model, its walk-forward check and the prediction at the latest close."""

    model: WindowRiskModel
    check: Dict[str, Any]
    choice: Dict[str, str]
    pred: Optional[Prediction] = None
    at: Optional[pd.Timestamp] = None
    regime: Optional[pd.Series] = None
    funding_pct: Optional[float] = None
    record: Optional[Dict[str, Any]] = None  # regime.trend_record, set at training

    def update(self, data: WindowData) -> None:
        pos = data.usable[-1:]
        if not len(pos):
            return
        self.pred = self.model.predict(data, pos)
        self.at = data.inputs.index[pos[0]]
        self.regime = data.regime.iloc[pos[0]]
        self.funding_pct = float(data.inputs["funding_pct"].iloc[pos[0]]) if "funding_pct" in data.inputs else None

    def view(self, leverage: float, stop: float, liquidation: float, spec: RiskSpec) -> Dict[str, Any]:
        pred, reg = self.pred, self.regime
        swing = float(pred.swing[0])
        out: Dict[str, Any] = {
            "at": self.at, "swing_pct": 100.0 * swing,
            "momentum": reg["momentum"], "momentum_label": reg["momentum_label"],
            "efficiency": reg["efficiency"], "vol_pct": reg["vol_pct"],
            "volatility": reg["volatility_label"], "state": reg["state"],
            "trend_record": self.record,
        }
        bad = {s: float(pred.price(self.choice[s], s)[0][spec.bad_index]) for s in SIDES}
        lo, hi = spec.coverage_band
        for side in SIDES:
            m, other = self.choice[side], "short" if side == "long" else "long"
            check = self.check.get(side, {})
            coverage = check.get(m, {}).get("coverage")
            hit = float(self.model.chance(pred, m, side, stop)[0])
            if m == "har":
                alphas = PAGE_ALPHAS
                moves = np.quantile(self.model.past[side], alphas) * swing
            else:
                alphas, moves = spec.alphas, pred.price(m, side)[0]
            out[side] = {
                "model": m, "checked": coverage is not None and lo <= coverage <= hi, "coverage": coverage,
                "bad_case_pct": 100.0 * bad[side], "bad_case_margin_pct": 100.0 * bad[side] * leverage,
                "stop_hit_chance": hit, "score": risk_score(hit),
                "liquidation_chance": float(self.model.chance(pred, m, side, liquidation)[0]),
                "reason": risk_reason(side, swing / self.model.typical_swing, self.funding_pct, reg,
                                      bad[side] / bad[other] if bad[other] > 0 else None),
                "quantiles": {"alphas": list(alphas), "moves_pct": [100.0 * float(v) for v in moves]},
            }
        return out


class CoinRisk:
    """Risk models for every window of one coin: trained (with the walk-forward check) when the
    history loads and once a day, scored at every close of a window's chart."""

    def __init__(self, symbol: str, benchmark: Optional[str], spec: RiskSpec = RiskSpec(),
                 regime_spec: RegimeSpec = RegimeSpec()) -> None:
        self.symbol, self.benchmark = symbol, benchmark
        self.spec, self.regime_spec = spec, regime_spec
        self.windows: Dict[str, WindowState] = {}
        self.trained_at: Optional[pd.Timestamp] = None

    def _data(self, candles: Dict[str, pd.DataFrame], window: Window) -> Optional[WindowData]:
        frame = candles.get(window.interval)
        if frame is None or frame.empty:
            return None
        return window_data(candles, self.symbol, window, self.benchmark, self.regime_spec)

    def train(self, candles: Dict[str, pd.DataFrame], stop: float, liquidation: float, lgbm: bool = True,
              progress: Optional[Any] = None) -> "CoinRisk":
        """Check and fit every window the candles allow. `stop` and `liquidation` (price
        fractions) only shape the check's stop-hit diagnostics; `progress(window, step, steps)`."""
        from deltarisk.stats.regime import WINDOWS

        windows: Dict[str, WindowState] = {}
        for i, window in enumerate(WINDOWS, start=1):
            if progress is not None:
                progress(window.name, i, len(WINDOWS))
            data = self._data(candles, window)
            if data is None or len(data.trainable) < MIN_TRAINING:
                continue
            check = walk_forward(data, stop, liquidation, self.spec, lgbm)
            choice = {s: check.get(s, {}).get("model", "har") for s in SIDES}
            state = WindowState(WindowRiskModel(self.spec).fit(data, data.trainable, lgbm), check, choice)
            state.record = trend_record(candles[window.interval], self.symbol, window, data.regime, ROUND_TRIP_PCT)
            state.update(data)
            windows[window.name] = state
        self.windows = windows
        self.trained_at = pd.Timestamp.now(tz="UTC")
        return self

    def score(self, candles: Dict[str, pd.DataFrame], intervals: Optional[List[str]] = None) -> None:
        """Predict at the latest close of the windows on `intervals` (default every chart)."""
        from deltarisk.stats.regime import WINDOW_BY_NAME

        for name, state in self.windows.items():
            window = WINDOW_BY_NAME[name]
            if intervals is not None and window.interval not in intervals:
                continue
            data = self._data(candles, window)
            if data is not None:
                state.update(data)

    def view(self, leverage: float, mmr: float, stop_pct: float) -> Dict[str, Any]:
        """Per window: momentum and state, and for each side the bad case, stop-hit and
        liquidation chances, score, reason, model, whether it passed the check, and the quantiles
        the page re-reads for another stop."""
        stop, liquidation = stop_move(stop_pct, leverage), liquidation_move(leverage, mmr)
        return {name: state.view(leverage, stop, liquidation, self.spec)
                for name, state in self.windows.items() if state.pred is not None}


# ───────── trained models on disk ─────────

MODEL_CACHE_VERSION = 1  # bump when CoinRisk's contents change, so old files are retrained


def _model_key(risk: CoinRisk) -> Dict[str, Any]:
    return {"version": MODEL_CACHE_VERSION, "symbol": risk.symbol, "benchmark": risk.benchmark,
            "spec": repr(risk.spec), "regime_spec": repr(risk.regime_spec)}


def save_models(path: Path, risk: CoinRisk) -> None:
    """Keep a coin's trained models on disk, so a restart within a day can skip training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump({**_model_key(risk), "risk": risk}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as err:
        tmp.unlink(missing_ok=True)
        logger.warning(f"Could not save risk models to {path}: {err}")


def load_models(path: Path, symbol: str, benchmark: Optional[str], max_age: pd.Timedelta,
                spec: RiskSpec = RiskSpec(), regime_spec: RegimeSpec = RegimeSpec()) -> Optional[CoinRisk]:
    """Models saved by save_models for this coin, benchmark and specs, trained less than `max_age`
    ago; None otherwise (or when the file can't be read). The file is this app's own: pickle."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        risk: CoinRisk = saved["risk"]
        want = {"version": MODEL_CACHE_VERSION, "symbol": symbol, "benchmark": benchmark,
                "spec": repr(spec), "regime_spec": repr(regime_spec)}
        if {k: saved.get(k) for k in want} != want or risk.trained_at is None:
            return None
        if pd.Timestamp.now(tz="UTC") - risk.trained_at >= max_age:
            return None
        return risk
    except Exception as err:
        logger.warning(f"Ignoring unreadable risk models {path}: {err}")
        return None
