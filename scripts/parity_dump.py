"""Throwaway (GO-REWRITE-PLAN.md, phase 0; deleted in phase 9): run the Python code on fixed inputs
and save its answers under internal/testdata/parity/, so the Go port can be checked against them.

    uv run python scripts/parity_dump.py            # reads ~/.deltarisk/cache (or $DELTARISK_HOME)

Inputs: ETH and its benchmark (BTC) cut from the candle cache to fixed spans ending at END_MS, and
ETH's settled funding rates, written in the cache's own Parquet format under parity/cache/ so the
Go cache reader loads them. Everything else is read back through those files (BinancePublicClient
with the caches, no network). Times are int64 ms (open times unless a name says otherwise); NaN
and ±Inf are null in JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np
import pandas as pd

from deltarisk.backtest import BacktestConfig, backtest_candles, backtest_wedge_candles
from deltarisk.features.setups import SetupSpec, find_setups
from deltarisk.features.wedges import WedgeSpec, find_wedge_setups
from deltarisk.ingestion.binance import BinanceMarket, BinancePublicClient, interval_to_ms
from deltarisk.ingestion.cache import CandleCache, FundingCache
from deltarisk.ingestion.factory import resolve_benchmark
from deltarisk.ingestion.storage import align_cross_asset_data
from deltarisk.labels.setups import setup_outcomes
from deltarisk.labels.wedges import wedge_outcomes
from deltarisk.paths import resolve
from deltarisk.stats.regime import WINDOWS, regime_frame, trend_record
from deltarisk.stats.risk import (
    HAR_COLUMNS, PAGE_ALPHAS, ROUND_TRIP_PCT, SIDES, TIERS, CoinRisk, RiskSpec, WindowRiskModel,
    exceed_chance, liquidation_move, stop_move, walk_forward, window_data,
)
from deltarisk.stats.signals import funding_view

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "internal" / "testdata" / "parity"
SYMBOL = "B-ETH_USDT"
END_MS = 1_792_267_200_000  # 2026-10-17 20:00 UTC: an hour inside the cache spans of 2026-10-18
DAYS = {"5m": 20, "15m": 40, "1h": 120}  # each window needs 96 windows of warm-up + 1,000 readings (8h: 74 days)
STOPS = (5.0, 7.5, 10.0)
LEVERAGES = (("megacap", 20.0), ("altcoin", 10.0))


# ───────── JSON ─────────

def plain(obj: Any) -> Any:
    """JSON-ready copy: numpy scalars and arrays to Python, timestamps to ms, NaN/±Inf to None."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [plain(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return int(obj.value // 1_000_000)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def write_json(name: str, obj: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(obj), indent=1, allow_nan=False) + "\n")


def write_frame(name: str, frame: pd.DataFrame) -> None:
    """A frame indexed by open time as Parquet: `time` (int64 ms) then its columns as they are."""
    out = frame.copy()
    out.insert(0, "time", (out.index.as_unit("ms").asi8 if isinstance(out.index, pd.DatetimeIndex)
                           else out.index.to_numpy()).astype("int64"))
    for c in out.columns:
        if out[c].dtype == object:  # labels: str or None
            out[c] = out[c].astype("string")
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")


def records(frame: pd.DataFrame) -> list:
    """Rows of a frame as dicts, a datetime index as `time` (ms)."""
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.insert(0, "time", out.index.as_unit("ms").asi8)
    return [plain(r) for r in out.reset_index(drop=True).to_dict(orient="records")]


# ───────── inputs: cut the user's cache ─────────

def cut_cache(benchmark: str) -> Dict[str, Any]:
    src_candles, src_funding = CandleCache(), FundingCache()
    dst_candles = CandleCache(OUT / "cache" / "candles")
    dst_funding = FundingCache(OUT / "cache" / "funding")
    spans = {}
    for market in (SYMBOL, benchmark):
        name = market.split("-")[1].replace("_", "")
        for iv, days in DAYS.items():
            step = interval_to_ms(iv)
            last_open = END_MS // step * step - step
            first_open = last_open - (days * 1440 * 60_000 // step - 1) * step
            have, span = src_candles.load(name, iv)
            if span is None or span[0] > first_open or span[1] < last_open:
                raise SystemExit(f"{name} {iv}: the cache doesn't cover {first_open}..{last_open} (has {span})")
            rows = have[(have["time"] >= first_open) & (have["time"] <= last_open)]
            dst_candles.store(name, iv, step, rows, first_open, last_open)
            spans[f"{name}_{iv}"] = {"covered_from": first_open, "covered_to": last_open, "rows": len(rows)}
    loaded = src_funding.load("ETHUSDT")
    if loaded is None or loaded[1] < END_MS:
        raise SystemExit("the ETHUSDT funding cache is missing or older than END_MS")
    _, fetched_at, events = loaded
    start = (END_MS - (DAYS["1h"] + 3) * 86_400_000) // 86_400_000 * 86_400_000
    kept = events[(events["time"] >= start) & (events["time"] <= END_MS)].reset_index(drop=True)
    dst_funding.store("ETHUSDT", start, END_MS, kept)
    spans["funding_ETHUSDT"] = {"covered_from": start, "fetched_at": END_MS, "rows": len(kept),
                                "source_fetched_at": fetched_at}
    return spans


def offline_client() -> BinancePublicClient:
    """A client on the fixture caches that never needs the exchange list."""
    client = BinancePublicClient(candle_cache=CandleCache(OUT / "cache" / "candles"),
                                 funding_cache=FundingCache(OUT / "cache" / "funding"))
    client._require = lambda symbol: BinanceMarket(symbol.split("-")[1].replace("_", "").replace("INR", "USDT"))
    return client


# ───────── outputs ─────────

def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    benchmark = resolve_benchmark(SYMBOL)
    print(f"{SYMBOL} against {benchmark}, from {resolve('cache')}")
    spans = cut_cache(benchmark)

    client = offline_client()
    raw: Dict[str, Dict[str, pd.DataFrame]] = {}
    candles: Dict[str, pd.DataFrame] = {}
    for iv in DAYS:
        bars = spans[f"ETHUSDT_{iv}"]["rows"]
        target, _ = client.fetch_candle_history(SYMBOL, iv, bars, end_time_ms=END_MS, funding=iv == "1h")
        bench, _ = client.fetch_candle_history(benchmark, iv, bars, end_time_ms=END_MS)
        raw[iv] = {"target": target, "bench": bench}
        candles[iv] = align_cross_asset_data(target, {benchmark: bench}, freq=f"{interval_to_ms(iv) // 60_000}min")
        write_frame(f"aligned/{iv}.parquet", candles[iv])
        print(f"  {iv}: {len(candles[iv]):,} aligned candles")

    # The 1h chart's funding_rate column before alignment (funding_known_at)
    write_frame("funding_known_at.parquet",
                raw["1h"]["target"].set_index("timestamp")[["funding_rate"]])

    tier = TIERS["megacap"]
    stop, liq = stop_move(5.0, tier.leverage), liquidation_move(tier.leverage, tier.mmr)
    spec = RiskSpec()
    windows_meta = {}
    for w in WINDOWS:
        data = window_data(candles, SYMBOL, w, benchmark)
        frame = data.inputs.copy()
        frame["adverse_long"] = data.adverse["long"]
        frame["adverse_short"] = data.adverse["short"]
        frame["future_vol"] = data.future_vol
        for c in data.regime.columns:
            frame[f"regime_{c}"] = data.regime[c]
        write_frame(f"windows/{w.name}.parquet", frame)

        check = walk_forward(data, stop, liq, spec, lgbm=False)
        write_json(f"walk_forward/{w.name}.json", {"stop": stop, "liquidation": liq, **check})
        write_json(f"trend_record/{w.name}.json",
                   trend_record(candles[w.interval], SYMBOL, w, data.regime, ROUND_TRIP_PCT))

        meta: Dict[str, Any] = {"interval": w.interval, "candles": w.candles, "gap": spec.gap(w),
                                "rows": len(frame), "usable": len(data.usable), "trainable": len(data.trainable),
                                "input_columns": list(data.inputs.columns)}
        if len(data.trainable):
            model = WindowRiskModel(spec).fit(data, data.trainable, lgbm=False)
            meta["model"] = {"har_coef": model.har.coef, "typical_swing": model.typical_swing,
                             "past_quantiles": {s: np.quantile(model.past[s], PAGE_ALPHAS) for s in SIDES}}
        windows_meta[w.name] = meta
    write_json("windows.json", {"har_columns": HAR_COLUMNS, "windows": windows_meta})

    risk = CoinRisk(SYMBOL, benchmark).train(candles, stop, liq, lgbm=False)
    views = []
    for tier_name, leverage in LEVERAGES:
        mmr = TIERS[tier_name].mmr
        for stop_pct in STOPS:
            views.append({"tier": tier_name, "leverage": leverage, "mmr": mmr, "stop_pct": stop_pct,
                          "stop": stop_move(stop_pct, leverage), "liquidation": liquidation_move(leverage, mmr),
                          "view": risk.view(leverage, mmr, stop_pct)})
    write_json("coin_risk_views.json", {"symbol": SYMBOL, "benchmark": benchmark, "train_stop": stop,
                                        "train_liquidation": liq, "views": views})
    write_json("funding_view.json", funding_view(candles, SYMBOL))

    # exceed_chance on a grid of quantile rows (ties, zeros, a flat tail) and moves
    rng = np.random.default_rng(11)
    alphas = spec.alphas
    rows = [np.sort(rng.gamma(2.0, 0.6, len(alphas))) for _ in range(6)]
    rows += [np.array([0.0, 0.0, 0.5, 1.0, 1.0]), np.array([0.2, 0.4, 0.4, 0.9, 2.0]), np.array([0.3, 0.6, 0.9, 1.2, 1.2])]
    xs = [0.0, 1e-9, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    write_json("exceed_chance.json", {
        "alphas": alphas, "page_alphas": PAGE_ALPHAS, "quantiles": rows, "x": xs,
        "chance": [[float(exceed_chance(q, alphas, np.array([x]))[0]) for x in xs] for q in rows],
    })

    # Backtests: one config each, on plain candles like the CLI loads them
    def plain_candles(iv: str) -> pd.DataFrame:
        df = raw[iv]["target"]
        return df.set_index(pd.DatetimeIndex(df["timestamp"]))[["open", "high", "low", "close", "volume"]].astype(float)

    fvg_spec = SetupSpec.for_charts("15m", "1h", targets=(2.0, 3.0, 4.0))
    fvg_cfg = BacktestConfig(target=3.0, grades=("A+", "B"), risk_pct=1.0, max_leverage=None)
    entry, context = plain_candles("15m"), plain_candles("1h")
    setups = find_setups(entry, context, fvg_spec)
    outcomes = setup_outcomes(setups, entry["high"], entry["low"], entry["close"], fvg_spec)
    result = backtest_candles(entry, context, fvg_spec, fvg_cfg)
    write_json("backtest_fvg.json", {
        "config": {"entry": "15m", "context": "1h", "targets": fvg_spec.targets, "target": fvg_cfg.target,
                   "grades": fvg_cfg.grades, "risk_pct": fvg_cfg.risk_pct, "max_leverage": fvg_cfg.max_leverage},
        "setups": records(setups), "outcomes": records(outcomes),
        "summary": result["summary"], "trades": records(result["trades"]), "monthly": result["monthly"],
        "sides": result["sides"], "shapes": result["shapes"],
    })
    print(f"  fvg: {len(setups)} setups, {result['summary']['trades']} trades")

    wedge_spec = WedgeSpec.for_chart("15m", targets=("measured", 2.0, 2.5), breakeven_r=WedgeSpec.breakeven_r)
    wedge_cfg = BacktestConfig(target=2.0, grades=("A+", "B"), risk_pct=1.0, max_leverage=5.0)
    bars = plain_candles("15m")
    setups = find_wedge_setups(bars, wedge_spec)
    outcomes = wedge_outcomes(setups, bars["high"], bars["low"], bars["close"], wedge_spec)
    result = backtest_wedge_candles(bars, wedge_spec, wedge_cfg)
    write_json("backtest_wedge.json", {
        "config": {"chart": "15m", "targets": wedge_spec.targets, "breakeven_r": wedge_spec.breakeven_r,
                   "target": wedge_cfg.target, "grades": wedge_cfg.grades, "risk_pct": wedge_cfg.risk_pct,
                   "max_leverage": wedge_cfg.max_leverage},
        "setups": records(setups), "funnel": dict(setups.attrs["funnel"]), "outcomes": records(outcomes),
        "summary": result["summary"], "trades": records(result["trades"]), "monthly": result["monthly"],
        "sides": result["sides"], "shapes": result["shapes"],
    })
    print(f"  wedge: {len(setups)} setups, {result['summary']['trades']} trades")

    write_json("inputs.json", {"symbol": SYMBOL, "benchmark": benchmark, "end_ms": END_MS, "days": DAYS,
                               "spans": spans})
    client.close()
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Wrote {OUT.relative_to(ROOT)} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
