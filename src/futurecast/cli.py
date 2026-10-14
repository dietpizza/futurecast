"""Command Line Interface for futurecast: Air-Gapped Residual Alpha Forecaster."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import time
import typer
from rich.console import Console
from rich.table import Table
import numpy as np
import pandas as pd

from futurecast.config import load_config
from futurecast.ingestion.binance import BinancePublicClient, generate_synthetic_data
from futurecast.ingestion.coindcx import CoinDCXPublicClient
from futurecast.ingestion.storage import ParquetStorage, align_cross_asset_data
from futurecast.ingestion.cache import DataCache
from futurecast.ingestion.factory import (
    detect_exchange,
    resolve_benchmarks,
    get_exchange_client,
    sanitize_symbol_for_filename,
)
from futurecast.features.pipeline import FeaturePipeline
from futurecast.labels.residual_alpha import compute_residual_alpha_target
from futurecast.validation.purged_cv import PurgedWalkForwardCV
from futurecast.validation.metrics import compute_regression_metrics, compute_theoretical_long_short_sharpe
from futurecast.models.trainer import ModelTrainer
from futurecast.models.optimizer import HyperparameterOptimizer
from futurecast.models.ensemble import MultiModelEnsemble
from futurecast.engine.forecaster import PureForecastEngine

app = typer.Typer(
    name="futurecast",
    help="Air-gapped cross-asset LightGBM residual alpha forecasting engine.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    days: int = typer.Option(30, help="Number of historical days to ingest"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate realistic synthetic data for offline testing"),
    config_dir: str = typer.Option("config", help="Configuration directory"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass local .cache/ folder and force fresh download"),
) -> None:
    """Ingest 1-minute market data for target and benchmarks (Strictly Read-Only)."""
    cfg = load_config(config_dir)
    storage = ParquetStorage(cfg.assets.storage.raw_dir.split("/")[0])
    target = cfg.assets.target_asset
    benchmarks = cfg.assets.benchmark_assets

    console.print(f"[bold green]Starting ingestion for {target} and benchmarks: {benchmarks}[/bold green]")
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=days)

    if synthetic:
        console.print("[yellow]Generating realistic synthetic market data (offline mode)...[/yellow]")
        periods = days * 1440
        # Generate target (high beta)
        doge_df = generate_synthetic_data(target, start=start_time, periods=periods, base_price=0.14, volatility=0.85, seed=101)
        storage.save_raw(doge_df, target)

        # Generate benchmarks
        bench_dfs = {}
        btc_df = generate_synthetic_data("BTCUSDT", start=start_time, periods=periods, base_price=68000.0, volatility=0.45, seed=102)
        storage.save_raw(btc_df, "BTCUSDT")
        bench_dfs["BTCUSDT"] = btc_df

        eth_df = generate_synthetic_data("ETHUSDT", start=start_time, periods=periods, base_price=2600.0, volatility=0.55, seed=103)
        storage.save_raw(eth_df, "ETHUSDT")
        bench_dfs["ETHUSDT"] = eth_df
    else:
        cache = DataCache(
            cache_dir=cfg.assets.storage.cache_dir,
            enabled=cfg.assets.storage.use_cache and not no_cache,
        )
        if cache.enabled:
            console.print(f"[cyan]Using local cache directory: {cache.cache_dir}[/cyan]")

        client = BinancePublicClient(base_url=cfg.assets.data_source.base_url, cache=cache)
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        console.print(f"Fetching public REST klines from {cfg.assets.data_source.base_url}...")
        doge_df = client.fetch_klines(target, start_time_ms=start_ms, end_time_ms=end_ms, limit=1500)
        if doge_df.empty:
            console.print(f"[red]No data returned for {target}. Check connection or use --synthetic.[/red]")
            raise typer.Exit(1)
        storage.save_raw(doge_df, target)

        bench_dfs = {}
        for b in benchmarks:
            b_df = client.fetch_klines(b, start_time_ms=start_ms, end_time_ms=end_ms, limit=1500)
            storage.save_raw(b_df, b)
            bench_dfs[b] = b_df
        client.close()

    # Align to master UTC grid
    aligned = align_cross_asset_data(doge_df, bench_dfs)
    storage.save_processed(aligned, name="aligned_1m_data")
    console.print(f"[bold green]Successfully aligned {len(aligned)} continuous 1m bars.[/bold green]")


@app.command()
def features(
    config_dir: str = typer.Option("config", help="Configuration directory"),
    audit_stationarity: bool = typer.Option(True, "--audit/--no-audit", help="Run ADF stationarity test"),
) -> None:
    """Compute 35 stationary features across lead-lag, volume, volatility, and derivatives."""
    cfg = load_config(config_dir)
    storage = ParquetStorage(cfg.assets.storage.raw_dir.split("/")[0])

    aligned = storage.load_processed("aligned_1m_data")
    pipe = FeaturePipeline(
        config=cfg.features,
        target_symbol=cfg.assets.target_asset,
        primary_benchmark=cfg.assets.benchmark_assets[0],
        secondary_benchmark=cfg.assets.benchmark_assets[1],
    )

    feat_df = pipe.generate_features(aligned)
    storage.save_processed(feat_df, name="features_35")
    console.print(f"[bold green]Computed {feat_df.shape[1]} features across {len(feat_df)} bars.[/bold green]")

    if audit_stationarity:
        console.print("[cyan]Running Augmented Dickey-Fuller (ADF) stationarity test...[/cyan]")
        audit = pipe.run_stationarity_audit(feat_df, sample_size=min(5000, len(feat_df)))
        table = Table(title="ADF Stationarity Audit", show_header=True)
        table.add_column("Feature Name", style="cyan")
        table.add_column("ADF Stat", style="magenta")
        table.add_column("P-Value", style="green")
        table.add_column("Stationary?", style="bold")

        for f_name, res in list(audit.items())[:15]:
            stat_style = "[green]YES[/green]" if res["is_stationary"] else "[red]NO[/red]"
            table.add_row(f_name, f"{res['adf_stat']:.3f}", f"{res['p_value']:.4e}", stat_style)
        console.print(table)


@app.command()
def validate(
    config_dir: str = typer.Option("config", help="Configuration directory"),
) -> None:
    """Execute Purged & Embargoed Walk-Forward Cross-Validation."""
    cfg = load_config(config_dir)
    storage = ParquetStorage(cfg.assets.storage.raw_dir.split("/")[0])

    aligned = storage.load_processed("aligned_1m_data")
    features = storage.load_processed("features_35")

    target = compute_residual_alpha_target(
        aligned_df=aligned,
        features_df=features,
        target_symbol=cfg.assets.target_asset,
        benchmark_symbol=cfg.assets.benchmark_assets[0],
        horizon_minutes=cfg.features.target.horizon_min,
    )

    common_idx = features.dropna().index.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]

    # Convert days to hours/bars if dataset is small for demonstration
    total_days = (X.index.max() - X.index.min()).days
    train_days = min(cfg.validation.train_window_days, max(5, int(total_days * 0.5)))
    test_days = min(cfg.validation.test_window_days, max(2, int(total_days * 0.2)))
    step_days = min(cfg.validation.step_window_days, max(2, int(total_days * 0.2)))

    cv = PurgedWalkForwardCV(
        train_window_days=train_days,
        test_window_days=test_days,
        step_window_days=step_days,
        purge_window_minutes=cfg.validation.purge_window_minutes,
        embargo_window_minutes=cfg.validation.embargo_window_minutes,
    )

    table = Table(title="Purged Walk-Forward CV Results", show_header=True)
    table.add_column("Fold", style="cyan")
    table.add_column("Train Range", style="white")
    table.add_column("Test Range", style="white")
    table.add_column("Spearman IC", style="magenta")
    table.add_column("Hit Rate", style="green")
    table.add_column("Theoretical Sharpe", style="yellow")

    fold_ics = []
    splits = list(cv.split(X))
    if not splits:
        console.print("[red]Not enough data to form walk-forward folds. Ingest more history.[/red]")
        raise typer.Exit(1)

    for split in splits:
        X_tr = X.iloc[split.train_indices]
        y_tr = y.iloc[split.train_indices]
        X_te = X.iloc[split.test_indices]
        y_te = y.iloc[split.test_indices]

        trainer = ModelTrainer(cfg.model)
        trainer.train(X_tr, y_tr, X_val=X_te, y_val=y_te)

        preds = trainer.predict(X_te)
        metrics = compute_regression_metrics(preds, y_te)
        sharpe = compute_theoretical_long_short_sharpe(pd.Series(preds, index=X_te.index), y_te)

        fold_ics.append(metrics["spearman_ic"])
        table.add_row(
            str(split.fold),
            f"{split.train_start.strftime('%m-%d')} -> {split.train_end.strftime('%m-%d')}",
            f"{split.test_start.strftime('%m-%d')} -> {split.test_end.strftime('%m-%d')}",
            f"{metrics['spearman_ic']:+.4f}",
            f"{metrics['hit_rate']*100:.1f}%",
            f"{sharpe:.2f}",
        )

    console.print(table)
    mean_ic = float(np.mean(fold_ics))
    ic_ir = mean_ic / (float(np.std(fold_ics)) + 1e-8)
    console.print(f"[bold]Summary: Mean Out-of-Sample IC = {mean_ic:+.4f} | IC IR = {ic_ir:.2f}[/bold]")


@app.command()
def train(
    config_dir: str = typer.Option("config", help="Configuration directory"),
    output_model: str = typer.Option("models/lightgbm_doge_residual_alpha.txt", help="Path to save trained model"),
) -> None:
    """Train production LightGBM model and inspect top feature importances."""
    cfg = load_config(config_dir)
    storage = ParquetStorage(cfg.assets.storage.raw_dir.split("/")[0])

    aligned = storage.load_processed("aligned_1m_data")
    features = storage.load_processed("features_35")

    target = compute_residual_alpha_target(
        aligned_df=aligned,
        features_df=features,
        target_symbol=cfg.assets.target_asset,
        benchmark_symbol=cfg.assets.benchmark_assets[0],
        horizon_minutes=cfg.features.target.horizon_min,
    )

    common_idx = features.dropna().index.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]

    # Split last 20% for validation early stopping
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    trainer = ModelTrainer(cfg.model)
    trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)
    trainer.save(output_model)

    val_metrics = trainer.evaluate(X_val, y_val)
    console.print(f"[bold green]Validation IC: {val_metrics['spearman_ic']:+.4f} | RMSE: {val_metrics['rmse']:.6f}[/bold green]")

    importances = trainer.get_feature_importance("gain")
    table = Table(title="Top 10 Feature Importances (Gain)", show_header=True)
    table.add_column("Rank", style="cyan")
    table.add_column("Feature Name", style="magenta")
    table.add_column("Gain", style="green")

    for i, (name, val) in enumerate(importances.head(10).items(), 1):
        table.add_row(str(i), name, f"{val:.1f}")
    console.print(table)


import asyncio
from futurecast.publisher.server import LocalSignalStreamServer
from futurecast.publisher.live_dashboard import LiveSignalDashboard
from futurecast.publisher.broadcaster import SignalBroadcaster
from futurecast.ingestion.streamer import ContinuousKlineStreamer
from rich.live import Live


async def _run_live_stream(
    cfg,
    trainer: ModelTrainer,
    pipe: FeaturePipeline,
    aligned_history: pd.DataFrame,
    ws_server_enabled: bool,
    ws_port: int,
    dashboard_enabled: bool,
    max_bars: Optional[int],
    poll_interval: float,
    exchange: str = "binance",
) -> None:
    ws_server = None
    if ws_server_enabled:
        ws_server = LocalSignalStreamServer(port=ws_port)
        await ws_server.start()
        console.print(f"[bold green]Local Signal WebSocket server running at ws://127.0.0.1:{ws_port}[/bold green]")

    broadcaster = SignalBroadcaster(
        output_dir=cfg.forecasting.output_dir,
        ws_server=ws_server,
        render_terminal=not dashboard_enabled,
    )

    engine = PureForecastEngine(
        config=cfg,
        trainer=trainer,
        feature_pipeline=pipe,
        broadcaster=broadcaster,
        min_buffer_size=120,
    )

    cache = DataCache(cache_dir=cfg.assets.storage.cache_dir, enabled=cfg.assets.storage.use_cache)
    client = get_exchange_client(exchange, cache=cache)

    # Warmup engine buffer with available history or fetch recent 150 bars from exchange
    if aligned_history.empty:
        console.print(f"[cyan]Fetching recent 150 bars from {exchange.upper()} to warm up features...[/cyan]")
        try:
            doge_df = client.fetch_klines(cfg.assets.target_asset, limit=150)
            bench_dfs = {b: client.fetch_klines(b, limit=150) for b in cfg.assets.benchmark_assets}
            aligned_history = align_cross_asset_data(doge_df, bench_dfs)
        finally:
            client.close()

    dashboard = LiveSignalDashboard(
        target_symbol=cfg.assets.target_asset,
        benchmark_symbol=cfg.assets.benchmark_assets[0],
    )

    # Pre-populate dashboard immediately with the latest available closed bar
    if not aligned_history.empty and len(aligned_history) >= 60:
        warmup_slice = aligned_history.tail(150)
        engine.feed_bars(warmup_slice.iloc[:-1])
        initial_event = engine.feed_bars(warmup_slice.iloc[[-1]])
        if initial_event is not None:
            dashboard.update(initial_event, engine.drift_monitor.get_rolling_metrics())
            console.print(f"[cyan]Initialized signal on bar @ {initial_event.timestamp.strftime('%H:%M:%S UTC')}[/cyan]")

    streamer = ContinuousKlineStreamer(
        target_symbol=cfg.assets.target_asset,
        benchmark_symbols=cfg.assets.benchmark_assets,
        base_url=client.base_url,
        poll_interval_sec=poll_interval,
        exchange=exchange,
    )
    if not aligned_history.empty:
        streamer.last_processed_timestamp = aligned_history.index[-1]

    console.print(f"[bold green]Listening for live closed 1m bars from {exchange.upper()} Public API...[/bold green]")

    if dashboard_enabled:
        with Live(dashboard.render(), refresh_per_second=2, screen=False) as live:
            async for bar in streamer.stream_closed_bars(max_bars=max_bars):
                event = engine.feed_bars(bar)
                if event is not None:
                    dashboard.update(event, engine.drift_monitor.get_rolling_metrics())
                    live.update(dashboard.render())
    else:
        async for bar in streamer.stream_closed_bars(max_bars=max_bars):
            event = engine.feed_bars(bar)

    if ws_server is not None:
        await ws_server.stop()


async def _run_replay_stream(
    cfg,
    trainer: ModelTrainer,
    pipe: FeaturePipeline,
    aligned: pd.DataFrame,
    replay_bars: int,
    continuous: bool,
    ws_server_enabled: bool,
    ws_port: int,
    dashboard_enabled: bool,
    delay_sec: float,
) -> None:
    ws_server = None
    if ws_server_enabled:
        ws_server = LocalSignalStreamServer(port=ws_port)
        await ws_server.start()
        console.print(f"[bold green]Local Signal WebSocket server running at ws://127.0.0.1:{ws_port}[/bold green]")

    broadcaster = SignalBroadcaster(
        output_dir=cfg.forecasting.output_dir,
        ws_server=ws_server,
        render_terminal=not dashboard_enabled,
    )

    engine = PureForecastEngine(
        config=cfg,
        trainer=trainer,
        feature_pipeline=pipe,
        broadcaster=broadcaster,
        min_buffer_size=120,
    )

    dashboard = LiveSignalDashboard(
        target_symbol=cfg.assets.target_asset,
        benchmark_symbol=cfg.assets.benchmark_assets[0],
    )

    # Initial warmup
    warmup_slice = aligned.iloc[:-replay_bars] if len(aligned) > replay_bars else aligned
    engine.feed_bars(warmup_slice)

    async def _feed_loop(live_display=None):
        nonlocal engine, dashboard
        n_bars = len(aligned)
        start_idx = max(0, n_bars - replay_bars)

        while True:
            for idx in range(start_idx, n_bars):
                next_bar = aligned.iloc[[idx]]
                event = engine.feed_bars(next_bar)
                if event is not None and live_display is not None:
                    dashboard.update(event, engine.drift_monitor.get_rolling_metrics())
                    live_display.update(dashboard.render())
                await asyncio.sleep(delay_sec)

            if not continuous:
                break

    if dashboard_enabled:
        with Live(dashboard.render(), refresh_per_second=4, screen=False) as live:
            await _feed_loop(live)
    else:
        await _feed_loop(None)

    if ws_server is not None:
        await ws_server.stop()


@app.command("build")
def build(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol from CoinDCX (e.g. B-DOGE_USDT, B-SOL_USDT, I-BTC_INR) or Binance (e.g. DOGEUSDT)"),
    exchange: Optional[str] = typer.Option(None, "--exchange", "-e", help="Exchange ('coindcx' or 'binance'). Auto-detected by default."),
    days: int = typer.Option(14, "--days", "-d", help="Historical days to ingest (default: 14)"),
    config_dir: str = typer.Option("config", help="Configuration directory"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass local .cache/"),
) -> None:
    """Build, validate, and train a custom LightGBM decision tree for ANY symbol from CoinDCX or Binance."""
    cfg = load_config(config_dir)
    ex = (exchange or detect_exchange(symbol)).lower()
    benchmarks = resolve_benchmarks(symbol, ex)
    clean_symbol = sanitize_symbol_for_filename(symbol)

    console.print(f"[bold cyan]==================================================================[/bold cyan]")
    console.print(f"[bold cyan]  ⚡ Building LightGBM Decision Tree for: {symbol}[/bold cyan]")
    console.print(f"  Exchange: [bold magenta]{ex.upper()}[/bold magenta] │ Benchmarks: [bold yellow]{benchmarks}[/bold yellow]")
    console.print(f"[bold cyan]==================================================================[/bold cyan]")

    storage = ParquetStorage(cfg.assets.storage.raw_dir.split("/")[0])
    cache = DataCache(cache_dir=cfg.assets.storage.cache_dir, enabled=cfg.assets.storage.use_cache and not no_cache)
    client = get_exchange_client(ex, cache=cache)

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=days)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    console.print(f"[cyan]Step 1/5: Ingesting data for {symbol} and benchmarks from {ex.upper()}...[/cyan]")
    target_df = client.fetch_klines(symbol, start_time_ms=start_ms, end_time_ms=end_ms, limit=1500)
    if target_df.empty:
        target_df = client.fetch_klines(symbol, limit=1500)
    if target_df.empty:
        console.print(f"[bold red]Failed to fetch data for {symbol} from {ex.upper()}. Please verify the pair name.[/bold red]")
        raise typer.Exit(1)

    bench_dfs = {}
    for b in benchmarks:
        b_df = client.fetch_klines(b, start_time_ms=start_ms, end_time_ms=end_ms, limit=1500)
        if b_df.empty:
            b_df = client.fetch_klines(b, limit=1500)
        bench_dfs[b] = b_df
    client.close()

    aligned = align_cross_asset_data(target_df, bench_dfs)
    storage.save_processed(aligned, name=f"aligned_{clean_symbol}")
    console.print(f"[green]✓ Ingested and aligned {len(aligned)} continuous bars.[/green]")

    console.print("[cyan]Step 2/5: Computing 35 core cross-asset & microstructure features...[/cyan]")
    pipe = FeaturePipeline(
        config=cfg.features,
        target_symbol=symbol,
        primary_benchmark=benchmarks[0],
        secondary_benchmark=benchmarks[1] if len(benchmarks) > 1 else benchmarks[0],
    )
    features = pipe.generate_features(aligned)
    storage.save_processed(features, name=f"features_{clean_symbol}")
    console.print(f"[green]✓ Computed {features.shape[1]} features.[/green]")

    console.print("[cyan]Step 3/5: Formulating 15-minute residual alpha target...[/cyan]")
    target = compute_residual_alpha_target(
        aligned_df=aligned,
        features_df=features,
        target_symbol=symbol,
        benchmark_symbol=benchmarks[0],
        horizon_minutes=cfg.features.target.horizon_min,
    )

    valid_target_idx = target.dropna().index
    clean_features = features.loc[valid_target_idx].ffill().bfill().fillna(0.0)
    common_idx = clean_features.index.intersection(valid_target_idx)
    X = clean_features.loc[common_idx]
    y = target.loc[common_idx]

    if len(X) < 50:
        console.print(f"[bold red]Insufficient valid samples ({len(X)}) to train a model for {symbol}. Try increasing --days.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Step 4/5: Running Purged Walk-Forward Cross-Validation...[/cyan]")
    total_days = max(1, (X.index.max() - X.index.min()).days)
    cv = PurgedWalkForwardCV(
        train_window_days=min(cfg.validation.train_window_days, max(3, int(total_days * 0.5))),
        test_window_days=min(cfg.validation.test_window_days, max(1, int(total_days * 0.2))),
        step_window_days=min(cfg.validation.step_window_days, max(1, int(total_days * 0.2))),
        purge_window_minutes=cfg.validation.purge_window_minutes,
        embargo_window_minutes=cfg.validation.embargo_window_minutes,
    )
    splits = list(cv.split(X))
    if splits:
        table = Table(title=f"Walk-Forward CV Results ({symbol})", show_header=True)
        table.add_column("Fold", style="cyan")
        table.add_column("Spearman IC", style="magenta")
        table.add_column("Hit Rate", style="green")
        for s in splits[:3]:
            trainer = ModelTrainer(cfg.model)
            trainer.train(X.iloc[s.train_indices], y.iloc[s.train_indices], X.iloc[s.test_indices], y.iloc[s.test_indices])
            metrics = trainer.evaluate(X.iloc[s.test_indices], y.iloc[s.test_indices])
            table.add_row(str(s.fold), f"{metrics['spearman_ic']:+.4f}", f"{metrics['hit_rate']*100:.1f}%")
        console.print(table)
    else:
        console.print("[yellow]Dataset span is too short for multi-day purged walk-forward CV. Evaluating on 80/20 train/val split...[/yellow]")
        split_idx = int(len(X) * 0.8)
        val_trainer = ModelTrainer(cfg.model)
        val_trainer.train(X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:])
        val_metrics = val_trainer.evaluate(X.iloc[split_idx:], y.iloc[split_idx:])
        console.print(f"  Validation Spearman IC: [bold magenta]{val_metrics['spearman_ic']:+.4f}[/bold magenta] │ Hit Rate: [bold green]{val_metrics['hit_rate']*100:.1f}%[/bold green]")

    console.print("[cyan]Step 5/5: Training production Multi-Model Cross-Verification Suite (LightGBM, XGBoost, CatBoost)...[/cyan]")
    split_idx = int(len(X) * 0.8)
    trainer = ModelTrainer(cfg.model)
    trainer.train(X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:])

    output_model = f"models/lightgbm_{clean_symbol}_residual_alpha.txt"
    trainer.save(output_model)

    ensemble = MultiModelEnsemble(cfg.model)
    ens_metrics = ensemble.train_all(X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:])
    ensemble.save_models("models", clean_symbol)
    console.print(f"[bold green]✓ Multi-model suite (LightGBM, XGBoost, CatBoost) saved to models/[/bold green]")

    # Multi-model evaluation table
    model_table = Table(title=f"Multi-Model Cross-Verification Suite ({symbol})", show_header=True)
    model_table.add_column("Model Architecture", style="cyan")
    model_table.add_column("Validation Spearman IC", style="magenta")
    model_table.add_column("Directional Hit Rate", style="green")
    model_table.add_column("RMSE", style="yellow")
    for m_name, m_vals in ens_metrics.items():
        model_table.add_row(
            m_name,
            f"{m_vals.get('spearman_ic', 0.0):+.4f}",
            f"{m_vals.get('hit_rate', 0.5)*100:.1f}%",
            f"{m_vals.get('rmse', 0.0):.6f}",
        )
    console.print(model_table)

    importances = trainer.get_feature_importance("gain")
    top_table = Table(title=f"Top 5 LightGBM Features for {symbol}", show_header=True)
    top_table.add_column("Feature", style="cyan")
    top_table.add_column("Gain", style="green")
    for f_name, gain_val in importances.head(5).items():
        top_table.add_row(f_name, f"{gain_val:.1f}")
    console.print(top_table)

    console.print("\n[bold green]To stream live predictions or launch Web UI for this symbol:[/bold green]")
    console.print(f"  [bold white]futurecast stream --symbol {symbol}[/bold white]")
    console.print(f"  [bold white]futurecast web --symbols {symbol}[/bold white]\n")


@app.command()
def forecast(
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Symbol from CoinDCX or Binance"),
    exchange: Optional[str] = typer.Option(None, "--exchange", "-e", help="Exchange (coindcx or binance)"),
    model_path: Optional[str] = typer.Option(None, help="Trained model path (defaults to models/lightgbm_<symbol>_residual_alpha.txt)"),
    config_dir: str = typer.Option("config", help="Configuration directory"),
    live: bool = typer.Option(False, "--live", help="Stream live continuous market data"),
    replay_bars: int = typer.Option(10, "--replay-bars", help="Number of historical bars to replay"),
    continuous: bool = typer.Option(False, "--continuous", help="Loop replay continuously"),
    ws_server: bool = typer.Option(False, "--ws-server", help="Start local WebSocket server (ws://127.0.0.1:8765)"),
    ws_port: int = typer.Option(8765, "--ws-port", help="Port for local WebSocket broadcast"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Render real-time reactive Rich dashboard"),
    poll_interval: float = typer.Option(3.0, "--poll-interval", help="Seconds between polls in live mode"),
    max_bars: Optional[int] = typer.Option(None, "--max-bars", help="Exit after streaming this many bars"),
) -> None:
    """Run Air-Gapped Real-Time Forecasting Engine (STRICTLY ZERO ORDERS)."""
    cfg = load_config(config_dir)
    cfg.enforce_zero_execution_policy()

    target_symbol = symbol or cfg.assets.target_asset
    ex = (exchange or detect_exchange(target_symbol)).lower()
    benchmarks = resolve_benchmarks(target_symbol, ex)
    clean_symbol = sanitize_symbol_for_filename(target_symbol)

    cfg.assets.target_asset = target_symbol
    cfg.assets.benchmark_assets = benchmarks

    if model_path is None:
        model_path = f"models/lightgbm_{clean_symbol}_residual_alpha.txt"
        if not Path(model_path).exists():
            # Fallback to default doge model if custom symbol model isn't built yet
            fallback_model = "models/lightgbm_doge_residual_alpha.txt"
            if Path(fallback_model).exists():
                model_path = fallback_model
            else:
                console.print(f"[yellow]Model {model_path} not found. Building it automatically...[/yellow]")
                build(symbol=target_symbol, exchange=ex, config_dir=config_dir)

    storage = ParquetStorage(cfg.assets.storage.raw_dir.split("/")[0])
    try:
        aligned = storage.load_processed(f"aligned_{clean_symbol}")
    except Exception:
        try:
            aligned = storage.load_processed("aligned_1m_data")
        except Exception:
            aligned = pd.DataFrame()

    trainer = ModelTrainer(cfg.model)
    trainer.load(model_path)

    pipe = FeaturePipeline(
        config=cfg.features,
        target_symbol=target_symbol,
        primary_benchmark=benchmarks[0],
        secondary_benchmark=benchmarks[1] if len(benchmarks) > 1 else benchmarks[0],
    )

    console.print(f"[bold green]Starting Air-Gapped Signal Engine for {target_symbol} ({ex.upper()})...[/bold green]")
    console.print("[bold yellow]Trading execution: DISABLED (Zero private keys, Read-only)[/bold yellow]")

    if live:
        asyncio.run(
            _run_live_stream(
                cfg=cfg,
                trainer=trainer,
                pipe=pipe,
                aligned_history=aligned,
                ws_server_enabled=ws_server,
                ws_port=ws_port,
                dashboard_enabled=dashboard,
                max_bars=max_bars,
                poll_interval=poll_interval,
                exchange=ex,
            )
        )
    else:
        asyncio.run(
            _run_replay_stream(
                cfg=cfg,
                trainer=trainer,
                pipe=pipe,
                aligned=aligned,
                replay_bars=replay_bars,
                continuous=continuous,
                ws_server_enabled=ws_server,
                ws_port=ws_port,
                dashboard_enabled=dashboard,
                delay_sec=0.4,
            )
        )


@app.command()
def stream(
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Symbol from CoinDCX or Binance (e.g. B-DOGE_USDT, B-SOL_USDT)"),
    exchange: Optional[str] = typer.Option(None, "--exchange", "-e", help="Exchange (coindcx or binance)"),
    model_path: Optional[str] = typer.Option(None, help="Trained model path"),
    config_dir: str = typer.Option("config", help="Configuration directory"),
    ws_port: int = typer.Option(8765, "--ws-port", help="Port for local WebSocket broadcast"),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard", help="Render rich terminal live dashboard"),
) -> None:
    """Continuously stream live signal states to console and local WebSocket (Air-gapped)."""
    forecast(
        symbol=symbol,
        exchange=exchange,
        model_path=model_path,
        config_dir=config_dir,
        live=True,
        ws_server=True,
        ws_port=ws_port,
        dashboard=dashboard,
        poll_interval=2.5,
    )


@app.command("cache")
def cache_cmd(
    clear: bool = typer.Option(False, "--clear", help="Clear all cached market data in .cache/"),
    config_dir: str = typer.Option("config", help="Configuration directory"),
) -> None:
    """Inspect or manage the .cache/ downloaded market data storage."""
    cfg = load_config(config_dir)
    cache = DataCache(cache_dir=cfg.assets.storage.cache_dir)
    if clear:
        removed = cache.clear()
        console.print(f"[bold yellow]Cleared {removed} cache files from {cache.cache_dir}.[/bold yellow]")
    else:
        stats = cache.get_stats()
        table = Table(title="futurecast .cache/ Storage Status", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Cache Directory", stats["cache_dir"])
        table.add_row("Total Files", str(stats["file_count"]))
        table.add_row("Total Size", f"{stats['total_mb']:.2f} MB ({stats['total_bytes']:,} bytes)")
        if stats["categories"]:
            for cat, cnt in stats["categories"].items():
                table.add_row(f"Category: {cat}", f"{cnt} files")
        else:
            table.add_row("Categories", "Empty (no downloads cached yet)")
        console.print(table)


@app.command("web")
@app.command("ui")
def web_cmd(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind the Web UI"),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the Web UI"),
    symbols: Optional[str] = typer.Option(None, "--symbols", "-s", help="Comma-separated tokens to track initially (e.g. B-DOGE_USDT,B-SOL_USDT,DOGEUSDT)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Run with simulated market data feed for offline testing"),
    poll_interval: float = typer.Option(2.5, "--poll-interval", help="Polling interval in seconds"),
) -> None:
    """Launch Real-Time Multi-Coin Web Terminal with Multi-Model Cross-Verification."""
    from futurecast.publisher.web_server import run_web_server

    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    run_web_server(
        host=host,
        port=port,
        symbols=sym_list,
        live=not synthetic,
        poll_interval=poll_interval,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
