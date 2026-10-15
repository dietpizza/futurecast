"""Tests that every runtime file lands under the futurecast home folder."""

from pathlib import Path

from futurecast import paths
from futurecast.config import load_config
from futurecast.publisher.broadcaster import SignalBroadcaster, _order
from futurecast.publisher.multi_manager import MultiCoinStreamManager


def test_home_defaults_to_dot_futurecast(monkeypatch):
    monkeypatch.delenv("FUTURECAST_HOME")
    assert paths.home() == Path.home() / ".futurecast"


def test_relative_paths_resolve_under_home(futurecast_home, tmp_path):
    assert paths.home() == futurecast_home
    assert paths.resolve("models") == futurecast_home / "models"
    assert paths.resolve(tmp_path / "elsewhere") == tmp_path / "elsewhere"
    assert paths.resolve("~/x") == Path.home() / "x"


def test_runtime_files_default_to_home(futurecast_home):
    mgr = MultiCoinStreamManager(live=False)
    assert mgr.models_dir == futurecast_home / "models"
    assert mgr.persistence_path == futurecast_home / "tracked_symbols.json"

    mgr.add_symbol("B-DOGE_USDT")
    assert (futurecast_home / "tracked_symbols.json").exists()

    broadcaster = SignalBroadcaster(output_dir=mgr.config.forecasting.output_dir)
    assert broadcaster.output_dir == futurecast_home / "logs" / "forecasts"


def test_config_overrides_read_from_home(futurecast_home):
    assert load_config().forecasting.signal_threshold_z == 1.5  # no files: built-in defaults

    (futurecast_home / "config").mkdir(parents=True)
    (futurecast_home / "config" / "model.yaml").write_text("forecasting:\n  signal_threshold_z: 2.0\n")
    assert load_config().forecasting.signal_threshold_z == 2.0


_EVENT = {
    "symbol": "B-SOL_USDT", "benchmark_symbol": "B-BTC_USDT", "current_price": 150.0, "benchmark_price": 60_000.0,
    "expected_residual_alpha_15m_bps": 1.5, "z_score": 0.3, "beta_hedge_ratio": 1.1, "signal": "NEUTRAL",
    "confidence_percentile": 50.0, "microstructure_context": {"volume_zscore_60m": 0.2},
}


def test_forecast_logs_stay_under_their_limit(tmp_path):
    from datetime import datetime, timedelta, timezone

    from futurecast.schema import ForecastEvent

    folder = tmp_path / "forecasts"
    folder.mkdir()
    (folder / "forecasts_2026-01-01.jsonl").write_bytes(b"x" * 300_000)  # an old, oversized day
    broadcaster = SignalBroadcaster(output_dir=folder, max_mb=0.2)  # 200 KiB in total, 50 KiB per file
    assert not (folder / "forecasts_2026-01-01.jsonl").exists()  # trimmed as soon as logging starts

    start = datetime(2026, 1, 2, tzinfo=timezone.utc)
    for i in range(3000):
        event = ForecastEvent.model_validate({
            **_EVENT, "timestamp": (start + timedelta(minutes=i)).isoformat(),
        })
        broadcaster.broadcast(event)

    files = sorted(folder.glob("forecasts_*.jsonl"))
    assert sum(f.stat().st_size for f in files) <= broadcaster.max_bytes
    assert all(f.stat().st_size <= broadcaster.part_bytes for f in files)
    days = {f.name[len("forecasts_"):len("forecasts_2026-01-01")] for f in files}
    assert "2026-01-04" in days and "2026-01-02" not in days  # the newest kept, the oldest dropped
    newest = max(files, key=_order)
    last = newest.read_text().splitlines()[-1]
    assert '"2026-01-04T01:59:00Z"' in last  # the latest forecast is in the log
