"""Tests that every runtime file lands under the futurecast home folder."""

from pathlib import Path

from futurecast import paths
from futurecast.config import load_config
from futurecast.publisher.broadcaster import SignalBroadcaster
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
