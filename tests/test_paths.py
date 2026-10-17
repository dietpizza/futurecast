"""Tests that every runtime file lands under the deltarisk home folder."""

from pathlib import Path

from deltarisk import paths
from deltarisk.config import load_config
from deltarisk.publisher.multi_manager import MultiCoinStreamManager


def _no_env(monkeypatch, tmp_path):
    """No home set in the environment, and default/legacy folders under tmp (never the real ones)."""
    monkeypatch.delenv("DELTARISK_HOME", raising=False)
    monkeypatch.delenv("FUTURECAST_HOME", raising=False)
    monkeypatch.setattr(paths, "DEFAULT_HOME", str(tmp_path / ".deltarisk"))
    monkeypatch.setattr(paths, "LEGACY_HOME", str(tmp_path / ".futurecast"))


def test_home_defaults_to_dot_deltarisk(monkeypatch, tmp_path):
    _no_env(monkeypatch, tmp_path)
    assert paths.home() == tmp_path / ".deltarisk"
    assert paths.DEFAULT_HOME != "~/.deltarisk" and Path("~/.deltarisk").expanduser() == Path.home() / ".deltarisk"


def test_the_pre_rename_home_moves_over_once(monkeypatch, tmp_path):
    _no_env(monkeypatch, tmp_path)
    legacy = tmp_path / ".futurecast"
    (legacy / "cache").mkdir(parents=True)
    (legacy / "tracked_symbols.json").write_text('{"symbols": ["B-ETH_USDT"]}')
    assert paths.home() == tmp_path / ".deltarisk"
    assert not legacy.exists() and (tmp_path / ".deltarisk" / "tracked_symbols.json").exists()

    # Once the new folder exists, a legacy folder (say, recreated by an old build) is left alone
    legacy.mkdir()
    assert paths.home() == tmp_path / ".deltarisk" and legacy.exists()


def test_the_legacy_env_var_still_counts(monkeypatch, tmp_path):
    _no_env(monkeypatch, tmp_path)
    monkeypatch.setenv("FUTURECAST_HOME", str(tmp_path / "old"))
    assert paths.home() == tmp_path / "old"
    monkeypatch.setenv("DELTARISK_HOME", str(tmp_path / "new"))
    assert paths.home() == tmp_path / "new"


def test_relative_paths_resolve_under_home(deltarisk_home, tmp_path):
    assert paths.home() == deltarisk_home
    assert paths.resolve("models") == deltarisk_home / "models"
    assert paths.resolve(tmp_path / "elsewhere") == tmp_path / "elsewhere"
    assert paths.resolve("~/x") == Path.home() / "x"


def test_runtime_files_default_to_home(deltarisk_home):
    mgr = MultiCoinStreamManager(live=False)
    assert mgr.persistence_path == deltarisk_home / "tracked_symbols.json"

    mgr.add_symbol("B-DOGE_USDT")
    assert (deltarisk_home / "tracked_symbols.json").exists()
    assert paths.resolve(mgr.config.assets.storage.cache_dir) == deltarisk_home / "cache"


def test_config_overrides_read_from_home(deltarisk_home):
    assert load_config().assets.data_source.request_timeout_sec == 15  # no files: built-in defaults

    (deltarisk_home / "config").mkdir(parents=True)
    (deltarisk_home / "config" / "assets.yaml").write_text("data_source:\n  request_timeout_sec: 30\n")
    assert load_config().assets.data_source.request_timeout_sec == 30

