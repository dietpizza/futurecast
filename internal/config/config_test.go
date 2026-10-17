package config

import (
	"os"
	"path/filepath"
	"testing"

	"deltarisk/internal/paths"
)

// tempHome points DELTARISK_HOME at a temp folder so tests never touch ~/.deltarisk.
func tempHome(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	home := filepath.Join(tmp, "deltarisk_home")
	t.Setenv("DELTARISK_HOME", home)
	t.Setenv("FUTURECAST_HOME", "")
	oldDefault, oldLegacy := paths.DefaultHome, paths.LegacyHome
	paths.DefaultHome, paths.LegacyHome = filepath.Join(tmp, "default_home"), filepath.Join(tmp, "legacy_home")
	t.Cleanup(func() { paths.DefaultHome, paths.LegacyHome = oldDefault, oldLegacy })
	return home
}

func writeAssets(t *testing.T, home, text string) {
	t.Helper()
	dir := filepath.Join(home, "config")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "assets.yaml"), []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestConfigLoadsDefaults(t *testing.T) {
	tempHome(t)
	repo, _ := filepath.Abs("../../config")
	cfg, err := Load(repo)
	if err != nil {
		t.Fatal(err)
	}
	if cfg != Default() {
		t.Fatalf("the repo's assets.yaml should match the built-in defaults: %+v", cfg)
	}
	if cfg.Assets.DataSource.BaseURL != "https://fapi.binance.com" || !cfg.Assets.Storage.UseCache {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
}

func TestConfigOverridesReadFromHome(t *testing.T) {
	home := tempHome(t)
	cfg, err := Load("config")
	if err != nil || cfg.Assets.DataSource.RequestTimeoutSec != 15 { // no files: built-in defaults
		t.Fatalf("Load = %+v, %v", cfg, err)
	}

	writeAssets(t, home, "data_source:\n  request_timeout_sec: 30\n")
	cfg, err = Load("config")
	if err != nil {
		t.Fatal(err)
	}
	want := Default()
	want.Assets.DataSource.RequestTimeoutSec = 30
	if cfg != want {
		t.Fatalf("Load = %+v, want %+v", cfg, want)
	}
}

func TestUnknownKeysAndEmptyFilesKeepTheDefaults(t *testing.T) {
	home := tempHome(t)
	writeAssets(t, home, "")
	if cfg, err := Load("config"); err != nil || cfg != Default() {
		t.Fatalf("empty file: %+v, %v", cfg, err)
	}

	writeAssets(t, home, "symbols: [B-ETH_USDT]\nstorage:\n  use_cache: false\n  models_dir: models\n")
	cfg, err := Load("config")
	if err != nil {
		t.Fatal(err)
	}
	want := Default()
	want.Assets.Storage.UseCache = false
	if cfg != want {
		t.Fatalf("Load = %+v, want %+v", cfg, want)
	}
}

func TestBrokenYAMLIsAnError(t *testing.T) {
	home := tempHome(t)
	writeAssets(t, home, "data_source: [unclosed\n")
	if _, err := Load("config"); err == nil {
		t.Fatal("expected an error")
	}
}
