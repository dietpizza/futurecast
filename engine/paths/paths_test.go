package paths

import (
	"os"
	"path/filepath"
	"testing"
)

// noEnv clears the home variables and points the default and legacy folders under a temp folder.
func noEnv(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("DELTARISK_HOME", "")
	t.Setenv("FUTURECAST_HOME", "")
	oldDefault, oldLegacy := DefaultHome, LegacyHome
	DefaultHome, LegacyHome = filepath.Join(tmp, ".deltarisk"), filepath.Join(tmp, ".futurecast")
	t.Cleanup(func() { DefaultHome, LegacyHome = oldDefault, oldLegacy })
	return tmp
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func TestHomeDefaultsToDotDeltarisk(t *testing.T) {
	tmp := noEnv(t)
	if got := Home(); got != filepath.Join(tmp, ".deltarisk") {
		t.Fatalf("Home() = %q", got)
	}
	user, _ := os.UserHomeDir()
	if got := ExpandUser("~/.deltarisk"); got != filepath.Join(user, ".deltarisk") {
		t.Fatalf("ExpandUser = %q", got)
	}
}

func TestThePreRenameHomeMovesOverOnce(t *testing.T) {
	tmp := noEnv(t)
	legacy := filepath.Join(tmp, ".futurecast")
	if err := os.MkdirAll(filepath.Join(legacy, "cache"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(legacy, "tracked_symbols.json"), []byte(`{"symbols": ["B-ETH_USDT"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := Home(); got != filepath.Join(tmp, ".deltarisk") {
		t.Fatalf("Home() = %q", got)
	}
	if exists(legacy) || !exists(filepath.Join(tmp, ".deltarisk", "tracked_symbols.json")) {
		t.Fatal("the legacy folder was not moved")
	}

	// Once the new folder exists, a legacy folder (say, recreated by an old build) is left alone
	if err := os.Mkdir(legacy, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := Home(); got != filepath.Join(tmp, ".deltarisk") || !exists(legacy) {
		t.Fatalf("Home() = %q, legacy kept = %v", got, exists(legacy))
	}
}

func TestTheLegacyEnvVarStillCounts(t *testing.T) {
	tmp := noEnv(t)
	t.Setenv("FUTURECAST_HOME", filepath.Join(tmp, "old"))
	if got := Home(); got != filepath.Join(tmp, "old") {
		t.Fatalf("Home() = %q", got)
	}
	t.Setenv("DELTARISK_HOME", filepath.Join(tmp, "new"))
	if got := Home(); got != filepath.Join(tmp, "new") {
		t.Fatalf("Home() = %q", got)
	}
}

func TestRelativePathsResolveUnderHome(t *testing.T) {
	tmp := noEnv(t)
	home := filepath.Join(tmp, "deltarisk_home")
	t.Setenv("DELTARISK_HOME", home)
	user, _ := os.UserHomeDir()
	for in, want := range map[string]string{
		"models":                        filepath.Join(home, "models"),
		filepath.Join(tmp, "elsewhere"): filepath.Join(tmp, "elsewhere"),
		"~/x":                           filepath.Join(user, "x"),
	} {
		if got := Resolve(in); got != want {
			t.Errorf("Resolve(%q) = %q, want %q", in, got, want)
		}
	}
}
