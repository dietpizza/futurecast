// Package paths says where deltarisk keeps its runtime files.
//
// Everything the app reads or writes at runtime (config overrides, tracked coins, the candle and
// funding caches) lives under one home folder: ~/.deltarisk, or $DELTARISK_HOME when set.
// Relative paths given to deltarisk are taken relative to that folder.
//
// The app was called FutureCast until 2026-10-18: $FUTURECAST_HOME is still honoured, and an old
// ~/.futurecast folder is moved to ~/.deltarisk the first time the new default is needed.
package paths

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// Defaults, variables so tests can point them at temp folders (never the real ones).
var (
	DefaultHome = "~/.deltarisk"
	LegacyHome  = "~/.futurecast" // before the rename
)

// Home is the deltarisk home folder: $DELTARISK_HOME, then the legacy $FUTURECAST_HOME, default
// ~/.deltarisk, which takes over a legacy ~/.futurecast on first use.
func Home() string {
	for _, name := range []string{"DELTARISK_HOME", "FUTURECAST_HOME"} {
		if configured := os.Getenv(name); configured != "" {
			return ExpandUser(configured)
		}
	}
	def := ExpandUser(DefaultHome)
	adoptLegacyHome(def, ExpandUser(LegacyHome))
	return def
}

// adoptLegacyHome moves the pre-rename home folder to the new default, once, if only the old one exists.
func adoptLegacyHome(def, legacy string) {
	if _, err := os.Stat(def); err == nil {
		return
	}
	if info, err := os.Stat(legacy); err != nil || !info.IsDir() {
		return
	}
	if err := os.Rename(legacy, def); err != nil {
		slog.Warn("could not move the old home folder", "from", legacy, "to", def, "err", err)
		return
	}
	slog.Info("moved the old home folder (the app is now called DeltaRisk)", "from", legacy, "to", def)
}

// Resolve returns path itself when absolute (after expanding ~), otherwise the same path under the home folder.
func Resolve(path string) string {
	p := ExpandUser(path)
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Join(Home(), p)
}

// ExpandUser replaces a leading ~ with the user's home directory, like Python's Path.expanduser.
func ExpandUser(path string) string {
	if path != "~" && !strings.HasPrefix(path, "~/") {
		return path
	}
	dir, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(dir, path[1:])
}
