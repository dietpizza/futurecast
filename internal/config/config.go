// Package config loads deltarisk's settings: built-in defaults (Default), overridden by
// <home>/config/assets.yaml when it exists. Unknown keys are ignored, so old home config files
// still load. The repo's config/assets.yaml only documents the defaults; it is never read.
package config

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	"deltarisk/internal/paths"
)

type DataSource struct {
	BaseURL           string `yaml:"base_url"`
	RequestTimeoutSec int    `yaml:"request_timeout_sec"`
	MaxRetries        int    `yaml:"max_retries"`
}

type Storage struct {
	CacheDir string `yaml:"cache_dir"` // downloaded candles go in <cache_dir>/candles
	UseCache bool   `yaml:"use_cache"`
}

type Assets struct {
	DataSource DataSource `yaml:"data_source"`
	Storage    Storage    `yaml:"storage"`
}

type App struct {
	Assets Assets
}

// Default is the configuration without any override file.
func Default() App {
	return App{Assets: Assets{
		DataSource: DataSource{BaseURL: "https://fapi.binance.com", RequestTimeoutSec: 15, MaxRetries: 3},
		Storage:    Storage{CacheDir: "cache", UseCache: true},
	}}
}

// Load reads the settings in configDir (relative paths are under the home folder, so "config"
// is ~/.deltarisk/config); a missing file keeps the defaults, and so does a key it leaves out.
func Load(configDir string) (App, error) {
	cfg := Default()
	path := filepath.Join(paths.Resolve(configDir), "assets.yaml")
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return cfg, nil
	}
	if err != nil {
		return cfg, err
	}
	defer f.Close()
	// Decoding into the defaults keeps every field the file doesn't set
	if err := yaml.NewDecoder(f).Decode(&cfg.Assets); err != nil && !errors.Is(err, io.EOF) {
		return Default(), fmt.Errorf("%s: %w", path, err)
	}
	return cfg, nil
}
