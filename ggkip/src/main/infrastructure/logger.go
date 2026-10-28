package infrastructure

import (
	"log/slog"
	"os"
	"strings"
)

// NewLogger returns a text logger for local development and a JSON logger elsewhere.
func NewLogger(appEnv, level string) *slog.Logger {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(strings.ToUpper(level))); err != nil {
		lvl = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lvl}
	if appEnv == "local" {
		return slog.New(slog.NewTextHandler(os.Stdout, opts))
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, opts))
}
