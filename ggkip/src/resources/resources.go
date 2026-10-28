// Package resources embeds migrations, seed data and settings into the
// binary, so it runs without RESOURCES_DIR.
package resources

import "embed"

//go:embed migrations seed settings
var FS embed.FS
