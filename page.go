//go:build !embedweb

// Package deltarisk holds the dashboard page compiled into the binary. `mise run build` builds the
// page into web/dist and the binary with `-tags embedweb`, which embeds it (page_embed.go); a
// build without the tag (go run, go test) has no page and serves web/dist from disk.
package deltarisk

import "io/fs"

// Page is the built page when it is compiled into the binary; this build has none.
func Page() (fs.FS, bool) { return nil, false }
