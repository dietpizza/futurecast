//go:build embedweb

package deltarisk

import (
	"embed"
	"io/fs"
)

// The page as `mise run build` made it, compiled in (`go build -tags embedweb` needs web/dist built first).
//
//go:embed all:web/dist
var dist embed.FS

// Page is the built page (index.html, favicon.svg, assets/) when it is compiled into the binary.
func Page() (fs.FS, bool) {
	page, err := fs.Sub(dist, "web/dist")
	if err != nil {
		return nil, false
	}
	return page, true
}
