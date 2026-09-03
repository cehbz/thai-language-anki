// Command imgfetch downloads one URL to an out path only if it is a real
// image within size and format limits, printing a JSON line
// {format,width,height,bytes}.
package main

import (
	"fmt"
	"image"
	"os"
	"slices"
	"strings"
	"time"

	"mediafetch/internal/fetch"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
)

// Options bound what a fetch will accept.
type Options struct {
	MaxBytes int64
	Allow    []string // image formats as reported by image.DecodeConfig: jpeg, png, gif, webp
	Timeout  time.Duration
}

// Result describes a successfully fetched image.
type Result struct {
	Format string `json:"format"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Bytes  int64  `json:"bytes"`
}

const maxSide = 20000 // pixels; larger is a decompression bomb, not a picture

// Fetch downloads url to outPath, validating along the way. On any
// refusal outPath is left untouched and an error explains why.
func Fetch(url, outPath string, opts Options) (result Result, err error) {
	tmp, _, n, err := fetch.Download(url, fetch.Options{
		MaxBytes:     opts.MaxBytes,
		Timeout:      opts.Timeout,
		ContentTypes: []string{"image/"},
		OutPath:      outPath,
	})
	if err != nil {
		return Result{}, err
	}
	// Download succeeded and left tmp behind; from here on, any refusal
	// (returned as a non-nil err) must clean it up.
	defer func() {
		if err != nil {
			os.Remove(tmp)
		}
	}()

	f, err := os.Open(tmp)
	if err != nil {
		return Result{}, err
	}
	cfg, format, err := image.DecodeConfig(f)
	f.Close()
	if err != nil {
		return Result{}, fmt.Errorf("not a decodable image: %w", err)
	}
	if !slices.Contains(opts.Allow, format) {
		return Result{}, fmt.Errorf("format %q not allowed (allowed: %s)", format, strings.Join(opts.Allow, ","))
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || cfg.Width > maxSide || cfg.Height > maxSide {
		return Result{}, fmt.Errorf("unreasonable dimensions %dx%d", cfg.Width, cfg.Height)
	}

	if err = fetch.Commit(tmp, outPath); err != nil {
		return Result{}, fmt.Errorf("move into place: %w", err)
	}
	return Result{Format: format, Width: cfg.Width, Height: cfg.Height, Bytes: n}, nil
}
