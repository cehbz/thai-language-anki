// Package main implements imgfetch: download one URL to a path only if it
// is a real image within size and format limits.
package main

import (
	"errors"
	"fmt"
	"image"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

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

const (
	maxRedirects = 5
	maxSide      = 20000 // pixels; larger is a decompression bomb, not a picture
	// Wikimedia (and others) refuse anonymous default agents; identify the tool and a contact.
	userAgent = "imgfetch/1.0 (https://github.com/cehbz/thai-language-anki; deck media fetcher)"
)

// Fetch downloads url to outPath, validating along the way. On any
// refusal outPath is left untouched and an error explains why.
func Fetch(url, outPath string, opts Options) (Result, error) {
	client := &http.Client{
		Timeout: opts.Timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxRedirects {
				return errors.New("too many redirects")
			}
			return nil
		},
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return Result{}, fmt.Errorf("bad url: %w", err)
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		return Result{}, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Result{}, fmt.Errorf("http %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "image/") {
		return Result{}, fmt.Errorf("content-type %q is not an image", ct)
	}
	if resp.ContentLength > opts.MaxBytes {
		return Result{}, fmt.Errorf("too large: content-length %d > %d", resp.ContentLength, opts.MaxBytes)
	}

	tmp, err := os.CreateTemp(filepath.Dir(outPath), ".imgfetch-*")
	if err != nil {
		return Result{}, fmt.Errorf("temp file: %w", err)
	}
	defer os.Remove(tmp.Name()) // no-op once renamed into place
	defer tmp.Close()

	// +1 so a stream that exactly hits the cap is distinguishable from one that exceeds it.
	n, err := io.Copy(tmp, io.LimitReader(resp.Body, opts.MaxBytes+1))
	if err != nil {
		return Result{}, fmt.Errorf("download: %w", err)
	}
	if n > opts.MaxBytes {
		return Result{}, fmt.Errorf("too large: stream exceeds %d bytes", opts.MaxBytes)
	}
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		return Result{}, err
	}
	cfg, format, err := image.DecodeConfig(tmp)
	if err != nil {
		return Result{}, fmt.Errorf("not a decodable image: %w", err)
	}
	if !slices.Contains(opts.Allow, format) {
		return Result{}, fmt.Errorf("format %q not allowed (allowed: %s)", format, strings.Join(opts.Allow, ","))
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || cfg.Width > maxSide || cfg.Height > maxSide {
		return Result{}, fmt.Errorf("unreasonable dimensions %dx%d", cfg.Width, cfg.Height)
	}
	if err := tmp.Close(); err != nil {
		return Result{}, err
	}
	if err := os.Rename(tmp.Name(), outPath); err != nil {
		return Result{}, fmt.Errorf("move into place: %w", err)
	}
	return Result{Format: format, Width: cfg.Width, Height: cfg.Height, Bytes: n}, nil
}
