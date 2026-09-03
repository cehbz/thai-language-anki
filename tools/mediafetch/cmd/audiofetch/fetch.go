// Command audiofetch downloads one URL to an out path only if it is a real
// MP3 within size limits, printing a JSON line {format,bytes}.
package main

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"mediafetch/internal/fetch"
)

// Options bound what a fetch will accept.
type Options struct {
	MaxBytes int64
	Allow    []string // audio formats accepted; currently only "mp3" is recognized
	Timeout  time.Duration
}

// Result describes a successfully fetched audio file.
type Result struct {
	Format string `json:"format"`
	Bytes  int64  `json:"bytes"`
}

// headPeekBytes is how much of the temp file validate reads to sniff the
// MP3 header; ID3v2 tags and frame syncs both live in the first few bytes.
const headPeekBytes = 16

// Fetch downloads url to outPath, validating along the way. On any
// refusal outPath is left untouched and an error explains why.
func Fetch(url, outPath string, opts Options) (result Result, err error) {
	tmp, _, n, err := fetch.Download(url, fetch.Options{
		MaxBytes:     opts.MaxBytes,
		Timeout:      opts.Timeout,
		ContentTypes: []string{"audio/mpeg", "audio/mp3", "application/octet-stream"},
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

	if err = validate(tmp); err != nil {
		return Result{}, err
	}
	if !slices.Contains(opts.Allow, "mp3") {
		return Result{}, fmt.Errorf("format \"mp3\" not allowed (allowed: %s)", strings.Join(opts.Allow, ","))
	}

	if err = fetch.Commit(tmp, outPath); err != nil {
		return Result{}, fmt.Errorf("move into place: %w", err)
	}
	return Result{Format: "mp3", Bytes: n}, nil
}

// validate reports whether the temp file at path looks like an MP3.
func validate(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	head := make([]byte, headPeekBytes)
	n, err := f.Read(head)
	if err != nil && n == 0 {
		return fmt.Errorf("empty response")
	}
	if !fetch.LooksLikeMP3(head[:n]) {
		return fmt.Errorf("not a recognizable MP3 (no ID3 tag or frame sync)")
	}
	return nil
}
