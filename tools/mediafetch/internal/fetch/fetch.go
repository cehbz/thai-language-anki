// Package fetch implements the shared download machinery for mediafetch's
// commands: an HTTP GET with a descriptive User-Agent, a byte cap enforced
// on both the response header and the stream, and an atomic write via a
// temp file plus rename. Content validation specific to a media type
// (image decoding, MP3 header sniffing) is the caller's job.
package fetch

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Options bound what a Download will accept.
type Options struct {
	MaxBytes int64
	Timeout  time.Duration
	// ContentTypes lists acceptable response Content-Type values. An entry
	// ending in "/" matches by prefix (e.g. "image/" accepts "image/png");
	// any other entry must match exactly (e.g. "audio/mpeg").
	ContentTypes []string
	// OutPath is the eventual destination; the temp file is created in its
	// directory so the final Commit rename stays on one filesystem.
	OutPath string
}

const (
	maxRedirects = 5
	// Wikimedia (and others) refuse anonymous default agents; identify the tool and a contact.
	userAgent = "mediafetch/1.0 (https://github.com/cehbz/thai-language-anki; deck media fetcher)"
)

// Download fetches url, enforcing opts.ContentTypes and opts.MaxBytes, and
// writes the body to a temp file next to opts.OutPath. On any refusal no
// temp file is left behind. The caller is responsible for validating the
// returned temp file's contents and then calling Commit (or removing it).
func Download(url string, opts Options) (path string, contentType string, size int64, err error) {
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
		return "", "", 0, fmt.Errorf("bad url: %w", err)
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		return "", "", 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", 0, fmt.Errorf("http %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !contentTypeAllowed(ct, opts.ContentTypes) {
		return "", "", 0, fmt.Errorf("content-type %q is not allowed", ct)
	}
	if resp.ContentLength > opts.MaxBytes {
		return "", "", 0, fmt.Errorf("too large: content-length %d > %d", resp.ContentLength, opts.MaxBytes)
	}

	tmp, err := os.CreateTemp(filepath.Dir(opts.OutPath), ".mediafetch-*")
	if err != nil {
		return "", "", 0, fmt.Errorf("temp file: %w", err)
	}
	// Always close before removing: on any error return below, this closes
	// tmp first and then, seeing a non-nil err, removes it. On success the
	// explicit Close below already ran, so this is a harmless no-op.
	defer func() {
		tmp.Close()
		if err != nil {
			os.Remove(tmp.Name())
		}
	}()

	// +1 so a stream that exactly hits the cap is distinguishable from one that exceeds it.
	n, err := io.Copy(tmp, io.LimitReader(resp.Body, opts.MaxBytes+1))
	if err != nil {
		return "", "", 0, fmt.Errorf("download: %w", err)
	}
	if n > opts.MaxBytes {
		return "", "", 0, fmt.Errorf("too large: stream exceeds %d bytes", opts.MaxBytes)
	}
	if err = tmp.Close(); err != nil {
		return "", "", 0, err
	}
	return tmp.Name(), ct, n, nil
}

// Commit atomically moves tmp (as returned by Download) into place at dest.
func Commit(tmp, dest string) error {
	return os.Rename(tmp, dest)
}

// contentTypeAllowed reports whether ct is permitted by allowed. An entry
// ending in "/" matches by prefix; any other entry must match exactly.
func contentTypeAllowed(ct string, allowed []string) bool {
	for _, a := range allowed {
		if strings.HasSuffix(a, "/") {
			if strings.HasPrefix(ct, a) {
				return true
			}
		} else if ct == a {
			return true
		}
	}
	return false
}
