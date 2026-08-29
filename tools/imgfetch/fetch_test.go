package main

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func pngBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.RGBA{200, 30, 30, 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func jpegBytes(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 4, 3)), nil); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func serve(t *testing.T, contentType string, body []byte, extra func(http.ResponseWriter)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		if extra != nil {
			extra(w)
		}
		w.Write(body)
	}))
}

func opts() Options {
	return Options{MaxBytes: 1 << 20, Allow: []string{"jpeg", "png", "gif", "webp"}, Timeout: 5 * time.Second}
}

func TestFetchAcceptsValidPNG(t *testing.T) {
	srv := serve(t, "image/png", pngBytes(t, 8, 6), nil)
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "a.png")
	res, err := Fetch(srv.URL, out, opts())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Format != "png" || res.Width != 8 || res.Height != 6 {
		t.Fatalf("bad result: %+v", res)
	}
	data, err := os.ReadFile(out)
	if err != nil || res.Bytes != int64(len(data)) {
		t.Fatalf("output not written correctly: err=%v bytes=%d", err, len(data))
	}
}

func TestFetchAcceptsJPEGDespiteGenericContentType(t *testing.T) {
	// the decode is authoritative; a sloppy image/* header still passes
	srv := serve(t, "image/*", jpegBytes(t), nil)
	defer srv.Close()
	res, err := Fetch(srv.URL, filepath.Join(t.TempDir(), "a.jpg"), opts())
	if err != nil || res.Format != "jpeg" {
		t.Fatalf("got %+v, %v", res, err)
	}
}

func TestFetchRefusesNonImageContentType(t *testing.T) {
	srv := serve(t, "text/html", []byte("<html>"), nil)
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "a.png")
	if _, err := Fetch(srv.URL, out, opts()); err == nil || !strings.Contains(err.Error(), "content-type") {
		t.Fatalf("expected content-type refusal, got %v", err)
	}
	if _, statErr := os.Stat(out); statErr == nil {
		t.Fatal("output must not exist after refusal")
	}
}

func TestFetchRefusesOversizeContentLengthBeforeDownload(t *testing.T) {
	var served bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Length", "99999999")
		served = true
	}))
	defer srv.Close()
	o := opts()
	o.MaxBytes = 1000
	_, err := Fetch(srv.URL, filepath.Join(t.TempDir(), "a.png"), o)
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("expected size refusal, got %v", err)
	}
	_ = served
}

func TestFetchRefusesOversizeStreamWithoutContentLength(t *testing.T) {
	big := make([]byte, 5000)
	copy(big, pngBytes(t, 2, 2))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Transfer-Encoding", "chunked") // no Content-Length
		w.Write(big)
	}))
	defer srv.Close()
	o := opts()
	o.MaxBytes = 1000
	_, err := Fetch(srv.URL, filepath.Join(t.TempDir(), "a.png"), o)
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("expected stream size refusal, got %v", err)
	}
}

func TestFetchRefusesBytesThatAreNotAnImage(t *testing.T) {
	srv := serve(t, "image/png", []byte("definitely not a png"), nil)
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "a.png")
	_, err := Fetch(srv.URL, out, opts())
	if err == nil || !strings.Contains(err.Error(), "not a") {
		t.Fatalf("expected decode refusal, got %v", err)
	}
	if _, statErr := os.Stat(out); statErr == nil {
		t.Fatal("output must not exist after refusal")
	}
}

func TestFetchRefusesDisallowedFormat(t *testing.T) {
	srv := serve(t, "image/png", pngBytes(t, 2, 2), nil)
	defer srv.Close()
	o := opts()
	o.Allow = []string{"jpeg"}
	_, err := Fetch(srv.URL, filepath.Join(t.TempDir(), "a.png"), o)
	if err == nil || !strings.Contains(err.Error(), "format") {
		t.Fatalf("expected format refusal, got %v", err)
	}
}

func TestFetchRefusesHTTPStatusErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gone", http.StatusNotFound)
	}))
	defer srv.Close()
	if _, err := Fetch(srv.URL, filepath.Join(t.TempDir(), "a.png"), opts()); err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("expected status refusal, got %v", err)
	}
}

func TestFetchLeavesNoTempFileBehind(t *testing.T) {
	srv := serve(t, "image/png", []byte("nope"), nil)
	defer srv.Close()
	dir := t.TempDir()
	Fetch(srv.URL, filepath.Join(dir, "a.png"), opts())
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temp files left behind: %v", entries)
	}
}

func TestFetchSendsDescriptiveUserAgent(t *testing.T) {
	var ua string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ua = r.Header.Get("User-Agent")
		w.Header().Set("Content-Type", "image/png")
		w.Write(pngBytes(t, 2, 2))
	}))
	defer srv.Close()
	if _, err := Fetch(srv.URL, filepath.Join(t.TempDir(), "a.png"), opts()); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(ua, "imgfetch/") || !strings.Contains(ua, "github.com/cehbz") {
		t.Fatalf("user-agent %q must identify the tool and a contact URL (Wikimedia policy)", ua)
	}
}
