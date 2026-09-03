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

// Same cleanup guarantee, but through the disallowed-format branch rather
// than the decode-failure branch TestFetchLeavesNoTempFileBehind exercises
// above — both refusal paths share the same deferred os.Remove in Fetch.
func TestFetchLeavesNoTempFileBehindOnDisallowedFormat(t *testing.T) {
	srv := serve(t, "image/png", pngBytes(t, 2, 2), nil)
	defer srv.Close()
	dir := t.TempDir()
	o := opts()
	o.Allow = []string{"jpeg"}
	Fetch(srv.URL, filepath.Join(dir, "a.png"), o)
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temp files left behind: %v", entries)
	}
}
