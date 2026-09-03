package fetch

import (
	"bytes"
	"image"
	"image/color"
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

func imageOpts(dir string) Options {
	return Options{
		MaxBytes:     1 << 20,
		Timeout:      5 * time.Second,
		ContentTypes: []string{"image/"},
		OutPath:      filepath.Join(dir, "a.png"),
	}
}

func TestDownloadAcceptsMatchingContentType(t *testing.T) {
	srv := serve(t, "image/png", pngBytes(t, 8, 6), nil)
	defer srv.Close()
	dir := t.TempDir()
	tmp, ct, size, err := Download(srv.URL, imageOpts(dir))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(tmp)
	if ct != "image/png" {
		t.Fatalf("content-type = %q", ct)
	}
	data, err := os.ReadFile(tmp)
	if err != nil || size != int64(len(data)) {
		t.Fatalf("temp file not written correctly: err=%v size=%d len=%d", err, size, len(data))
	}
}

func TestDownloadRefusesNonMatchingContentType(t *testing.T) {
	srv := serve(t, "text/html", []byte("<html>"), nil)
	defer srv.Close()
	dir := t.TempDir()
	if _, _, _, err := Download(srv.URL, imageOpts(dir)); err == nil || !strings.Contains(err.Error(), "content-type") {
		t.Fatalf("expected content-type refusal, got %v", err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temp files left behind: %v", entries)
	}
}

func TestDownloadAcceptsExactContentTypeMatch(t *testing.T) {
	srv := serve(t, "audio/mpeg", []byte("ID3\x04\x00\x00\x00\x00\x00\x00"), nil)
	defer srv.Close()
	dir := t.TempDir()
	opts := Options{
		MaxBytes:     1 << 20,
		Timeout:      5 * time.Second,
		ContentTypes: []string{"audio/mpeg", "audio/mp3", "application/octet-stream"},
		OutPath:      filepath.Join(dir, "a.mp3"),
	}
	tmp, _, _, err := Download(srv.URL, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	os.Remove(tmp)
}

func TestDownloadRefusesOversizeContentLengthBeforeDownload(t *testing.T) {
	var served bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Length", "99999999")
		served = true
	}))
	defer srv.Close()
	dir := t.TempDir()
	o := imageOpts(dir)
	o.MaxBytes = 1000
	_, _, _, err := Download(srv.URL, o)
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("expected size refusal, got %v", err)
	}
	_ = served
}

func TestDownloadRefusesOversizeStreamWithoutContentLength(t *testing.T) {
	big := make([]byte, 5000)
	copy(big, pngBytes(t, 2, 2))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Transfer-Encoding", "chunked") // no Content-Length
		w.Write(big)
	}))
	defer srv.Close()
	dir := t.TempDir()
	o := imageOpts(dir)
	o.MaxBytes = 1000
	_, _, _, err := Download(srv.URL, o)
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("expected stream size refusal, got %v", err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temp files left behind: %v", entries)
	}
}

func TestDownloadRefusesHTTPStatusErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gone", http.StatusNotFound)
	}))
	defer srv.Close()
	dir := t.TempDir()
	if _, _, _, err := Download(srv.URL, imageOpts(dir)); err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("expected status refusal, got %v", err)
	}
}

func TestDownloadLeavesNoTempFileBehindOnRefusal(t *testing.T) {
	srv := serve(t, "text/plain", []byte("nope"), nil)
	defer srv.Close()
	dir := t.TempDir()
	Download(srv.URL, imageOpts(dir))
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temp files left behind: %v", entries)
	}
}

func TestDownloadSendsDescriptiveUserAgent(t *testing.T) {
	var ua string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ua = r.Header.Get("User-Agent")
		w.Header().Set("Content-Type", "image/png")
		w.Write(pngBytes(t, 2, 2))
	}))
	defer srv.Close()
	dir := t.TempDir()
	tmp, _, _, err := Download(srv.URL, imageOpts(dir))
	if err != nil {
		t.Fatal(err)
	}
	os.Remove(tmp)
	if !strings.HasPrefix(ua, "mediafetch/") || !strings.Contains(ua, "github.com/cehbz") {
		t.Fatalf("user-agent %q must identify the tool and a contact URL (Wikimedia policy)", ua)
	}
}

func TestCommitMovesTempFileIntoPlace(t *testing.T) {
	dir := t.TempDir()
	tmp, err := os.CreateTemp(dir, ".mediafetch-*")
	if err != nil {
		t.Fatal(err)
	}
	tmp.WriteString("payload")
	tmp.Close()
	dest := filepath.Join(dir, "out.bin")
	if err := Commit(tmp.Name(), dest); err != nil {
		t.Fatalf("Commit failed: %v", err)
	}
	data, err := os.ReadFile(dest)
	if err != nil || string(data) != "payload" {
		t.Fatalf("dest not written correctly: err=%v data=%q", err, data)
	}
	if _, statErr := os.Stat(tmp.Name()); statErr == nil {
		t.Fatal("temp file should no longer exist after Commit")
	}
}
