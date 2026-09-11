package fetch

import (
	"bytes"
	"errors"
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

func TestDownloadRefusalIsTypedHTTP(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gone", http.StatusNotFound)
	}))
	defer srv.Close()
	dir := t.TempDir()
	_, _, _, err := Download(srv.URL, imageOpts(dir))
	var r *Refusal
	if !errors.As(err, &r) {
		t.Fatalf("expected a *Refusal, got %v (%T)", err, err)
	}
	if r.Kind != "http" {
		t.Fatalf("Kind = %q, want %q", r.Kind, "http")
	}
}

func TestDownloadRefusalIsTypedContentType(t *testing.T) {
	srv := serve(t, "text/html", []byte("<html>"), nil)
	defer srv.Close()
	dir := t.TempDir()
	_, _, _, err := Download(srv.URL, imageOpts(dir))
	var r *Refusal
	if !errors.As(err, &r) {
		t.Fatalf("expected a *Refusal, got %v (%T)", err, err)
	}
	if r.Kind != "content-type" {
		t.Fatalf("Kind = %q, want %q", r.Kind, "content-type")
	}
}

// TestDownloadContentTypeRefusalCarriesBody covers Forvo's daily-limit
// defect (task 7 brief): its mp3 urls serve a JSON body
// `["Limit/day reached."]` as application/json, refused as content-type.
// Without the body on the Refusal, attempts.py cannot tell that refusal
// apart from any other content-type refusal and treats a spent quota as
// a transient failure (spec 3 section 6a).
func TestDownloadContentTypeRefusalCarriesBody(t *testing.T) {
	body := []byte(`["Limit/day reached."]`)
	srv := serve(t, "application/json; charset=utf-8", body, nil)
	defer srv.Close()
	dir := t.TempDir()
	_, _, _, err := Download(srv.URL, imageOpts(dir))
	var r *Refusal
	if !errors.As(err, &r) {
		t.Fatalf("expected a *Refusal, got %v (%T)", err, err)
	}
	if r.Kind != "content-type" {
		t.Fatalf("Kind = %q, want %q", r.Kind, "content-type")
	}
	if r.Body != string(body) {
		t.Fatalf("Body = %q, want %q", r.Body, string(body))
	}
}

// TestDownloadContentTypeRefusalBodyIsTruncatedTo512Bytes covers the
// brief's "first 512 bytes" bound: an oversized refusal body must not
// grow the refusal line without limit.
func TestDownloadContentTypeRefusalBodyIsTruncatedTo512Bytes(t *testing.T) {
	big := bytes.Repeat([]byte("a"), 600)
	srv := serve(t, "application/json", big, nil)
	defer srv.Close()
	dir := t.TempDir()
	_, _, _, err := Download(srv.URL, imageOpts(dir))
	var r *Refusal
	if !errors.As(err, &r) {
		t.Fatalf("expected a *Refusal, got %v (%T)", err, err)
	}
	if len(r.Body) != 512 {
		t.Fatalf("len(Body) = %d, want 512", len(r.Body))
	}
}

// TestDownloadOtherRefusalKindsCarryNoBody covers the brief's "a
// content-type Refusal carries ... a new field Body": every other kind
// leaves Body empty.
func TestDownloadOtherRefusalKindsCarryNoBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gone", http.StatusNotFound)
	}))
	defer srv.Close()
	dir := t.TempDir()
	_, _, _, err := Download(srv.URL, imageOpts(dir))
	var r *Refusal
	if !errors.As(err, &r) {
		t.Fatalf("expected a *Refusal, got %v (%T)", err, err)
	}
	if r.Body != "" {
		t.Fatalf("Body = %q, want empty", r.Body)
	}
}

func TestDownloadRefusalIsTypedTooLarge(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Length", "99999999")
	}))
	defer srv.Close()
	dir := t.TempDir()
	o := imageOpts(dir)
	o.MaxBytes = 1000
	_, _, _, err := Download(srv.URL, o)
	var r *Refusal
	if !errors.As(err, &r) {
		t.Fatalf("expected a *Refusal, got %v (%T)", err, err)
	}
	if r.Kind != "too-large" {
		t.Fatalf("Kind = %q, want %q", r.Kind, "too-large")
	}
}

func TestDownloadRefusalIsTypedWireOnUnreachableHost(t *testing.T) {
	dir := t.TempDir()
	_, _, _, err := Download("http://127.0.0.1:1", imageOpts(dir))
	var r *Refusal
	if !errors.As(err, &r) {
		t.Fatalf("expected a *Refusal, got %v (%T)", err, err)
	}
	if r.Kind != "wire" {
		t.Fatalf("Kind = %q, want %q", r.Kind, "wire")
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
