package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func serve(t *testing.T, contentType string, body []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		w.Write(body)
	}))
}

func opts() Options {
	return Options{MaxBytes: 1 << 20, Allow: []string{"mp3"}, Timeout: 5 * time.Second}
}

func TestFetchAcceptsValidMP3(t *testing.T) {
	body := append([]byte("ID3\x04\x00\x00\x00\x00\x00\x00"), []byte("...rest of file...")...)
	srv := serve(t, "audio/mpeg", body)
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "a.mp3")
	res, err := Fetch(srv.URL, out, opts())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Format != "mp3" || res.Bytes != int64(len(body)) {
		t.Fatalf("bad result: %+v", res)
	}
	data, err := os.ReadFile(out)
	if err != nil || string(data) != string(body) {
		t.Fatalf("output not written correctly: err=%v", err)
	}
}

func TestFetchRefusesHTMLBody(t *testing.T) {
	srv := serve(t, "text/html", []byte("<html>not audio</html>"))
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "a.mp3")
	_, err := Fetch(srv.URL, out, opts())
	if err == nil || !strings.Contains(err.Error(), "content-type") {
		t.Fatalf("expected content-type refusal, got %v", err)
	}
	if _, statErr := os.Stat(out); statErr == nil {
		t.Fatal("output must not exist after refusal")
	}
}

func TestFetchRefusesBytesThatAreNotMP3(t *testing.T) {
	// application/octet-stream is an accepted content-type, so this refusal
	// must come from the MP3 header sniff, not the content-type check.
	srv := serve(t, "application/octet-stream", []byte("definitely not an mp3 stream"))
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "a.mp3")
	_, err := Fetch(srv.URL, out, opts())
	if err == nil || !strings.Contains(err.Error(), "MP3") {
		t.Fatalf("expected mp3-sniff refusal, got %v", err)
	}
	if _, statErr := os.Stat(out); statErr == nil {
		t.Fatal("output must not exist after refusal")
	}
}

func TestFetchRefusesDisallowedFormat(t *testing.T) {
	body := []byte("ID3\x04\x00\x00\x00\x00\x00\x00")
	srv := serve(t, "audio/mpeg", body)
	defer srv.Close()
	o := opts()
	o.Allow = []string{"wav"}
	_, err := Fetch(srv.URL, filepath.Join(t.TempDir(), "a.mp3"), o)
	if err == nil || !strings.Contains(err.Error(), "format") {
		t.Fatalf("expected format refusal, got %v", err)
	}
}

func TestFetchLeavesNoTempFileBehind(t *testing.T) {
	srv := serve(t, "audio/mpeg", []byte("nope"))
	defer srv.Close()
	dir := t.TempDir()
	Fetch(srv.URL, filepath.Join(dir, "a.mp3"), opts())
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temp files left behind: %v", entries)
	}
}

// Same cleanup guarantee, but through the -allow-mismatch branch rather
// than the MP3-sniff-failure branch TestFetchLeavesNoTempFileBehind
// exercises above — both refusal paths share the same deferred os.Remove
// in Fetch.
func TestFetchLeavesNoTempFileBehindOnDisallowedFormat(t *testing.T) {
	body := []byte("ID3\x04\x00\x00\x00\x00\x00\x00")
	srv := serve(t, "audio/mpeg", body)
	defer srv.Close()
	dir := t.TempDir()
	o := opts()
	o.Allow = []string{"wav"}
	Fetch(srv.URL, filepath.Join(dir, "a.mp3"), o)
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temp files left behind: %v", entries)
	}
}
