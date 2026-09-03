package fetch

import "testing"

func TestLooksLikeMP3(t *testing.T) {
	cases := []struct {
		name string
		head []byte
		want bool
	}{
		{"id3v2", []byte("ID3\x04\x00\x00\x00\x00\x00\x00"), true},
		{"frame sync", []byte{0xFF, 0xFB, 0x90, 0x00, 0x00}, true},
		{"png", []byte("\x89PNG\r\n\x1a\n"), false},
		{"short", []byte{0xFF}, false},
		{"empty", nil, false},
	}
	for _, c := range cases {
		if got := LooksLikeMP3(c.head); got != c.want {
			t.Errorf("%s: LooksLikeMP3 = %v, want %v", c.name, got, c.want)
		}
	}
}
