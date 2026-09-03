package fetch

// LooksLikeMP3 reports whether head starts an MPEG audio stream: an ID3v2
// tag or an MPEG frame sync (11 set bits). Header check only; duration
// and decodability are the mechanical assessor's job (ffprobe).
func LooksLikeMP3(head []byte) bool {
	if len(head) >= 3 && head[0] == 'I' && head[1] == 'D' && head[2] == '3' {
		return true
	}
	return len(head) >= 2 && head[0] == 0xFF && head[1]&0xE0 == 0xE0
}
