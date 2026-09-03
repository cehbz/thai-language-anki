package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

func main() {
	maxBytes := flag.Int64("max-bytes", 20<<20, "refuse responses larger than this")
	allow := flag.String("allow", "mp3", "comma-separated audio formats to accept")
	timeout := flag.Duration("timeout", 30*time.Second, "whole-request timeout")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: audiofetch [flags] <url> <out-path>\n\nFetch one URL to out-path only if it is a real MP3 within limits.\nPrints a JSON line {format,bytes} on success.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 2 {
		flag.Usage()
		os.Exit(2)
	}
	opts := Options{MaxBytes: *maxBytes, Allow: strings.Split(*allow, ","), Timeout: *timeout}
	res, err := Fetch(flag.Arg(0), flag.Arg(1), opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "audiofetch: refused %s: %v\n", flag.Arg(0), err)
		os.Exit(1)
	}
	json.NewEncoder(os.Stdout).Encode(res)
}
