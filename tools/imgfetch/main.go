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
	maxBytes := flag.Int64("max-bytes", 10<<20, "refuse responses larger than this")
	allow := flag.String("allow", "jpeg,png,gif,webp", "comma-separated image formats to accept")
	timeout := flag.Duration("timeout", 30*time.Second, "whole-request timeout")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: imgfetch [flags] <url> <out-path>\n\nFetch one URL to out-path only if it is a real image within limits.\nPrints a JSON line {format,width,height,bytes} on success.\n\n")
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
		fmt.Fprintf(os.Stderr, "imgfetch: refused %s: %v\n", flag.Arg(0), err)
		os.Exit(1)
	}
	json.NewEncoder(os.Stdout).Encode(res)
}
