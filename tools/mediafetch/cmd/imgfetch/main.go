package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"mediafetch/internal/fetch"
)

func main() {
	maxBytes := flag.Int64("max-bytes", 10<<20, "refuse responses larger than this")
	allow := flag.String("allow", "jpeg,png,gif,webp", "comma-separated image formats to accept")
	timeout := flag.Duration("timeout", 30*time.Second, "whole-request timeout")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: imgfetch [flags] <url> <out-path>\n\nFetch one URL to out-path only if it is a real image within limits.\nPrints a JSON line {format,width,height,bytes} on success, {refused,detail} on refusal.\n\n")
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
		kind := "io"
		var r *fetch.Refusal
		if errors.As(err, &r) {
			kind = r.Kind
		}
		line := map[string]string{"refused": kind, "detail": err.Error()}
		if r != nil && r.Body != "" {
			line["body"] = r.Body
		}
		json.NewEncoder(os.Stdout).Encode(line)
		fmt.Fprintf(os.Stderr, "imgfetch: refused %s: %v\n", flag.Arg(0), err)
		os.Exit(1)
	}
	json.NewEncoder(os.Stdout).Encode(res)
}
