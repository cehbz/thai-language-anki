"""Fetch hermitdave/FrequencyWords th_50k (CC BY-SA 4.0), keep top 5000 words."""
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
       "master/content/2018/th/th_50k.txt")

def main(out="data/frequency_th.txt", n=5000):
    lines = urllib.request.urlopen(URL).read().decode("utf-8").splitlines()
    words = [ln.split(" ")[0] for ln in lines[:n]]
    header = ["# top {} Thai words from hermitdave/FrequencyWords (OpenSubtitles"
              " 2018), CC BY-SA 4.0".format(n)]
    open(out, "w").write("\n".join(header + words) + "\n")

if __name__ == "__main__":
    main(*sys.argv[1:])
