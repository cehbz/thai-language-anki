"""Fetch hermitdave/FrequencyWords th_50k (CC BY-SA 4.0), keep top 5000
Thai-script single-token words."""
import re
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
       "master/content/2018/th/th_50k.txt")

# Thai-script single tokens only: no spaces, no Latin, no ASCII digits, no
# punctuation. The source list is raw OpenSubtitles frequency data and
# includes plenty of non-Thai/Latin/interjection noise ("you", "the", "ok",
# ๆ, etc.) that pollutes the reference list; this filter excludes it.
_THAI_TOKEN = re.compile(r"^[ก-๛]+$")

def main(out="data/frequency_th.txt", n=5000):
    lines = urllib.request.urlopen(URL).read().decode("utf-8").splitlines()
    words = []
    for ln in lines:
        word = ln.split(" ")[0]
        if _THAI_TOKEN.match(word):
            words.append(word)
        if len(words) >= n:
            break
    header = ["# top {} Thai-script single-token words from "
              "hermitdave/FrequencyWords (OpenSubtitles 2018), CC BY-SA 4.0"
              " -- filtered to ^[ก-๛]+$ (Thai script only, single token,"
              " no Latin/digits/punctuation)".format(n)]
    open(out, "w").write("\n".join(header + words) + "\n")

if __name__ == "__main__":
    main(*sys.argv[1:])
