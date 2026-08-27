"""Build data/frequency_th.txt: a top-5000 Thai word frequency list blended
from two sources, weighted toward daily spoken/colloquial Thai (this
project's use case is a spoken-Thai flashcard deck, not written registers).

Sources:
  * hermitdave/FrequencyWords th_50k (OpenSubtitles 2018 subtitle corpus),
    CC BY-SA 4.0: https://github.com/hermitdave/FrequencyWords -- colloquial/
    spoken-register proxy (film & TV dialogue), but noisy: raw lines include
    mojibake, Latin script, and punctuation.
  * pythainlp's bundled TNC (Thai National Corpus) word_freqs(), used here
    as a written-register robustness/plausibility check on the subtitle
    list's ranks (pythainlp is BSD-3-licensed; the TNC data it bundles is
    redistributed under pythainlp's corpus license -- see
    https://github.com/PyThaiNLP/pythainlp for corpus-specific terms).

Filter: both sources are restricted to `^[ก-๛]+$` (Thai-script-only, single
token, no Latin/digits/punctuation/spaces). The subtitle list is additionally
required to be a `pythainlp.corpus.thai_words()` dictionary entry, since
OpenSubtitles' raw frequency dump is otherwise dominated by mojibake and
transliteration noise that isn't real Thai vocabulary. TNC's entries are
corpus-attested by construction (real running text), so no dictionary
membership filter is applied there -- only the script regex.

Blend: each source is independently sorted by descending frequency/count and
converted to a normalized rank `r_norm = rank / N` (0 = most frequent word in
that source), with `r_norm = 1.0` (worst-case penalty) for a word absent from
a given source. `score = 0.7 * subtitle_r_norm + 0.3 * tnc_r_norm`, ascending
(lower = more frequent); the top 5000 words by score are kept. The 0.7/0.3
split favors the colloquial/subtitle source over the written-corpus source,
per this project's goal (this is a daily-conversational-Thai deck, and TNC
skews toward news/formal writing); TNC still meaningfully re-ranks words
that are subtitle-common but written-rare (or vice versa) rather than being
purely decorative.
"""
import re
import sys
import urllib.request

SUBTITLE_URL = ("https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
                "master/content/2018/th/th_50k.txt")

# Thai-script single tokens only: no spaces, no Latin, no ASCII digits, no
# punctuation. Both sources include plenty of non-Thai/noise ("you", "the",
# "ok", ๆ, mojibake, etc.) that this filter excludes.
_THAI_TOKEN = re.compile(r"^[ก-๛]+$")


def _is_thai_token(word: str) -> bool:
    return bool(_THAI_TOKEN.match(word))


def _filter_thai_tokens(words: list[str]) -> list[str]:
    """Keep only Thai-script single tokens, preserving order."""
    return [w for w in words if _is_thai_token(w)]


def _filter_dictionary_words(words: list[str], dictionary: set[str]) -> list[str]:
    """Keep only Thai-script tokens that are also dictionary entries,
    preserving order."""
    return [w for w in words if _is_thai_token(w) and w in dictionary]


def _ranks(words: list[str]) -> dict[str, int]:
    """1-based rank by position (words must already be frequency-sorted,
    descending)."""
    return {w: i + 1 for i, w in enumerate(words)}


def _blend(subtitle_words: list[str], tnc_words: list[str],
          w_subtitle: float = 0.7, w_tnc: float = 0.3) -> list[str]:
    """Blend two frequency-sorted (descending) word lists into a single
    ranking via normalized-rank scoring (see module docstring). Returns
    words sorted ascending by score (most frequent/best-supported first)."""
    sub_rank = _ranks(subtitle_words)
    tnc_rank = _ranks(tnc_words)
    n_sub, n_tnc = len(subtitle_words), len(tnc_words)
    vocab = set(sub_rank) | set(tnc_rank)

    def score(w: str) -> float:
        r_sub = sub_rank[w] / n_sub if w in sub_rank else 1.0
        r_tnc = tnc_rank[w] / n_tnc if w in tnc_rank else 1.0
        return w_subtitle * r_sub + w_tnc * r_tnc

    return sorted(vocab, key=score)


def _fetch_subtitle_words(url: str = SUBTITLE_URL) -> list[str]:
    lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
    return [ln.split(" ")[0] for ln in lines if ln.strip()]


def _fetch_tnc_words() -> list[str]:
    from pythainlp.corpus import tnc
    pairs = tnc.word_freqs()  # list[tuple[word, count]], not pre-sorted
    pairs.sort(key=lambda p: p[1], reverse=True)
    return [w for w, _count in pairs]


def _thai_dictionary() -> set[str]:
    from pythainlp.corpus import thai_words
    return set(thai_words())


_HEADER = """\
# top {n} Thai words, blended from two sources (see scripts/fetch_frequency.py
# for full rationale):
#   - hermitdave/FrequencyWords th_50k (OpenSubtitles 2018), CC BY-SA 4.0,
#     https://github.com/hermitdave/FrequencyWords -- filtered to Thai-script
#     single tokens (^[ก-๛]+$) present in pythainlp.corpus.thai_words()
#   - pythainlp's bundled TNC (Thai National Corpus) word_freqs() --
#     filtered to Thai-script single tokens only (corpus-attested, no
#     dictionary-membership filter applied)
# Blend: score = 0.7 * subtitle_normalized_rank + 0.3 * tnc_normalized_rank,
# ascending (word missing from a source gets that source's worst-case
# normalized rank, 1.0). The 0.7/0.3 split weights this list toward
# colloquial/spoken Thai (subtitle dialogue) over TNC's more written/formal
# register, matching this project's daily-spoken-Thai deck goal.
"""


def main(out: str = "data/frequency_th.txt", n: int = 5000,
        w_subtitle: float = 0.7, w_tnc: float = 0.3) -> list[str]:
    n = int(n)
    subtitle_raw = _fetch_subtitle_words()
    dictionary = _thai_dictionary()
    subtitle_words = _filter_dictionary_words(subtitle_raw, dictionary)

    tnc_raw = _fetch_tnc_words()
    tnc_words = _filter_thai_tokens(tnc_raw)

    blended = _blend(subtitle_words, tnc_words, w_subtitle, w_tnc)[:n]
    header = _HEADER.format(n=n)
    open(out, "w").write(header + "\n".join(blended) + "\n")
    return blended


if __name__ == "__main__":
    main(*sys.argv[1:])
