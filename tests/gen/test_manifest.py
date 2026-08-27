from thai_deck_gen.media.manifest import Manifest, MediaEntry

def _entry(f="media/audio/x.mp3"):
    return MediaEntry(file=f, channel="forvo", origin="https://x",
                      speaker="forvo:joe", fetched="2026-08-27")

def test_manifest_round_trip(tmp_path):
    m = Manifest.load(tmp_path)
    assert m.entries == {}
    m.record(_entry())
    m.save(tmp_path)
    m2 = Manifest.load(tmp_path)
    assert m2.channel_of("media/audio/x.mp3") == "forvo"
