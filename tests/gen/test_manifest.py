import yaml
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


def test_record_persists_immediately_when_the_manifest_knows_its_deck(tmp_path):
    """A media file on disk must never outlive its provenance: a run killed
    inside a filler leaves entries for everything already fetched."""
    from thai_deck_gen.media.manifest import Manifest, MediaEntry
    Manifest.load(tmp_path).record(MediaEntry(
        file="media/images/pw-0.jpg", channel="openverse",
        origin="http://x/y.jpg", fetched="2026-08-29"))
    assert Manifest.load(tmp_path).channel_of("media/images/pw-0.jpg") == "openverse"


def test_record_without_a_deck_root_stays_in_memory(tmp_path):
    from thai_deck_gen.media.manifest import Manifest, MediaEntry
    m = Manifest()
    m.record(MediaEntry(file="media/images/pw-0.jpg", channel="openverse",
                        origin="http://x/y.jpg", fetched="2026-08-29"))
    assert not (tmp_path / "media_manifest.yaml").exists()
    assert m.channel_of("media/images/pw-0.jpg") == "openverse"


def test_repeated_records_keep_the_latest_and_save_compacts(tmp_path):
    from thai_deck_gen.media.manifest import Manifest, MediaEntry
    m = Manifest.load(tmp_path)
    for channel in ("wikimedia", "openverse"):
        m.record(MediaEntry(file="media/images/pw-0.jpg", channel=channel,
                            origin=f"http://{channel}", fetched="2026-08-29"))
    assert Manifest.load(tmp_path).channel_of("media/images/pw-0.jpg") == "openverse"
    m.save(tmp_path)
    raw = yaml.safe_load((tmp_path / "media_manifest.yaml").read_text(encoding="utf-8"))
    assert len(raw["entries"]) == 1          # save() rewrites canonically


def test_recording_appends_rather_than_rewriting_every_entry(tmp_path):
    """Provenance is written per item; rewriting the whole file each time is
    quadratic over a deck-sized run."""
    from thai_deck_gen.media.manifest import Manifest, MediaEntry
    m = Manifest.load(tmp_path)
    rewrites = []
    original_save = Manifest.save
    Manifest.save = lambda self, root: (rewrites.append(len(self.entries)),
                                        original_save(self, root))[1]
    try:
        for i in range(3):
            m.record(MediaEntry(file=f"media/images/pw-{i}.jpg", channel="openverse",
                                origin=f"http://x/{i}.jpg", fetched="2026-08-29"))
    finally:
        Manifest.save = original_save
    assert rewrites == []                      # no full rewrite per record
    assert len(Manifest.load(tmp_path).entries) == 3
