from thai_deck_eval.data_io import (FileFrequencyList, load_contrasts,
                                    load_function_words, load_g2p_exceptions,
                                    load_spelling_targets)

def test_contrasts_load_and_weights():
    entries = load_contrasts()
    ids = {e.id for e in entries}
    assert "tone:mid-low" in ids
    assert max(entries, key=lambda e: e.weight).id == "tone:mid-low"

def test_spelling_targets_counts():
    t = load_spelling_targets()
    assert len(t["consonants"]) == 42 and len(t["tone_marks"]) == 4

def test_function_words():
    assert "ที่" in load_function_words()

def test_g2p_exceptions():
    assert load_g2p_exceptions()["น้ำ"] == "naːm˦˥"

def test_frequency_list(tmp_path):
    p = tmp_path / "freq.txt"
    p.write_text("# header\nที่\nของ\n")
    fl = FileFrequencyList(p)
    assert fl.rank("ที่") == 1 and fl.rank("ของ") == 2 and fl.rank("x") is None
