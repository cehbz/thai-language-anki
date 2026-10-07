# systemd: the quota pacer and the review site

User units, instanced by the deck's directory name under `~/decks`
(`thai-syllabus-pacer@thai-ff`). `thai-syllabus-pacer@.timer` runs
`tools/quota_pacer.py --require-measured` hourly (spec 3 §8 `pacer`), so a
deck whose `pacer` block does not set `points_per_call` and
`session_points_per_call` never runs on the schedule.
`thai-syllabus-review@.service` keeps the review site up on port 8877, bound
to the address providers.yaml names. `PATH` names where `claude` and `uv`
(`~/.local/bin`) live; the logs go to the deck's `work/`. Linger keeps the
units running without a login session (`loginctl enable-linger`).
Install:

```sh
mkdir -p ~/.config/systemd/user ~/decks/thai-ff/work
cp ~/projects/thai-language-anki/tools/systemd/thai-syllabus-*@.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now thai-syllabus-pacer@thai-ff.timer thai-syllabus-review@thai-ff.service
```

Each tick syncs the deck's own Anki collection with AnkiWeb (providers.yaml
`anki`, default `~/decks/<deck>/anki/collection.anki2`) and harvests reviews, flags
and ReviewNotes from it; after a tick that changed the deck the pacer
compiles it (never with `--force`), imports it into that collection and
syncs again (spec 4 §4). Log in to AnkiWeb once so the pacer holds a sync
key: `uv run thai-syllabus anki-login --deck ~/decks/thai-ff` writes it to the
owner-only file `anki.sync_key` names. A tick whose sync fails or meets a
full sync other than the first download, or whose build gate is closed,
logs why and leaves the import pending in `work/pacer-import.json`
for a later tick. `pacer: {import: false}` turns off the sync and the
harvest as well as the import.

Try a tick without running a cycle:
`uv run python tools/quota_pacer.py --deck ~/decks/thai-ff --dry-run --require-measured`.
Run a tick now: `systemctl --user start thai-syllabus-pacer@thai-ff.service`.
Remove: `systemctl --user disable --now thai-syllabus-pacer@thai-ff.timer thai-syllabus-review@thai-ff.service`.
