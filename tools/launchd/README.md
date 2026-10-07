# launchd: the quota pacer

`com.ceh.thai-syllabus.pacer.plist` runs `tools/quota_pacer.py
--require-measured` hourly (spec 3 §8 `pacer`), so a deck whose `pacer`
block does not set `points_per_call` and `session_points_per_call` never
runs on the schedule. `PATH` names where `claude` (`~/.local/bin`) and
`uv` (`/opt/homebrew/bin`) live. The log goes beside the deck's other run logs.
Install:

```sh
REPO=$HOME/projects/thai-language-anki DECK=$HOME/decks/thai-ff LOG=$DECK/work/pacer.log
mkdir -p "$(dirname "$LOG")"
sed -e "s|__REPO__|$REPO|g" -e "s|__DECK__|$DECK|g" -e "s|__LOG__|$LOG|g" \
    -e "s|__HOME__|$HOME|g" \
  "$REPO/tools/launchd/com.ceh.thai-syllabus.pacer.plist" \
  > ~/Library/LaunchAgents/com.ceh.thai-syllabus.pacer.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ceh.thai-syllabus.pacer.plist
```

Each tick syncs the deck's own Anki collection with AnkiWeb (providers.yaml
`anki`, default `$DECK/anki/collection.anki2`) and harvests reviews, flags
and ReviewNotes from it; after a tick that changed the deck the pacer
compiles it (never with `--force`), imports it into that collection and
syncs again (spec 4 §4). Log in to AnkiWeb once so the pacer holds a sync
key: `uv run thai-syllabus anki-login --deck $DECK` writes it to the
owner-only file `anki.sync_key` names. A tick whose sync fails or meets a
full sync other than the first download, or whose build gate is closed,
logs why and leaves the import pending in `$DECK/work/pacer-import.json`
for a later tick. `pacer: {import: false}` turns off the sync and the
harvest as well as the import.

Try a tick without running a cycle:
`uv run python tools/quota_pacer.py --deck $DECK --dry-run --require-measured`.
Run a tick now: `launchctl kickstart gui/$(id -u)/com.ceh.thai-syllabus.pacer`.
Remove: `launchctl bootout gui/$(id -u)/com.ceh.thai-syllabus.pacer`.
