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

After a tick that changed the deck the pacer compiles it (never with
`--force`) and imports it into Anki through AnkiConnect (spec 3 §8
`pacer.import`, `pacer.anki_connect_url`). Install AnkiConnect once: in
Anki, Tools > Add-ons > Get Add-ons, code `2055492159`, then restart Anki.
AnkiConnect serves only while Anki is open, so imports happen only while
Anki runs; a tick with Anki closed or the build gate closed logs why and
leaves the import pending in `$DECK/work/pacer-import.json` for a later
tick.

Each tick first harvests reviews, flags and ReviewNotes into the record:
through AnkiConnect while Anki is open (its collection file is locked
then), else from `pacer.collection_path` (default
`~/Library/Application Support/Anki2/User 1/collection.anki2`).
`pacer: {import: false}` turns off the harvest as well as the import.

Try a tick without running a cycle:
`uv run python tools/quota_pacer.py --deck $DECK --dry-run --require-measured`.
Run a tick now: `launchctl kickstart gui/$(id -u)/com.ceh.thai-syllabus.pacer`.
Remove: `launchctl bootout gui/$(id -u)/com.ceh.thai-syllabus.pacer`.
