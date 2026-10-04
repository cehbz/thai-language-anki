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

Try a tick without running a cycle:
`uv run python tools/quota_pacer.py --deck $DECK --dry-run --require-measured`.
Run a tick now: `launchctl kickstart gui/$(id -u)/com.ceh.thai-syllabus.pacer`.
Remove: `launchctl bootout gui/$(id -u)/com.ceh.thai-syllabus.pacer`.
