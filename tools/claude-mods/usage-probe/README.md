# usage-probe

A Claude Code mod (≥ 2.1.287) for `tools/quota_pacer.py`: after each turn it
writes `$.session.usage()`'s `rateLimits` (`five_hour`, `seven_day`), `context`
and `cost`, with `read_at`, as JSON to the file named by
`THAI_SYLLABUS_USAGE_OUT`. No UI. Loaded per call with `--plugin-dir`.
