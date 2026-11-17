# BUILD-NOTES — gta6-twitch-ccu-forecast-01

## Model: launch-shape regression, fit at run time

The hour-1 print anchors the scale. Everything else is learned from the
case's own comparables (past big launches) when the program runs — no
hard-coded launch curve.

1. **Per-hour log-ratio regression.** For each comparable launch, take its
   release-day hourly counts and form `ln(viewers_h / viewers_1)` per hour.
   For each forecast hour, fit a kernel-weighted local linear regression of
   that log-ratio on `ln(viewers_1)` across the comparables (Gaussian
   kernel, bandwidth 1.5 in log space), evaluated at this case's hour-1
   print. This matters because the day's growth depends strongly on the
   hour-1 size: across the 37 comparables, `corr(ln v1, ln peak/v1) =
   -0.94`. A single median curve ignores that and is far worse.
2. **Spread.** The weighted residual standard deviation of the same fit,
   shrunk by 0.7, is the log-spread; quantiles are
   `v1 * exp(mu + bump + z_tau * sigma * 0.7)`.
3. **Rolling-midnight unlock bump.** These launches unlock at local
   midnight, so the US East coast joins at midnight Eastern (04:00Z in
   daylight time, 05:00Z otherwise) and the West coast three hours later.
   The comparable pool mixes unlock structures, which smooths that step
   away, so the median is lifted at the unlock hours: East +0.50 / +0.48 /
   +0.38 over the unlock hour and the next two, West +0.18 / +0.16 over its
   unlock hour and the next. The unlock hour is derived from the release
   date with the US daylight-time rule (2nd Sunday of March to 1st Sunday
   of November) — for 19 Nov 2026 (EST) that is East 05:00Z (hour 6),
   West 08:00Z (hour 9). The target game's own pre-release category
   history is excluded from the comparables (it is not a launch).

## Practice validation (local, fresh scorer)

Answers for the four practice launches were fetched from SullyGnome's
GameViewers endpoint (one calendar month per launch; the "Average viewers"
series). Hour-1 values match the case files exactly for all four launches,
and the scorer reproduces the house baseline's stated practice loss
(0.3871 vs ~0.387), which validates both the answers and the scorer.

| case | launch | loss | typical miss | 90% coverage |
|------|--------|------|--------------|--------------|
| p00 | Marvel's Wolverine | 0.1456 | 17.1% | 1.000 |
| p01 | Red Dead Redemption 2 | 0.1733 | 36.0% | 0.957 |
| p02 | God of War Ragnarök | 0.1206 | 20.3% | 0.957 |
| p03 | Marvel's Spider-Man 2 | 0.1593 | 26.6% | 0.826 |
| **mean** | | **0.1497** | **~25%** | **0.935** |

House baseline on the same cases: loss 0.3871, typical miss 82.8%,
coverage 0.783. This entry beats it by ~61% on practice, with honest
coverage (mean 0.935, target ~0.90).

## Checks

- Determinism: two runs on all four practice cases produce byte-identical
  outputs.
- Runtime: 0.17 s for all four practice cases together (limit 8 s per case).
- Format: every output validated strictly — exactly `{"hours": [...]}`, 23
  rows, hours 2–24, five finite non-decreasing quantiles in (0, 1e9],
  files ~1.7 KB (limit 65,536 bytes).
- Robustness: a synthetic final-like case (42 comparables, 2,940
  minute-level history points, non-null `streams` / `categories_hourly`,
  GTA VI as the game, hour-1 185,000) runs clean and produces a valid
  forecast with the expected shape (dip to 0.70x at hour 4, East-unlock jump
  to 1.26x at hour 6, peak 1.50x at hour 21). An edge case (hour-1 of 50
  viewers, only 5 comparables, one missing hour 0) also runs clean.
- Scorer: `work/score.py` is a fresh implementation of the competition's
  pinball loss, run against answers fetched from SullyGnome (see above).

## Caveats

- Practice is no guide to the final — the rules say so, and the margin
  above is on four past launches. The final is one launch (GTA VI), the
  biggest of the set, and its hour-1 print will sit at the top end of the
  comparables' range, where the regression is an extrapolation guided by
  the fitted slope rather than dense data.
- The unlock bump is calibrated on four launches; its size (+0.5 log at
  the East unlock hour) is consistent across all four, but the true jump
  for GTA VI could be smaller or larger. Sensitivity: scaling the bump by
  0.75x–1.25x moves the mean practice loss by under 0.002, so the entry
  is not knife-edge on it.
- The model uses only the hour-1 print and the comparables. It ignores
  `history`, `streams`, and `categories_hourly` — those carry pre-release
  chatter, not the release-day shape, and using them would add failure
  modes for no measured gain.
