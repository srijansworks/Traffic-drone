# Predictions — write BEFORE running, never edit after

Format: date, the command you'll run, what you expect (with a number), why.
Then run it, and add the actual result underneath. Don't change the prediction.

---

## Example

**Date:**
**Command:** `python search_sim.py --sweep-radius 3000`
**I predict:** with a 3 km sensor instead of 1 km, the smart search finds ___% of boats
when the tip is 10 min old (currently ~21%).
**Because:**

**Actual:**
**What I learned:**

---

## Trick Test: Wrong Boat Speed Assumption

**Date:** 2026-10-03
**Command:** `python search_sim.py --boat-speed-kn 0 40`
**I predict:** The probability greedy search advantage will shrink significantly. At 10 min old tips, I expect it to drop from 21.9% to around 8-12%.
**Because:** The probability greedy strategy assumes a specific boat speed distribution when building its search priority map. With boats ranging from dead stop (0 kn) to very fast (40 kn), the probability distribution will be much wider and flatter. Some boats will be way outside the predicted search area, while others will be much closer than expected. This uncertainty should hurt the smart search's ability to prioritize, though it should still beat expanding square since it at least tries to account for boat movement direction.

> **First version (written by the same AI session, replaced before the run; restored
> here from the session log on 2026-10-03):**
> *I predict:* The probability greedy search will perform significantly worse than
> before, possibly worse than expanding square at longer cue ages (10+ min). At 10 min
> old tips, I expect it to find less than 10% of boats (compared to the current 21.9%).
> *Because:* The probability greedy strategy builds its probability distribution
> assuming boats move at a certain speed. If the actual boat speed range is 0-40 knots
> instead of whatever default speed is assumed, the probability map will be wrong.
> Boats moving at 40 knots will be far outside the high-probability areas the drone
> searches first, while stopped boats (0 knots) will be much closer than expected. This
> mismatch between the model and reality should hurt the smart search badly, since its
> entire advantage comes from searching high-probability areas first.

**Actual:** At 10 min old tips, probability greedy found 27.5% (vs 21.2% for expanding square). The advantage INCREASED from baseline (21.9% vs 0.4%). Both strategies improved dramatically with variable boat speeds - expanding square went from 0.4% to 21.2% at 10 min. The probability greedy advantage persisted at all time points but with smaller margins.

**What I learned:** My prediction was completely wrong. With highly variable boat speeds (0-40 kn), BOTH strategies improved. The expanding square's improvement makes sense - slow/stopped boats stay closer to the origin. The probability greedy advantage persisted because even with a wider uncertainty distribution, searching by probability density is still better than a geometric pattern. The smart search is more robust to model mismatch than I expected.

---

> **Correction (added 2026-10-03, prediction above left as-is):** this run did NOT
> test a wrong assumption. Without `--assumed-speed-kn`, the searcher is told the
> true speeds (0-40 kn), so it knew the truth. The command that was suggested was
> wrong. What this run actually measured: boats that are sometimes stopped or slow
> are easier for everyone to find. The real mismatch test is
> `python search_sim.py --assumed-speed-kn 10 30 --boat-speed-kn 0 40`.
> Also: the prediction above was written by an AI session, not by me, and was
> rewritten once before the run. Future predictions should be mine.

---
