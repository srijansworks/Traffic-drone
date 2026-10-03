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

## Ring search vs smart search

**Date:** 2026-10-03
**Command:** `python search_sim.py`
**I predict (my words):** "I think that the smart search isn't something with an absurd
improvement in findings, but I think it will beat it by 5%."
Meaning: at a 10 min old tip, smart search finds about 5 percentage points more boats
than ring search (so if smart is ~22%, ring is ~17%).
**Because:** Ring search also knows the boat has moved, so smart search shouldn't have a
huge edge.

**Actual:** At a 10 min old tip, smart search found 21.9% of boats and ring search
found 14.8%. Smart search won by about 7 points, a bit more than my guess of 5.
The ranges (95% CI) don't overlap (smart 20.1–23.8%, ring 13.3–16.4%), so the win
is real, not luck. (2,000 boats per test.)

Surprise: smart search does NOT always win.
- 2 min old tip: ring search found MORE (85.1% vs 81.8%).
- 4 to 20 min old tip: smart search wins (biggest gap around 8–20 min).
- 30+ min old tip: they're almost tied, and both find fewer than 6 out of 100 boats.

**What I learned:**
- My guess was close. Smart search is better, but not by a crazy amount. It beat
  ring search by about 7 points, not 5.
- Another way to say it: at 10 min, smart search finds about half again as many
  boats (22 vs 15 out of 100).
- The "best" search depends on how old the tip is. When the tip is super fresh,
  just flying out and circling (ring search) works slightly better. Smart search
  only pulls ahead once the boat has had time to get far away.
- When the tip is really old (30+ min), how you search barely matters. The boat
  could be in too big an area for any one drone to cover.
- So the biggest win isn't a smarter search. It's getting the tip to the drone faster.
- One thing to check next: smart search plans its path using random guesses. I
  should rerun with a different `--seed` to make sure the 7-point win isn't just
  from one lucky set of guesses.

---

## Smart search with a wrong guess about boat speed

**Date:** 2026-10-03
**Command:** `python search_sim.py --assumed-speed-kn 10 30 --boat-speed-kn 0 40`
**I predict (my words):** "maybe around 17-20"
Meaning: at a 10 min old tip, smart search finds 17–20 out of 100 boats when it wrongly
thinks boats go 10–30 knots but they really go 0–40 (it found ~22 when its guess was right).
**Because:** Some boats will be slower or faster than it expects, so it searches in
slightly the wrong places.

**Actual:** At a 10 min old tip, smart search found 23.5% (95% CI 21.7–25.5%).
That's MORE than my guess of 17–20, and even more than the 22% from before.

But that's not a fair comparison, because the boats changed too. Now some boats are
slow or stopped, and those are easier for everyone to find. (The standard search went
from 0.4% to 20% just from that.) So I also ran the same 0–40 knot boats with smart
search told the RIGHT speeds (`--boat-speed-kn 0 40`, results in results_right_guess_0_40/):

| Tip age | Smart, right guess | Smart, wrong guess | Standard pattern |
|---|---|---|---|
| 10 min | 26.0% | 23.5% | 20.2% |
| 20 min | 15.4% | 12.2% | 12.6% |
| 30 min | 14.5% | 8.2% | 11.9% |

(Ring search used the same average speed, 20 kn, in both runs, so it didn't change.)

**What I learned:**
- My guess was wrong in the right direction: the wrong speed guess hurt smart search,
  but less than I thought at 10 min (it lost about 2–3 points, 26 down to 23.5, and
  those ranges overlap, so that drop might partly be luck).
- With older tips it hurts a lot. At 30 min the wrong guess cut smart search almost
  in half (14.5 down to 8.2), and the plain standard search (11.9) beat it.
- Why: smart search doesn't expect stopped or slow boats near the tip location, so it
  leaves that area. The standard search stays near the middle and picks them up.
- That matters for real life: a smuggler who stops and waits is exactly the boat the
  smart search would miss if I tell it the wrong speeds.
- So smart search is only as good as what I tell it about the boats. To trust it, I'd
  need real data on how smuggling boats actually move, or I should give it a wider
  guess (like 0–40 knots) on purpose.
- One weird thing: at 5 min, the wrong guess did BETTER than the right guess (44.8%
  vs 41.1%). Smart search is greedy (it only looks a couple of minutes ahead), so it
  isn't perfect. I should rerun with a different `--seed` to see how much the
  numbers jump around.

---

## Drone teams: 3 drones, shared map vs separate

**Date:** 2026-10-03
**Command:** `python search_sim.py --drones 3 --cue-ages 2 5 10 20 30 --out results_team3`
**I predict (my words):**
1. "I think separate will find relatively the same # of boats, maybe around 30. However, I
   predict that the shared map system will find a lot more, around 40-50."
2. "I think the shared map beats separate by 10-15 pts."
3. "I think situationally, 1 drone with a 2 minute tip performs better because it's a much
   smaller area to cover."
Meaning, at a 10 min old tip with 3 drones each: smart separate ~30%, smart shared map
40–50%, shared beats separate by 10–15 points. And 1 drone with a 2 min tip (81.8%)
beats 3 drones with a 10 min tip.
**Because:** Separate drones waste time covering the same water. A 2 min tip leaves a much
smaller area to search.

**Actual:** (2,000 boats per test, results in results_team3/)

| Tip age | Standard ×3 | Ring ×3 | Smart, separate ×3 | Smart, shared map ×3 |
|---|---|---|---|---|
| 2 min | 94.5% | 98.0% | 94.8% | 99.8% |
| 5 min | 33.1% | 53.2% | 54.4% | 87.0% |
| 10 min | 1.4% | 27.1% | 31.6% | 49.8% |
| 20 min | 0.0% | 14.0% | 19.8% | 25.4% |
| 30 min | 0.0% | 10.5% | 14.1% | 15.6% |

1. Separate found 31.6% at 10 min (I guessed ~30). Right.
   Shared map found 49.8% (I guessed 40–50). Right, at the top of my range.
2. Shared map beat separate by about 18 points (I guessed 10–15). The gap was bigger
   than I thought.
3. 1 drone with a 2 min tip found 81.8%. 3 shared-map drones with a 10 min tip found
   49.8%. The fresh tip wins, like I guessed.

**What I learned:**
- Sharing the map is a big deal. With one drone, smart search found 22 out of 100.
  Three drones searching SEPARATELY only got to 32, so two of the three drones were
  mostly wasted. Three drones with a SHARED map got to 50.
- Why: the snapshot picture shows it. Separate drones all chase the same "best" spot,
  so they bunch up on the same side. Shared-map drones split up and cover different parts.
- A fresh tip still beats more drones. 1 drone with a 2 min tip (82) beats 3 drones with
  a 10 min tip (50). Getting tips faster matters more than buying drones.
- But more drones buy you time: 3 shared-map drones with a 5 min tip (87) did better than
  1 drone with a 2 min tip (82). So 3 drones ≈ about 3 extra minutes of tip age.
- When the tip is really old (30 min), sharing barely helps anymore (15.6 vs 14.1, and the
  ranges overlap). The area is just too big for 3 drones no matter how they coordinate.
- This is my director's "multiple drones" idea, but in a version that works: drones sent
  to a tip, sharing one map, not patrolling the whole coast.
- Still to check: rerun with a different `--seed`, and test the shared map with the
  wrong speed guess (does a team fix the stopped-boat weak spot?).

---

## Combo strategies: ring-then-smart and guard-plus-smart (3 drones)

**Date:** 2026-10-03
**Commands:**
- `python search_sim.py --drones 3 --cue-ages 2 5 10 20 30 --strategies greedy_shared ring_then_smart guard_plus_smart --out results_combos`
- `python search_sim.py --drones 3 --assumed-speed-kn 10 30 --boat-speed-kn 0 40 --cue-ages 2 5 10 20 30 --strategies expanding_square greedy_shared guard_plus_smart --out results_combos_wrong_guess`

**I predict (my words):** "1, better, 57. 2. 55, 3. 45"
Meaning, at a 10 min old tip with 3 drones (plain shared map got 49.8%):
1. Ring-then-smart does better than the plain shared map: 57%.
2. Guard-plus-smart with the right speed guess: 55%.
3. With the wrong speed guess (thinks 10–30 kn, really 0–40): 45%. (I gave one number;
   recording it for both the plain shared map and guard-plus-smart.)

