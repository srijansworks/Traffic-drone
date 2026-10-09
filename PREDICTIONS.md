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

**Actual:** (3 drones, 2,000 boats per test)

Right speed guess (results_combos/):

| Tip age | Shared map | Ring-then-smart | Guard-plus-smart |
|---|---|---|---|
| 5 min | 87.0% | 81.7% | 76.4% |
| 10 min | 49.8% | 48.6% | 36.6% |
| 30 min | 15.6% | 17.2% | 11.1% |

Wrong speed guess, thinks 10–30 kn, really 0–40 (results_combos_wrong_guess/), with the
same boats and the RIGHT guess for comparison (results_combos_right_guess_0_40/):

| Tip age | Shared, right guess | Shared, wrong guess | Guard, wrong guess | Standard ×3 |
|---|---|---|---|---|
| 2 min | 98.2% | 78.6% | 78.0% | 79.3% |
| 10 min | 44.2% | 43.4% | 39.1% | 23.8% |
| 30 min | 21.6% | 15.2% | 18.4% | 13.7% |

1. Ring-then-smart: 48.6% (I guessed 57, better). It basically TIED the plain shared map
   (49.8%, the ranges overlap). Wrong.
2. Guard-plus-smart, right guess: 36.6% (I guessed 55). It did WORSE. Wrong.
3. Wrong guess: shared map 43.4%, guard 39.1% (I guessed 45). Close for the shared map;
   the guard came in lower.

**What I learned:**
- Combining ideas doesn't automatically make things better. Both combos lost or tied.
- Ring-then-smart tied because the shared-map smart search already heads outward to where
  the boats are. Forcing a straight dash first didn't add anything, and with a fresh tip
  (5 min) it actually cost a few points (81.7 vs 87.0).
- The guard lost when the speed guess was right because there was nothing for it to find:
  every boat goes at least 10 knots, so after 10 min no boat is still near the tip
  location. That's one drone out of three wasted, so the team was really just 2 searchers.
- The guard only paid off in one case: wrong speed guess AND an old tip. At 30 min it
  beat the plain shared map (18.4 vs 15.2, the ranges barely don't overlap). That's the
  stopped-boat weak spot I found before, and the guard partly fixes it, but only late.
- Big surprise: with 3 drones, the wrong speed guess barely hurt at 10 min (44.2 vs 43.4).
  But with a fresh 2 min tip it hurt a LOT (98.2 down to 78.6), and the guard didn't help
  there. So those misses probably aren't the stopped boats. My guess: they're the fast
  boats (30–40 kn) that got farther than the searchers expected. I should check which
  boats get missed instead of guessing.
- Best plan so far is still the plain shared map. If I'm unsure how boats move, giving
  the search a wider speed guess on purpose is probably better than adding a guard.

---

---

### Note (2026-10-09): random-number fix

Not an experiment, just a fix. Each tip age now gets its own random boats, so running
one age alone gives the same boats as running many ages. (An uncommitted edit tried
this but at a 1-minute tip it gave the searcher the exact same boats it was guessing,
which would be cheating. Caught and fixed, with a test.) Results logged ABOVE this note
used the old random numbers, so reruns of the same commands will come out slightly
different. Compare new results only to runs made after this fix.

---

## Experiments 9–11: seed check, Skydio X10 vs F10, which boats get missed (2026-10-09)

Run after the random-number fix (commit c2535ad). All at the default model unless
noted (boats 10–30 kn, sees 1 km, 2000 boats per test unless noted).

**Experiment 9: seed check.** Rerun at a 10 min tip with seeds 1–5. Is the smart search
really good or did it get lucky with one set of guesses?
- 1 drone smart search (was 21.9%), 3 drones shared map (was 49.8%).

**Experiment 10: X10 vs F10.** 3 drones, shared map, 10 min tip.
- F10-like: 45 m/s, 60 min search (same as the old default).
- X10-like: 15 m/s, 30 min search (launched from a boat; numbers are guesses).

**Experiment 11: which boats get missed.** 3 drones, shared map, 2 min tip, searchers
think boats go 10–30 kn but really 0–40 kn (last time: 78.6% overall). Find rate inside
each speed group, 4000 boats.

### My predictions (my words)

1. "I think the smart search has scope to be better than current standards. the highest
   single drone mission can maybe yield 25%, and 3 drone mission can yield 50%."
   (Highest of 5 seeds: 1 drone 25%, 3 drones 50%. No lowest given.)
2. "45%" (three X10s)
3. "I think it will find 80% of nearly stopped boats and 50% of the fastest boats."
   (0–5 kn: 80%, 35–40 kn: 50%)

### Actual (2026-10-09)

**9. Seed check, 10 min tip, seeds 1–5** (the seed changes both the boats and the search's guesses):

| Seed | 1 drone smart | 3 drones shared map |
|---|---|---|
| 1 | 22.0% | 51.8% |
| 2 | 21.7% | 51.6% |
| 3 | 22.7% | 50.8% |
| 4 | 21.6% | 52.2% |
| 5 | 22.4% | 49.4% |

Range: 1 drone 21.6–22.7%, 3 drones 49.4–52.2%. I guessed the highest would be 25% and
50%. One drone stayed under my guess; 3 drones went slightly over.

**10. X10 vs F10, 3 drones, 10 min tip:**

| Drone | Shared map | Standard spiral ×3 |
|---|---|---|
| F10-like (45 m/s, 60 min) | 51.8% | 1.7% |
| X10-like (15 m/s, 30 min) | 11.6% | 0.0% |

I guessed 45% for the X10s. Actual 11.6%. Way off.

**11. Which boats get missed** (3 drones shared map, 2 min tip, think 10–30 kn, really
0–40 kn, 4000 boats; overall 79.4%):

| True boat speed | Found |
|---|---|
| 0–25 kn (every group) | 100% |
| 25–30 kn | 98.8% |
| 30–35 kn | 31.7% |
| 35–40 kn | 1.8% |

I guessed 80% of nearly stopped boats (actual 100%) and 50% of the fastest (actual 1.8%).
I got the direction right (fast boats are the problem) but not how big.

### What I learned

- The smart search isn't luck. Changing the random seed only moves it about 1 point
  (1 drone) or 3 points (3 drones). The results are solid, inside this model.
- The X10 is too slow for this job against fast boats. 15 m/s is about 29 knots, so a
  30-knot boat is as fast as the drone. It can't catch up to boats that already have a
  10-minute head start. The F10's speed is what makes this work.
- That splits my use cases: slow boats (fishing boats, many under ~12 kn) might suit a
  cheap quadcopter from a patrol boat. Go-fast drug boats need a fast fixed-wing like
  the F10. I should test the X10 against slow boats next.
- With a fresh 2 min tip, stopped boats aren't a problem at all: they're still near the
  tip spot where the drones start. ALL the misses are boats faster than the searchers
  assumed (30+ kn). So if I'm unsure about boat speed, guess too high, not too low.

---

## Experiments 12–13: X10 vs slow boats, guessing boat speed high (2026-10-09)

**Experiment 12: can the X10 catch slow boats?** 3 X10-like drones (15 m/s, 30 min),
shared map. Boats are slow, like fishing boats: 0–12 kn, and the search knows that.
Tips 10 and 30 min old.

**Experiment 13: does guessing high fix the misses?** Same as experiment 11 (3 drones
F10-like, shared map, 2 min tip, boats really 0–40 kn) but the search assumes 10–40 kn
instead of 10–30. Last time: 79.4%.

### My predictions (my words)

12. "I think the X10 can find 30% in a 10 minute tip, and 10% for a 30 min tip."
13. "I think it will still result around 80%"

### Actual (2026-10-09)

**12. Three X10s vs slow boats (0–12 kn):**

| Tip age | Shared map | Standard spiral ×3 |
|---|---|---|
| 10 min | 80.1% | 61.6% |
| 30 min | 40.1% | 27.9% |

I guessed 30% and 10%. Both way too low.

**13. Guessing boat speed high (assume 10–40, really 0–40), 2 min tip:** 97.9% (2000 boats;
the 4000-boat breakdown run got 98.3%). Last time with the low guess: 79.4%. Earlier, the
same boats with the exactly right guess got 98.2%. Breakdown: every speed group 100%
except 35–40 kn at 86.1%.

I guessed it would stay around 80%. Wrong.

### What I learned

- Cheap quadcopters DO work for slow boats. Three X10s find 80% of slow boats from a
  10-minute tip, vs 11.6% of 10–30 kn boats. So the X10 fits illegal fishing (slow
  boats); the F10 fits fast drug boats.
- With slow boats, the smart search still wins, but by less (80 vs 62). Slow boats don't
  get far, so even the plain spiral reaches a lot of them. The smart search matters most
  when boats are fast.
- Guessing boat speed too high costs almost nothing (97.9% vs 98.2% with the exact right
  guess). Guessing too low costs ~19 points (79.4%). So the rule is: when unsure, tell
  the search the boats might be faster than you think.

---

## Experiment 14: real fishing-boat paths (2026-10-09)

Three X10-like drones (15 m/s, 30 min), 10 min tip, search assumes 0–12 kn. Boats are
REAL US fishing vessels (AIS type 30) from NOAA data for Jan 1 2023: a random vessel,
a random moment it was moving at 2+ kn, then where it really went (real_tracks_sim.py).
On straight-line 0–12 kn boats (experiment 12): smart 80.1%, spiral 61.6%.

### My prediction (my words)

"I think both of them will significantly reduce in number. Smart will avg. 50-60% and
spiral will avg. 30-40%"

### Actual (2026-10-09)

| | Straight-line 0–12 kn (exp 12) | Real fishing paths |
|---|---|---|
| Smart, shared map | 80.1% | 88.1% |
| Standard spiral ×3 | 61.6% | 80.1% |

(2000 test boats drawn from 251 different vessels, so the real uncertainty is bigger than
the ±1.5% the confidence interval says: many test boats are the same vessel at another
moment. Only one day of data.)

I guessed both would drop a lot (smart 50–60, spiral 30–40). Both went UP. Wrong direction.

### What I learned

- Real fishing boats are EASIER to find than my straight-line boats, because they don't
  go anywhere fast: they wander, turn, and loiter near where they were seen. Over an
  hour most end up only a few knots' worth from the start.
- That helps the plain spiral the most (61.6 → 80.1), since a spiral is built for targets
  that stay near the start. The smart search's lead shrank from 18 points to 8.
- So for illegal fishing with a fresh-ish tip, even simple searches do OK. The smart
  search matters most for boats that are actually running: fast boats, or a boat that
  knows it was spotted. These AIS boats weren't hiding from anyone.
- Next: test older tips (30–60 min) on real paths, and other days of data, and look at
  real fast boats (not just fishing).

---

## Experiments 15–16: older tips on real fishing paths, real fast boats (2026-10-09)

**Experiment 15.** Same as 14 (three X10-like drones, real fishing paths, search assumes
0–12 kn) with 30 and 60 min tips. Straight-line slow boats at 30 min: smart 40.1%,
spiral 27.9%.

**Experiment 16.** Real fast boats: AIS pleasure craft (type 37) doing 15+ kn at the
tip, as a stand-in for go-fast boats. Three F10-like drones (45 m/s, 60 min), 10 min
tip, search assumes 10–40 kn. Straight-line 10–30 kn boats: smart 51.8%, spiral 1.7%.

### My predictions (my words)

15. "smart gets 64% spiral gets 50%, for 60 minutes, smart gets 40% and spiral gets 15%"
    (30 min tip: smart 64, spiral 50. 60 min tip: smart 40, spiral 15.)
16. "smart will get 40% and spiral gets 0%"

### Actual (2026-10-09)

**15. Real fishing paths, three X10s:**

| Tip age | Smart, shared map | Standard spiral ×3 |
|---|---|---|
| 10 min (exp 14) | 88.1% | 80.1% |
| 30 min | 55.8% | 54.8% |
| 60 min | 39.9% | 35.1% |

I guessed 30 min: smart 64, spiral 50. 60 min: smart 40, spiral 15. Smart at 60 min was
spot on; the spiral did much better than I thought.

**16. Real fast boats (pleasure craft at 15+ kn), three F10s, 10 min tip:**
smart 54.9%, spiral 49.5%. (Straight-line 10–30 kn boats: 51.8% vs 1.7%.)

I guessed smart 40, spiral 0. The spiral was way off.

### What I learned

- On real paths, my smart search barely beats the plain spiral: +8, +1, +5 points on
  fishing boats and +5 on fast boats. On straight-line boats it won by 12 to 50 points.
- The reason: real boats, even fast ones, slow down, turn, stop, and stay closer to where
  they were seen than a straight-line boat would. My smart search assumes every boat
  keeps going straight at 10–40 kn, so it flies out to a ring where real boats mostly
  aren't. The spiral searches near the start, which is where real boats end up.
- So my model's biggest weak spot is the straight-line boat. The smart search is only as
  good as its guess about how boats move.
- Fix to test next: let the smart search guess using REAL past tracks (from a different
  day, so it isn't cheating) instead of straight lines.
- Caveat: these AIS boats aren't running from anyone. A boat that knows it was spotted
  might really go straight and fast. I need a real patrol person to tell me which is true.
