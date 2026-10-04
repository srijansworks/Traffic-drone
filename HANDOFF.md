# Handoff: traffic-drone project (as of 2026-10-03)

Read this first, then `README.md` and `PREDICTIONS.md`.

## Who I am and how to work with me

- High-school student in the TKS program. I know Python, OpenCV, and DJI Tello control
  (PID, state machines). Math is around precalculus. I use AI tools heavily for code.
- **Explain things simply and keep answers short.** Plain words, define any jargon in one
  line. Long sectioned write-ups lose me.
- I care about doing this like a real experiment: hypothesis → baseline → experiment →
  numbers → failure analysis. Not "I built it and it worked."

## The rules for experiments (important)

1. Before any new experiment, ask me for my prediction (a number).
2. Write it in `PREDICTIONS.md` **in my own words**, then `git commit` it **before** running.
3. Run, then add **Actual** and **What I learned** (simple words) and commit again.
4. Never edit a prediction after the fact. If something was wrong, add a labeled
   correction note under it instead.
5. Don't show me results of an experiment I haven't predicted yet.

## Background: where the idea came from

- My TKS director proposed: traffickers use small boats in cloudy weather, "satellites
  can't see through clouds," so put drone patrols along the coast to catch every boat.
  She has connections (Interpol, military, DARPA). **Keeping momentum with her is my top
  priority.**
- Research showed the premise is wrong: **radar satellites (SAR: ICEYE, Capella, Umbra,
  Sentinel-1) see through clouds and at night**, and companies already flag "dark vessels"
  (boats detected by radar that don't broadcast AIS location). Blanket drone patrols
  don't add up: holding a 1,000 km line takes ~60 drones airborne nonstop.
- What does work: **cue and identify.** A satellite spots a suspicious boat and drones go
  check it. The open problem I picked: **the tip is stale by the time a drone arrives,**
  because the boat has moved. How do you find it?
- Also related: illegal fishing (IUU) is a big, well-documented use case for the same
  approach (paper: Desai & Shambaugh, PLOS ONE 2021, links illegal fishing to piracy).
  Don't mix up drug smuggling, human trafficking, migrant smuggling, piracy, and illegal
  fishing. They're different things.
- I paused a separate project ("Project A": acoustic detector for low-flying aircraft,
  checked against ADS-B) to focus on this.

## Timeline and people

- **Sunday 2026-10-04:** show my director the demo.
- **~Dec 10 2026:** TKS demo day.
- A Skydio engineer meets with me monthly. The director can introduce me to people. The
  question to ask a Coast Guard or maritime contact: *"How old are tips when you get them,
  how far can a drone camera spot a small boat, and how do these boats actually move?"*

## The code (`~/Documents/traffic-drone`, a git repo)

```bash
cd ~/Documents/traffic-drone && source venv/bin/activate
python tests/test_cuesearch.py          # all tests should pass
python search_sim.py                    # 1 drone: standard vs ring vs smart
python search_sim.py --drones 3         # teams: adds smart separate vs shared map
python search_sim.py --drones 3 --strategies greedy_shared ring_then_smart guard_plus_smart
python search_sim.py --assumed-speed-kn 10 30 --boat-speed-kn 0 40   # wrong speed guess
python export_demo.py                   # regenerate demo/demo_data.js (~2 min)
```

| File | What it is |
|---|---|
| `cuesearch/model.py` | Boat, drone, all assumptions |
| `cuesearch/planners.py` | Single-drone searches: `expanding_square`, `ring_search`, `probability_greedy` (v1 smart search) |
| `cuesearch/team.py` | N-drone versions plus `greedy_independent`, `greedy_shared`, `ring_then_smart`, `guard_plus_smart` |
| `cuesearch/evaluate.py` | Time to find, team time to find, Wilson 95% confidence interval |
| `search_sim.py` | Runs experiments and writes `results*/` (git-ignored) with a CSV and 3 plots |
| `export_demo.py`, `demo/` | Browser demo; the page only plays back what Python computed |
| `PREDICTIONS.md` | Every prediction and result, in order |

Git tag `v1-smart-search` = the single-drone smart search, saved as "solution v1".

**Model assumptions (all guesses, not field data):** drone 45 m/s (~100 mph), sees a boat
within 1 km (perfect inside, blind outside), 60 min of search time, boats 10–30 knots on
a straight line in a random direction, satellite position error ±100 m. Time 0 = the
satellite pass. Strategies plan one fixed path (no feedback until the boat is found), so
each path is scored against thousands of random boats.

**Strategies:**
- `expanding_square`: Coast Guard square spiral (baseline; meant for slow drifting targets).
- `ring_search`: fair baseline; flies out to the ring where the boat probably is and
  spirals with it.
- `probability_greedy` (smart): cloud of 2,000 guesses; every 30 s it picks the heading
  that sweeps the most guesses in the next 2 min and deletes guesses in water it checked.
- `greedy_shared`: team with one shared map; drones "claim" guesses so they split up.
  With 1 drone it flies exactly the v1 path (tested).

## Results so far (10-minute-old tip, % of boats found)

| Setup | Result |
|---|---|
| 1 drone: standard / ring / smart | 0.4 / 14.8 / 21.9 |
| 3 drones: standard / ring / smart separate / smart shared map | 1.4 / 27.1 / 31.6 / 49.8 |
| 3 drones: ring-then-smart / guard-plus-smart | 48.6 / 36.6 (neither beat the plain shared map) |
| 1 drone, 2-min tip | 81.8 (a fresh tip beats 3 drones with a 10-min tip) |
| Wrong speed guess, 3-drone shared map (right guess, same boats) | 43.4 (44.2); at 30 min: 15.2 (21.6) |

Key takeaways: tips go stale in minutes. A shared map more than doubles one drone, while
3 separate drones mostly waste 2 of them. A fresh tip matters more than more drones. The
smart search is only as good as its assumption about boat speed (it skips stopped boats).
Full details are in `PREDICTIONS.md`.

## The demo

Private web page (Claude artifact): https://claude.ai/artifact/6c4bnppRiXypLGCs5YLV83.
Source: `demo/index.html` + `demo/demo_data.js`. Switches: Team (1 drone / 3 drones /
3 drones: combos), Boat speed guess (right / wrong), tip age (2/5/10/20 min), play,
scrubber, results chart. To update it: run `export_demo.py`, then republish (from a Claude
session, publish to that URL; or open `demo/index.html` locally in a browser). It hasn't
been visually checked yet. If something looks broken, fix that first.

## Known issues and caveats

- **Random-number quirk:** boats are drawn in order of tip age, so `--cue-ages 10` alone
  gives different numbers than a run with several ages. Only compare runs that used the
  same command. Fix idea: seed each age separately (this shifts the logged numbers slightly).
- Each smart path comes from one random set of guesses. **Not yet checked with other
  `--seed` values.** Some results (e.g. the 5 min wrong-guess oddity) may be noise.
- Greedy only looks 2 minutes ahead, so it's not optimal. Sensor is cookie-cutter. Boats
  never turn or stop mid-search.
- An earlier AI session made two mistakes that were caught and fixed: a "fair baseline"
  that was secretly the expanding square again, and a "wrong guess" test that didn't
  actually give the searcher a wrong guess. Check AI-written experiments carefully.

## Next steps (roughly in order)

1. Visually check the demo; prepare for Sunday.
2. Seed check: rerun key results with `--seed 2 3 4` to see how much they move.
3. Failure analysis: which boats get missed (fast ones? stopped ones?) instead of guessing.
4. Fix the per-age random-number quirk.
5. Sensitivity: how results change with sensor range (`--sweep-radius`) and drone speed.
6. Physical demo: Tello runs the smart search toward a stand-in target on grass
   (the Tello drifts over water). Existing Tello code is in `~/Documents/dronefollow`.
7. Get real numbers from a practitioner (tip age, detection range, boat behavior).
