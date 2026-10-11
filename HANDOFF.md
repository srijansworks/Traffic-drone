# Handoff: traffic-drone project (as of 2026-10-09)

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

- **2026-10-04:** showed my director (Esther Kim, TKS) the demo. Her direction: before
  building more, understand the users' specific problem (one feature they need could
  change the whole build). Get a warm intro through the Skydio engineer; make a list of
  people to talk to (names, emails, LinkedIn) and share it with her.
- **~Dec 10 2026:** TKS demo day.
- **Skydio engineer (monthly mentor) said (2026-10-09):** the F10 has ~120 miles of range
  but needs LTE to talk to the ground station. The X10 can launch from ships and boats
  with the handheld controller (likely how the Coast Guard uses it) and has a speaker
  attachment. The Coast Guard is buying 125 X10Ds (announced Sept 2025).
- **Outreach list (emails drafted, sent via ChatGPT):** Pacific Islands FFA Regional
  Fisheries Surveillance Centre; Ted Schmitt and Namrata Kolla (Skylight, Ai2); Nick Wise
  (OceanMind); Trygg Mat Tracking; David Kroodsma and Fernando Paolo (Global Fishing
  Watch); Lawrence Stone (Metron, SAROPS); Marijana Peti (Univ. of Zagreb); USCG R&D
  Center; Shield AI. Key questions: how old are tips, what do patrols do with them today,
  how do boats really move, would a "fly here next" plan help.
- **What's different about this project:** planning flight paths for drones to find a
  boat that's moving away from an old satellite tip. SAROPS (Coast Guard) only models
  drifting targets and plans back-and-forth boxes; Peti 2023 searched a small box with
  live radar; Skylight sends satellites, not drones (~2 h to a follow-up image).

## The code (`~/Documents/traffic-drone`, a git repo)

```bash
cd ~/Documents/traffic-drone && source venv/bin/activate
python tests/test_cuesearch.py          # all tests should pass
python search_sim.py                    # 1 drone: standard vs ring vs smart
python search_sim.py --drones 3         # teams: adds smart separate vs shared map
python search_sim.py --drones 3 --strategies greedy_shared ring_then_smart guard_plus_smart
python search_sim.py --assumed-speed-kn 10 30 --boat-speed-kn 0 40   # wrong speed guess
python search_sim.py --drone x10 --drones 3   # Skydio X10-like (15 m/s, 30 min); also f10
python failure_analysis.py --drones 3 --cue-age 2 --assumed-speed-kn 10 30 --boat-speed-kn 0 40
python real_tracks_sim.py --drone x10 --drones 3 --cue-ages 10 30 60   # REAL fishing boat paths
python real_tracks_sim.py --drone x10 --drones 3 --cue-ages 10 \
    --searcher-tracks data/AIS_2023_01_02.zip data/AIS_2023_01_03.zip # search learns from other days
python export_demo.py                   # regenerate demo/demo_data.js (~2 min)
python export_real_results.py           # regenerate demo/real_results.js (~3 min)
```

Real data: NOAA Marine Cadastre daily AIS files in `data/` (git-ignored, ~300 MB each):
`curl -o data/AIS_2023_01_01.zip https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip`
(and `_02`, `_03`). Jan 1 is the test day; Jan 2–3 are what the search learns from.

| File | What it is |
|---|---|
| `cuesearch/model.py` | Boat, drone, all assumptions |
| `cuesearch/planners.py` | Single-drone searches: `expanding_square`, `ring_search`, `probability_greedy` (v1 smart search) |
| `cuesearch/team.py` | N-drone versions plus `greedy_independent`, `greedy_shared`, `ring_then_smart`, `guard_plus_smart` |
| `cuesearch/evaluate.py` | Time to find, team time to find, Wilson 95% confidence interval |
| `search_sim.py` | Runs experiments and writes `results*/` (git-ignored) with a CSV and 3 plots |
| `export_demo.py`, `demo/` | Browser demo; the page only plays back what Python computed |
| `cuesearch/ais.py` | Real AIS tracks → test boats (`sample_boats`) and a learned guess cloud (`TrackPrior`) |
| `real_tracks_sim.py` | Runs the searches against real boat paths |
| `failure_analysis.py` | Find rate inside each true-speed group |
| `export_real_results.py` | Real-boat numbers for the demo page |
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

Since 2026-10-09 (after the random-number fix; experiments 9–17):

| Setup | Result |
|---|---|
| Seed check, 5 seeds: 1 drone smart / 3 drones shared map | 21.6–22.7 / 49.4–52.2 (stable, not luck) |
| 3 drones shared map: F10-like / X10-like (10–30 kn boats) | 51.8 / 11.6 (X10 can't catch fast boats) |
| 3 X10s vs slow 0–12 kn boats, 10 / 30 min tip: smart / spiral | 80.1 / 61.6 and 40.1 / 27.9 |
| 2 min tip, boats 0–40 kn: search guesses 10–30 / 10–40 / exactly right | 79.4 / 97.9 / 98.2 |
| REAL fishing paths, 3 X10s, 10 / 30 / 60 min: spiral | 80.1 / 54.8 / 35.1 |
| ...smart, straight-line guess | 88.1 / 55.8 / 39.9 |
| ...smart, learned from real tracks on other days | 92.2 / 61.5 / 47.1 |
| REAL fast boats (pleasure craft 15+ kn), 3 F10s, 10 min: smart / spiral | 54.9 / 49.5 |
| ...same, smart with learned guess | 65.3 |
| REAL fishing, Jan 4 test day, 3 X10s, 10 min: learned smart / spiral | 86.5 / 73.9 (lead holds) |
| Brown's FAB (classic optimal, random-search detection), 3 F10s, 10–30 kn, 10 min | 44.9 (smart 51.8) |
| Camera range 500 m instead of 1 km, 3 F10s shared map, 10 min | 28.1 (was 51.8) |

Newest takeaways: X10-class quadcopters fit slow boats (illegal fishing); fast boats
need an F10-class drone. All misses come from boats faster than the search assumed, so
guess speed high. On real paths the straight-line smart search barely beats the spiral,
because real boats slow, turn, and stop; learning from real tracks brings the lead back.
Knowing how boats really move matters as much as the search method.

Older takeaways: tips go stale in minutes. A shared map more than doubles one drone, while
3 separate drones mostly waste 2 of them. A fresh tip matters more than more drones. The
smart search is only as good as its assumption about boat speed (it skips stopped boats).
Full details are in `PREDICTIONS.md`.

## The demo

3D replay (same data, three.js): https://claude.ai/artifact/7Cba79VGgUurc2FRPPMuy8,
source `demo/search3d.html`. Republish it with demo_data.js copied from the 2D artifact.

Private web page (Claude artifact): https://claude.ai/artifact/6c4bnppRiXypLGCs5YLV83.
Source: `demo/index.html` + `demo/demo_data.js`. Switches: Team (1 drone / 3 drones /
3 drones: combos), Boat speed guess (right / wrong), tip age (2/5/10/20 min), play,
scrubber, results chart. To update it: run `export_demo.py`, then republish (from a Claude
session, publish to that URL; or open `demo/index.html` locally in a browser). It hasn't
been visually checked yet. If something looks broken, fix that first.

## Known issues and caveats

- AIS boats broadcast their position, so they're legal boats, not boats running from a
  patrol. No public data exists for that; ask practitioners how fleeing boats behave.
- Real-track results use one test day and ~200–250 vessels per row, so the true
  uncertainty is bigger than the confidence intervals show. Fast-boat learned guess not
  tested yet.
- X10/F10 speeds and search times are guesses. The F10 needs LTE, which fades offshore;
  plans are computed before launch, so a pre-loaded route might not need a live link (ask).
- Greedy only looks 2 minutes ahead, so it's not optimal. Sensor is cookie-cutter
  (perfect inside 1 km). No land, so boats and drones can cross coastlines.
- Brown's FAB (`cuesearch/fab.py`, `fab_sim.py`) needs ~300k particles or it overfits
  its own guesses. It assumes random-search detection, so it's not a ceiling for the
  model's perfect 1 km camera.
- Tello demo scripts and `waypoint_follower.py` from an earlier AI session are still
  uncommitted and unreviewed.
- Earlier AI sessions made mistakes that were caught: a fake "fair baseline", a "wrong
  guess" test that wasn't wrong, a seed fix that let the search see the answers, and a
  misleading failure-analysis script. Check AI-written experiments carefully.

## Next steps (roughly in order)

1. Hear back from outreach and the Skydio engineer; rerun with real numbers (camera range,
   drone speed, tip age).
2. Real fast boats with a learned guess; more test days; other regions.
3. Ask how reliable a drone camera is at spotting a small boat (perfect vs ~63% per pass
   changes whether FAB-style planning beats the smart search).
4. Review and commit the Tello demo code; physical demo on grass with the Tello
   (`~/Documents/dronefollow`).
5. Sensitivity: sensor range (`--sweep-radius`).
