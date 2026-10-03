# traffic-drone

A satellite spots a suspicious boat. By the time a drone gets there, the boat
has moved. How old can the tip be before the drone can't find it -- and can a
smart search beat the standard Coast Guard pattern?

## Run it

```bash
cd ~/Documents/traffic-drone
source venv/bin/activate
python tests/test_cuesearch.py     # check everything works
python search_sim.py               # the experiment (~1 min), plots go in results/
python search_sim.py --help        # all the knobs: drone speed, sensor range, boat speed...
```

First time on a new computer: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Files

| File | What it does |
|---|---|
| `cuesearch/model.py` | The boat, the drone, and every assumption (with the reasoning) |
| `cuesearch/planners.py` | The three single-drone searches: expanding square, ring search, and smart search (v1) |
| `cuesearch/evaluate.py` | Did the drone find the boat, how fast, and the confidence interval |
| `cuesearch/team.py` | Drone teams: every strategy with N drones, plus smart search with and without a shared map |
| `search_sim.py` | Runs the experiment and makes the graphs (`--drones 3` for teams) |
| `export_demo.py`, `demo/` | The browser demo: export runs, then open `demo/index.html` |
| `PREDICTIONS.md` | Write your guess here before every run |

## Solutions so far

**v1: single-drone smart search** (git tag `v1-smart-search`, code in `cuesearch/planners.py`).
Keeps a cloud of guesses about where the boat is, flies toward the most guesses, and
crosses off water it has already checked. Results so far (10 min old tip, 1 drone):
smart 21.9%, ring search 14.8%, standard pattern 0.4%. Weak spot: if it's told the
wrong boat speeds, it skips stopped boats near the tip location (see PREDICTIONS.md).

**v2 (in progress): drone teams** (`cuesearch/team.py`). Same number of drones for
every strategy. The question: does a shared map (drones cross off water for each
other and go after different guesses) beat drones searching separately?
