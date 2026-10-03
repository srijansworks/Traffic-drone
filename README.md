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
| `cuesearch/planners.py` | The two search strategies: expanding square (baseline) and probability greedy |
| `cuesearch/evaluate.py` | Did the drone find the boat, how fast, and the confidence interval |
| `search_sim.py` | Runs the experiment and makes the graphs |
| `PREDICTIONS.md` | Write your guess here before every run |
