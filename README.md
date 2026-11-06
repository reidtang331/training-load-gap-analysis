# Workload Gap

Quantifying the gap between an athlete's current training load and his
position-group peers, to support injury-prevention decisions.

This repository holds the **statistical and analytical methodology** from a
larger applied project built for a Division I college football program's
sports-science staff, using their wearable GPS/accelerometer data (Catapult).
It contains no athlete data, no roster information, and no team-identifying
detail — those are confidential under the data-use agreement covering that
project. What's here is the general-purpose statistical engine: the same
code, with the real export's specifics (team name, season, roster) stripped
out and replaced by a synthetic/generic example configuration.

## The problem

A position group's wearable-load metrics (total distance, high-speed
running, collisions, change-of-direction, etc.) are not on a common scale
across positions, are individually noisy session to session, and are
correlated with each other in ways that make a hand-picked weighted score
misleading. The question this project answers: **how far behind his
position-group peers (or a specific reference player, e.g. the starter) is
this athlete, on each dimension of load, and overall — accounting for how
reliable that number actually is?**

## Methods

- **Within-group z-scoring** (`src/features.py`) — every raw metric is
  standardised inside its own position group before anything else happens, so
  a 320 lb lineman and a 175 lb corner are never compared on a shared scale.
  Metrics are grouped into a small number of conceptually-defined dimensions
  (volume, mechanical load, change of direction, high speed, collision) with
  equal weight inside each — deliberately not data-driven (PCA) weights, which
  over-weight whichever indicators happen to be most correlated.

- **ICC / Spearman-Brown reliability** (`src/gap.py`) — a one-way
  random-effects intraclass correlation estimates how much of a dimension's
  variance is a stable between-athlete difference versus session-to-session
  noise. This sets a hard floor on how few sessions can support a ranking at
  all, and drives:

- **Empirical-Bayes shrinkage** — each athlete's own mean is pulled toward
  his group's mean, weighted by the Spearman-Brown reliability of his own
  session count. An athlete seen three times gets pulled harder toward the
  group than one seen forty times, which is what stops a thin sample from
  topping a list on luck.

- **Mahalanobis distance** — rather than choose weights for a multi-dimension
  gap, the group's own within-group covariance structure does the weighting:
  correlated dimensions are discounted rather than double-counted, and the
  result doesn't depend on the arbitrary units each dimension happens to be
  in. The distance is then decomposed back into per-dimension contributions,
  so a single number stays interpretable ("62% of this gap is collision
  load"). A **deficit distance** variant zeroes out any dimension where the
  athlete is *above* the reference, since doing more than the reference isn't
  a readiness concern.

- **Bootstrap confidence intervals** on small-group statistics (position
  groups here commonly hold well under 20 athletes).

- **Quality control** (`src/clean.py`) — rule-based detection of GPS dropout
  (indoor sessions with no satellite fix), physically-impossible velocity
  spikes (unit/satellite artefacts), non-participation rows (unit on, athlete
  not actually working), and athletes logged under more than one position
  label.

See the docstrings in `src/gap.py` and `src/features.py` for the full
derivation of each method, including why each design choice was made over
the obvious alternative.

## Running it

```
pip install -r requirements.txt
pytest tests/
```

Every test runs against synthetic data with a known, derivable answer (e.g.
"this ICC must come back within 0.06 of 0.80 given this between/within
variance ratio") — a failure means a formula is wrong, not that some
external dataset changed. No real data of any kind is used anywhere in this
repository.

## What's not here

The full applied project also includes a data-intake layer for a specific
GPS vendor's export format, a depth-chart reader, an HTML report generator,
a Streamlit app with a local offline LLM assistant for querying results, and
a packaged desktop app — none of which are methodology, so none of it is
included here.
