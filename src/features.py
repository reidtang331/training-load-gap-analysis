"""
Turning raw accelerometer/GPS columns into load dimensions.

The central design decision, and the reason this file is short:

  We do NOT collapse everything into one weighted "workload score".

Three reasons, all from the sports-science methods literature:

 1. Weights derived from the data (PCA) systematically over-weight whichever
    indicators are most correlated with each other, because a correlated cluster
    dominates the first component. In wearable-load datasets like this, one
    general component typically explains most of the variance with every
    loading positive -- that component is usually just overall volume, and
    building weights from it would rediscover a metric already known to be
    uninformative on its own.

 2. Nominal weights are not importance. Once indicators are correlated, an
    indicator's influence on the final score depends on the covariance structure,
    not on the number you assigned it. Hand-picked weights therefore do not mean
    what they appear to mean.

 3. Athlete profiles are individually heterogeneous. Collapsing to one number
    hides the compensation pattern -- the athlete who is low on one dimension and
    high on another -- which is exactly the information a position coach needs.

So: a small number of conceptually-defined dimensions, equal weights inside each,
kept separate. Where a single number is genuinely needed, gap.py uses Mahalanobis
distance, which handles the correlation properly and requires no chosen weights
at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def athlete_session_matrix(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per athlete-session, columns = the raw metrics the dimensions need."""
    # Only keep metrics that are both configured (listed under some dimension
    # in config.yaml) AND actually present in this export -- `missing` is
    # purely informational, surfaced in the run report so a coach can see if
    # an export is thinner than expected.
    wanted = [m for m in cfg.dimension_metrics if m in df.columns]
    missing = [m for m in cfg.dimension_metrics if m not in df.columns]
    keep = ["athlete", "group", "position", "archetype", "date", "indoor"] + wanted
    out = df[keep].copy()
    out.attrs["missing_metrics"] = missing
    return out


def zscore_within_group(df: pd.DataFrame, metrics: list[str],
                        group_col: str = "group") -> pd.DataFrame:
    """Standardise each metric inside its comparison group.

    Within-group is not a detail, it is the whole point. Player Load is
    accelerometer-derived and carries no body-mass normalisation, so a 320 lb
    tackle and a 175 lb corner are not on the same scale and a raw comparison
    between them is meaningless. Standardising inside the group makes the scale
    question disappear, because the comparison never crosses a group.
    """
    out = df.copy()
    for m in metrics:
        # `.transform` broadcasts each group's mean/SD back onto every row of
        # that group (not a group-summary table), so this stays one row per
        # athlete-session, just with a new z_<metric> column next to it.
        g = out.groupby(group_col)[m]
        mu, sd = g.transform("mean"), g.transform("std")
        # Guard against a group with zero variance (every athlete recorded
        # the exact same value) -- divide-by-zero would otherwise produce inf.
        out[f"z_{m}"] = np.where(sd > 0, (out[m] - mu) / sd, np.nan)
    return out


def build_dimensions(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Average the z-scored member metrics of each dimension, equal weight.

    Averaging z-scores rather than raw values means a dimension is not dominated
    by whichever member happens to have the largest units, and a member that is
    missing for a given athlete-session simply drops out of that mean instead of
    voiding the dimension.
    """
    # First, z-score every raw metric within its group (see above) -- this is
    # the step that makes a 320 lb tackle and a 175 lb corner comparable.
    metrics = [m for m in cfg.dimension_metrics if m in df.columns]
    out = zscore_within_group(df, metrics)

    # Then average each dimension's member z-scores together, equal weight,
    # per config.yaml's dimension -> metrics mapping. `skipna=True` means a
    # metric missing for one athlete-session just drops out of that mean
    # instead of voiding the whole dimension for that row.
    for name, spec in cfg.enabled_dimensions.items():
        members = [f"z_{m}" for m in spec["metrics"] if f"z_{m}" in out.columns]
        if not members:
            out[f"dim_{name}"] = np.nan
            continue
        out[f"dim_{name}"] = out[members].mean(axis=1, skipna=True)
        # How many of this dimension's metrics actually contributed to this
        # row's average -- useful for spotting a thin/unreliable dimension.
        out[f"dim_{name}_n_metrics"] = out[members].notna().sum(axis=1)
    return out


def zscore_within_group_by_player_mean(df: pd.DataFrame, metrics: list[str],
                                       group_col: str = "group",
                                       athlete_col: str = "athlete") -> pd.DataFrame:
    """Standardise each metric using the spread of PLAYER averages within the
    group, instead of the spread of individual sessions (see
    zscore_within_group above).

    Most practice-to-practice variation is day-to-day noise, not a real
    difference between players (ICCs commonly 0.1-0.3 in this kind of data --
    see docs/methods.md's reliability section), so the session-pooled SD that
    zscore_within_group divides by is substantially smaller than the spread
    of players' own averages. A "1 SD" gap in that unit therefore reads
    roughly half the size it would in player-to-player terms. This produces
    the player-to-player unit instead, reported ALONGSIDE the session unit
    (see build_dimensions_player_unit below), not in place of it.
    """
    out = df.copy()
    for m in metrics:
        # One row per athlete's own mean first, so the SD below is the
        # spread of PLAYER averages -- a session-weighted SD-of-means would
        # double-count whichever athlete happened to have the most sessions.
        athlete_mean = out.groupby([group_col, athlete_col])[m].transform("mean")
        player_means = (out.assign(_player_mean=athlete_mean)
                           .drop_duplicates([group_col, athlete_col]))
        spread = player_means.groupby(group_col)["_player_mean"].std()
        centre = player_means.groupby(group_col)["_player_mean"].mean()
        sd = out[group_col].map(spread)
        mu = out[group_col].map(centre)
        out[f"zplayer_{m}"] = np.where(sd > 0, (out[m] - mu) / sd, np.nan)
    return out


def build_dimensions_player_unit(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Same as build_dimensions, but in the player-to-player SD unit (see
    zscore_within_group_by_player_mean) rather than the session-pooled one.

    Column names match build_dimensions exactly (dim_<name>, dim_<name>_
    n_metrics), so this frame runs through athlete_profiles / gap.shrink /
    gap.dimension_gaps / gap.multivariate_gap completely unchanged -- same
    pipeline, differently-scaled input, not a second implementation of any
    of that logic.
    """
    metrics = [m for m in cfg.dimension_metrics if m in df.columns]
    out = zscore_within_group_by_player_mean(df, metrics)

    for name, spec in cfg.enabled_dimensions.items():
        members = [f"zplayer_{m}" for m in spec["metrics"] if f"zplayer_{m}" in out.columns]
        if not members:
            out[f"dim_{name}"] = np.nan
            continue
        out[f"dim_{name}"] = out[members].mean(axis=1, skipna=True)
        out[f"dim_{name}_n_metrics"] = out[members].notna().sum(axis=1)
    return out


def dimension_columns(cfg: Config) -> list[str]:
    """The ["dim_volume", "dim_mechanical", ...] column names, in config order."""
    return [f"dim_{k}" for k in cfg.enabled_dimensions]


def trend_summary(dims_df: pd.DataFrame, cfg: Config, step_sd: float = 0.5) -> pd.DataFrame:
    """Per athlete, per dimension: is load trending up, down, or flat across
    the window? Splits each athlete's own sessions at the midpoint -- first
    half vs second half, by date -- and compares the two halves' average, in
    the same within-group standard deviations as every other gap number in
    this pipeline. A before/after split, not a fitted regression -- for the
    session counts this pipeline actually sees (often well under 20), a
    regression's extra precision isn't supported by the data; this stays
    legible and is still exactly what "is this rising" needs.

    Built for the chat assistant: existing per-athlete tables are already-
    collapsed one-row averages with no trajectory in them. `athlete_session_
    dimensions` has the trajectory but is too large to hand an LLM directly
    (and asking it to eyeball a trend from raw rows is the same reliability
    problem this pipeline exists to avoid) -- this table is the small,
    pre-computed middle ground.
    """
    dims = dimension_columns(cfg)
    rows = []
    for (group, athlete), g in dims_df.groupby(["group", "athlete"]):
        g = g.sort_values("date")
        n = len(g)
        if n < 4:  # too few sessions for a before/after split to mean anything
            continue
        mid = n // 2
        first, second = g.iloc[:mid], g.iloc[mid:]
        for d in dims:
            if d not in g.columns:
                continue
            first_mean, second_mean = first[d].mean(), second[d].mean()
            if pd.isna(first_mean) or pd.isna(second_mean):
                continue
            delta = second_mean - first_mean
            trend = "rising" if delta >= step_sd else "falling" if delta <= -step_sd else "stable"
            rows.append({
                "group": group, "athlete": athlete, "dimension": d.replace("dim_", ""),
                "first_half_mean": round(first_mean, 3), "second_half_mean": round(second_mean, 3),
                "delta": round(delta, 3), "trend": trend, "n_sessions": n,
            })
    return pd.DataFrame(rows)


def athlete_profiles(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Collapse athlete-sessions to one profile per athlete.

    The mean over sessions, plus the count and the 90th percentile. The 90th
    percentile is the athlete's demonstrated ceiling -- what he has actually
    proven he can absorb -- which is the quantity the readiness question is
    really about. The mean answers "what is his normal week"; the ceiling answers
    "has he ever been near the demand he would inherit".
    """
    dims = dimension_columns(cfg)
    # For every dimension, compute three stats per athlete at once: mean
    # (his normal week), std (his own session-to-session spread), and p90
    # (his demonstrated ceiling). `date: nunique` in the same groupby gets his
    # session count for free, in the same pass.
    agg = {d: ["mean", "std", lambda s: s.quantile(0.90)] for d in dims}
    prof = df.groupby(["group", "athlete"]).agg({**agg, "date": "nunique"})

    # pandas names a multi-stat aggregation's columns as tuples, e.g.
    # ("dim_volume", "mean") -- flatten those into "dim_volume_mean", and
    # rename the generic lambda name into the readable "_p90" suffix.
    prof.columns = ["_".join(c) if isinstance(c, tuple) else c for c in prof.columns]
    prof = prof.rename(columns={"date_nunique": "n_sessions"})
    prof.columns = [c.replace("_<lambda_0>", "_p90") for c in prof.columns]
    return prof.reset_index()
