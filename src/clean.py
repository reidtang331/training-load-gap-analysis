"""
Quality control.

Every rule here exists because of something found in real wearable-GPS exports
in practice, not a hypothetical. Each returns a flag column rather than silently
dropping rows, so a run report can say exactly how much data each rule touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def canonical_position(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One position per athlete.

    A noticeable minority of athletes in a real roster export carry more than
    one position label across their sessions. Most are a single stray session;
    the genuine ones are nearly all secondary-position churn. Every comparison
    in this project is within-group, so the group an athlete belongs to has to
    be decided once and applied everywhere.
    """
    df = df.copy()

    # --- Step 1: merge near-duplicate labels (e.g. "DB" -> "CB") before we
    # even look at who plays what, so aliasing doesn't itself create fake
    # "multi-label" athletes.
    df["position"] = df["position_raw"].replace(cfg["groups"]["label_aliases"])

    # --- Step 2: one athlete, one position, for the whole dataset. Whatever
    # label shows up most often across his sessions wins; every row for that
    # athlete gets overwritten with it, even sessions that were logged under
    # a different (minority) label.
    mode = df.groupby("athlete")["position"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
    df["position"] = df["athlete"].map(mode)

    # Count how many athletes actually had conflicting labels before the fix
    # above -- purely for the QC report, doesn't affect the data itself.
    n_multi = int((df.groupby("athlete")["position_raw"].nunique() > 1).sum())
    df.attrs.setdefault("qc", {})["athletes_with_multiple_labels"] = n_multi

    # --- Step 3: decide each athlete's comparison "group" -- the bucket every
    # gap calculation downstream groups by. If the staff have supplied real
    # substitution groups (who can actually sub for whom, which may not match
    # the Catapult position label), use those; otherwise group is just the
    # position computed above.
    groups = cfg["groups"].get("substitution_groups") or {}
    if groups and not cfg["groups"]["use_raw_position"]:
        lookup = {a: g for g, members in groups.items() for a in members}
        df["group"] = df["athlete"].map(lookup).fillna(df["position"])
    else:
        df["group"] = df["position"]

    # --- Step 4: also tag a coarse archetype (big / big_skill / skill) per
    # athlete -- a fallback grouping for when his specific position group is
    # too small to analyse on its own (see analysis_frame's size check below).
    arche = {p: a for a, ps in cfg["groups"]["archetypes"].items() for p in ps}
    df["archetype"] = df["position"].map(arche)
    return df


def flag_participation(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """A unit switched on while the athlete did essentially nothing is not a
    light day -- it is a non-observation, and averaging it in drags the athlete
    down for a reason that has nothing to do with his training."""
    df = df.copy()
    # True/False per row: did this athlete actually do enough work this
    # session to count, or was the unit just on? Everything downstream that
    # should ignore non-participation rows checks this column rather than
    # dropping rows outright, so the row is still visible in the QC count.
    df["participated"] = df["total_player_load"] >= cfg["quality"]["min_player_load"]
    return df


def flag_indoor(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Detect sessions with no GPS fix.

    Indoors the unit still records accelerometer data, so Player Load and every
    IMA count remain valid, but distance and velocity are structural zeros. A
    season of outdoor practice data typically has a handful of sessions where
    effectively the whole squad reads max velocity exactly zero. Averaged in
    unflagged they suppress every athlete's speed exposure by an amount that
    depends on how many indoor days happen to fall in the window -- which is a
    bias, not noise.
    """
    df = df.copy()
    thresh = cfg["quality"]["indoor_zero_velocity_share"]

    # Only look at rows where the athlete actually participated -- a session
    # full of non-participants would otherwise look falsely "indoor" (lots of
    # zero-velocity rows for a reason that has nothing to do with GPS).
    played = df[df.get("participated", True)]

    # For each date, what share of participating athletes recorded EXACTLY
    # zero max velocity? A normal practice has a few genuinely slow players;
    # an indoor day with no GPS fix has almost everyone at zero.
    share = played.groupby("date")["max_velocity"].apply(lambda s: (s == 0).mean())
    indoor_dates = set(share[share > thresh].index)

    # Flag every row on an indoor date, regardless of that specific athlete's
    # own velocity that day -- the whole date is unreliable for GPS metrics.
    df["indoor"] = df["date"].isin(indoor_dates)
    df.attrs.setdefault("qc", {})["indoor_sessions"] = sorted(str(pd.Timestamp(d).date()) for d in indoor_dates)
    return df


def clean_velocity(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """GPS spikes. Real exports occasionally contain max velocities well above
    anything physically possible (multiple hundreds of mph) -- a unit/satellite
    artefact, not a real reading."""
    df = df.copy()
    ceiling = cfg["quality"]["max_velocity_ceiling_mph"]

    # Anything above the physically-possible ceiling (or exactly zero, which
    # is a structural "no reading" rather than a real value) gets blanked to
    # NaN instead of clipped -- clipping would silently invent a fake ceiling
    # value; NaN honestly says "we don't know."
    n_bad = int((df["max_velocity"] > ceiling).sum())
    df["max_velocity"] = df["max_velocity"].where(
        (df["max_velocity"] > 0) & (df["max_velocity"] <= ceiling))

    # Indoors there is no velocity at all, so blank the GPS fields rather than
    # letting a structural zero read as "this athlete did no running".
    gps = [c for c in ["max_velocity", "high_speed_distance", "high_speed_efforts", "total_distance"]
           if c in df.columns]
    df.loc[df["indoor"], gps] = np.nan
    df.attrs.setdefault("qc", {})["impossible_velocity_rows"] = n_bad
    return df


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Full QC chain. Order matters: participation before indoor detection,
    indoor before velocity blanking."""
    qc: dict = {"rows_in": len(df)}

    # Run each QC step in sequence, folding its reported stats into one
    # combined `qc` dict as we go -- this is what ends up in run_report.json.
    for step in (canonical_position, flag_participation, flag_indoor, clean_velocity):
        df = step(df, cfg)
        qc.update(df.attrs.get("qc", {}))

    # --- Hard date exclusions: known equipment/collection failures (e.g. a
    # unit left on all day), named explicitly in config rather than detected
    # statistically. These rows are dropped outright, not just flagged --
    # unlike the steps above, there's no legitimate reading to salvage here.
    excluded = sorted(set(cfg["quality"].get("excluded_dates") or []))
    if excluded:
        excluded_ts = pd.to_datetime(excluded)
        n_excluded = int(df["date"].isin(excluded_ts).sum())
        df = df[~df["date"].isin(excluded_ts)].copy()
        qc["excluded_dates"] = excluded
        qc["excluded_date_rows"] = n_excluded

    # Final tallies for the run report -- computed after every QC step and
    # exclusion above, so these numbers describe the data as it leaves clean().
    qc["non_participation_rows"] = int((~df["participated"]).sum())
    qc["athletes"] = int(df["athlete"].nunique())
    qc["sessions"] = int(df["date"].nunique())
    qc["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    df.attrs["qc"] = qc
    return df


def analysis_frame(df: pd.DataFrame, cfg: Config,
                   start=None, end=None) -> pd.DataFrame:
    """Rows eligible for the gap analysis: participating, in an analysable group,
    and belonging to an athlete with enough sessions to estimate anything."""
    # Start from only the rows where the athlete actually participated --
    # everything below further narrows this, never widens it.
    out = df[df["participated"]].copy()

    # --- Optional date window (the --start/--end CLI flags) -- restricts to
    # a specific slice of the season, e.g. "just fall camp onward".
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]

    # Drop specialist/non-comparable groups (e.g. "Coach", "SPT") that were
    # never meant to be analysed against a position group.
    out = out[~out["group"].isin(cfg["groups"]["exclude_positions"])]

    # Snapshot of this window BEFORE the session/group-size filters below --
    # purely diagnostic, so a thin or empty result can explain itself (e.g.
    # "this window only has 5 practice dates, but 10 per athlete is the
    # floor") instead of just silently vanishing. A narrow custom date range
    # can easily contain fewer calendar practice dates than the floor
    # requires, which is a completely different situation from an athlete
    # who just skipped most of a long window, so both get reported.
    window_sessions_available = int(out["date"].nunique())
    n = out.groupby("athlete")["date"].nunique()
    max_sessions_any_athlete = int(n.max()) if len(n) else 0

    # --- Minimum-sessions filter: an athlete with only 1-2 sessions in this
    # window can't support any of the statistics downstream (reliability,
    # shrinkage) -- drop him from the window entirely rather than let a
    # near-empty profile distort his group.
    min_required = cfg["quality"]["min_sessions_per_athlete"]
    keep = n[n >= min_required].index
    athletes_before = set(out["athlete"].unique())
    out = out[out["athlete"].isin(keep)]
    athletes_dropped_too_few_sessions = len(athletes_before) - len(set(keep))

    # --- Minimum-group-size filter: a group with too few athletes left (after
    # the filter above) can't support a meaningful within-group comparison --
    # drop the whole group for this window; it reappears once enough players
    # and sessions accumulate.
    sizes = out.groupby("group")["athlete"].nunique()
    big_enough = sizes[sizes >= cfg["groups"]["min_players_per_group"]].index
    out = out[out["group"].isin(big_enough)]

    # Summary of exactly what window/roster this run ended up analysing --
    # read by run_pipeline.py for its console output, by app.py to explain
    # a thin result in the UI (not just bury it in the pipeline log), and
    # by report.py for the "Window ... to ..." line at the top of the HTML
    # report.
    out.attrs["window"] = {
        "start": str(out["date"].min().date()) if len(out) else None,
        "end": str(out["date"].max().date()) if len(out) else None,
        "sessions": int(out["date"].nunique()),
        "athletes": int(out["athlete"].nunique()),
        "groups": sorted(out["group"].unique().tolist()),
        "groups_dropped_too_small": sorted(set(sizes.index) - set(big_enough)),
        "window_sessions_available": window_sessions_available,
        "max_sessions_any_athlete": max_sessions_any_athlete,
        "min_sessions_required": min_required,
        "athletes_dropped_too_few_sessions": athletes_dropped_too_few_sessions,
    }
    return out
