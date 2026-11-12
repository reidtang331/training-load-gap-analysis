"""
Quantifying the gap.

Four questions, four functions:

  reliability()      Can we trust an athlete's number at all, given how many
                     sessions we have of him?
  shrink()           Pull thin-sample athletes toward their group so nobody tops
                     a list on luck.
  dimension_gaps()   How far is each athlete from the reference, per dimension,
                     in units a coach can act on?
  multivariate_gap() One number for the whole profile, with no chosen weights.

The last one is the answer to "how should we weight the factors". The honest
answer is that you should not choose weights by hand: Mahalanobis distance uses
the group's own covariance structure, so correlated dimensions are automatically
discounted rather than double-counted, and the result is invariant to the scale
of each dimension. It is then decomposed back into per-dimension contributions,
so the single number stays interpretable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .features import dimension_columns


# ------------------------------------------------------------- reliability --
def icc_and_reliability(df: pd.DataFrame, value_col: str,
                        athlete_col: str = "athlete") -> dict:
    """One-way random-effects ICC(1) on repeated athlete-session measurements.

    ICC is the share of total variance that is a stable difference between
    athletes rather than day-to-day fluctuation. It matters here because it sets
    a hard limit on how short a window can be and still support a ranking: at
    ICC = 0.2, a three-session average is mostly noise.

        ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)

    Spearman-Brown then gives the reliability of a k-session mean:

        R(k) = k * ICC / (1 + (k - 1) * ICC)
    """
    # ni = each athlete's session count, mi = each athlete's mean. N = total
    # observations, k = number of athletes. Need at least 4 athletes and more
    # observations than athletes, or the variance decomposition below is
    # meaningless -- bail out with NaN rather than report a fake number.
    d = df[[athlete_col, value_col]].dropna()
    g = d.groupby(athlete_col)[value_col]
    ni, mi = g.count(), g.mean()
    N, k = int(ni.sum()), int(len(ni))
    if k < 4 or N <= k:
        return {"icc": np.nan, "f_ratio": np.nan, "n_athletes": k, "n_obs": N,
                "sessions_for_080": None}

    # Classic one-way ANOVA variance decomposition:
    #   msb (mean square BETWEEN athletes) -- how much athletes differ from
    #   each other on average
    #   msw (mean square WITHIN an athlete) -- how much a single athlete's own
    #   sessions bounce around his own mean (the "noise" ICC is measuring
    #   against)
    #   n0  -- an adjustment for unequal session counts per athlete
    grand = d[value_col].mean()
    msb = float((ni * (mi - grand) ** 2).sum() / (k - 1))
    msw = float(((d[value_col] - d[athlete_col].map(mi)) ** 2).sum() / (N - k))
    n0 = float((N - (ni ** 2).sum() / N) / (k - 1))

    # ICC = share of total variance that's stable between-athlete difference,
    # not noise. Clamp to 0 if between-variance doesn't even exceed
    # within-variance (can happen with small/noisy samples).
    icc = 0.0 if msb <= msw else (msb - msw) / (msb + (n0 - 1) * msw)
    # How many sessions, averaged together, would it take for the
    # Spearman-Brown-predicted reliability to cross 0.80? Just tries k=1,2,3...
    # until the formula clears the bar.
    need = next((j for j in range(1, 200)
                 if icc > 0 and j * icc / (1 + (j - 1) * icc) >= 0.80), None)
    return {"icc": float(icc), "f_ratio": float(msb / msw) if msw > 0 else np.inf,
            "n_athletes": k, "n_obs": N, "sessions_for_080": need}


def reliability(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """ICC per dimension, pooled as the median across comparison groups."""
    rows = []
    for dim in dimension_columns(cfg):
        # Compute ICC separately within EACH position group (reliability can
        # genuinely differ by group, e.g. a noisier metric for skill players
        # than linemen), then take the median across groups as this
        # dimension's one headline number for the "Reliability" section.
        per_group = [icc_and_reliability(g, dim) for _, g in df.groupby("group")]
        per_group = [r for r in per_group if not np.isnan(r["icc"])]
        if not per_group:
            continue
        icc = float(np.median([r["icc"] for r in per_group]))
        need = next((k for k in range(1, 200)
                     if icc > 0 and k * icc / (1 + (k - 1) * icc) >= 0.80), None)
        rows.append({"dimension": dim.replace("dim_", ""),
                     "icc": round(icc, 3),
                     "f_ratio_median": round(float(np.median([r["f_ratio"] for r in per_group])), 2),
                     "sessions_for_reliability_080": need,
                     # What reliability looks like today, at a representative
                     # 3-session sample -- gives the report something concrete
                     # to say besides the abstract ICC number.
                     "reliability_at_3_sessions": round(3 * icc / (1 + 2 * icc), 3) if icc > 0 else 0.0})
    return pd.DataFrame(rows)


def shrink(profiles: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of each athlete's mean toward his group mean.

    An athlete seen nine times is estimated less precisely than one seen thirty
    times, so his observed mean should not be taken at face value. The shrinkage
    weight is the Spearman-Brown reliability of his own session count:

        estimate = w * observed + (1 - w) * group mean,   w = R(n_i)

    This is what stops a small sample from topping a list on luck, and it is the
    principled version of "don't rank on a two-day window".
    """
    out = profiles.copy()
    # Need each dimension's ICC (computed above) to know how much to shrink.
    rel = reliability(df, cfg).set_index("dimension")["icc"].to_dict()

    for dim in dimension_columns(cfg):
        name = dim.replace("dim_", "")
        icc = rel.get(name, np.nan)
        col, n = f"{dim}_mean", out["n_sessions"]
        # No usable ICC for this dimension -- can't compute a sensible
        # shrinkage weight, so pass the raw mean through unshrunk rather than
        # guess at a weight.
        if col not in out.columns or not np.isfinite(icc) or icc <= 0:
            out[f"{dim}_shrunk"] = out.get(col, np.nan)
            out[f"{dim}_weight"] = np.nan
            continue
        # Spearman-Brown reliability of THIS athlete's own session count --
        # this is also his shrinkage weight: how much of his own observed
        # mean to trust vs. pull him toward the group. More sessions, or a
        # more reliable dimension, means w closer to 1 (trust him more).
        w = n * icc / (1 + (n - 1) * icc)
        group_mean = out.groupby("group")[col].transform("mean")
        # The actual shrinkage: a weighted blend of "his own average" and
        # "his group's average," weighted by how much we trust his own number.
        out[f"{dim}_shrunk"] = w * out[col] + (1 - w) * group_mean
        out[f"{dim}_weight"] = w.round(3)
    return out


# -------------------------------------------------------------- references --
def _reference_rows(profiles: pd.DataFrame, cfg: Config,
                    depth: pd.DataFrame | None,
                    reference_player: str | list[str] | None = None,
                    mode: str | None = None,
                    value_suffix: str = "shrunk") -> pd.DataFrame:
    """Who each group is measured against.

    With a depth chart the reference is the listed starter(s) for that group.
    A Catapult group can cover more than one starting slot on the depth chart
    (e.g. three starting linebacker spots all carry the Catapult label "LB"),
    in which case the reference is the AVERAGE of every listed starter in that
    group -- there is no single "the starter" for that group by construction.
    `reference_members` records exactly who was averaged (or, for a
    single-starter group, the one name), so this is never left implicit.
    Without a depth chart, the busiest athlete stands in -- a proxy that must
    be labelled as such wherever it is reported.

    `reference_player` overrides this for one or more groups: pass a single
    name, or a list of names (one per group you want to hand-pick), so a
    coach can ask "is anyone in this room ready to be THIS guy" for any
    specific player. Every group not named keeps whatever the normal mode
    would have picked.

    `mode` overrides cfg["gap"]["reference"] for this call only, so a caller
    can compute a second, alternate reference basis (e.g. busiest-athlete)
    in the same run without mutating config or re-reading data -- see
    run_pipeline.py's "vs. depth chart" / "vs. busiest player" toggle.

    `value_suffix` picks which per-athlete estimate to compare: "shrunk"
    (default -- gap.shrink's empirical-Bayes estimate, pulled toward the
    group mean by how reliable the dimension is) or "mean" (the athlete's
    plain, un-adjusted session average). Same reference logic either way --
    this only changes which number is being compared, so a caller can hold
    the reference fixed and isolate the effect of shrinkage alone.
    """
    mode = mode or cfg["gap"]["reference"]
    dims = [f"{d}_{value_suffix}" for d in dimension_columns(cfg)]

    # --- Three ways to build the per-group reference row, chosen by `mode`:
    if depth is not None and mode == "depth_rank_1":
        # Mode 1 (the default, when a depth chart is supplied): average
        # together every athlete marked depth_rank==1 in each group. This is
        # one row even when a group covers multiple real starting slots (see
        # the big comment in the docstring above) -- `reference_members`
        # records who actually went into that average.
        starters = depth[depth["depth_rank"] == 1][["group", "athlete"]]
        matched = profiles.merge(starters, on=["group", "athlete"])
        ref = matched.groupby("group")[dims].mean()
        ref["reference_kind"] = "depth chart starter"
        ref["reference_members"] = matched.groupby("group")["athlete"].apply(list)
    elif mode == "group_median":
        # Mode 2: the typical player in the group, no depth chart needed.
        ref = profiles.groupby("group")[dims].median()
        ref["reference_kind"] = "group median"
        ref["reference_members"] = [[] for _ in range(len(ref))]
    else:
        # Mode 3 (fallback / "busiest player"): no depth chart at all, so
        # stand in with whichever athlete in each group has the highest value
        # on the FIRST configured dimension -- a rough proxy for "the guy who
        # plays the most."
        key = f"dim_{list(cfg.enabled_dimensions)[0]}_{value_suffix}"
        idx = profiles.groupby("group")[key].idxmax()
        ref = profiles.loc[idx].set_index("group")[dims]
        ref["reference_kind"] = "busiest athlete (proxy for starter)"
        ref["reference_members"] = [[a] for a in profiles.loc[idx, "athlete"]]

    # --- Manual overrides: for any athlete named in `reference_player`,
    # replace his WHOLE group's reference row with his own numbers alone,
    # regardless of which mode picked the default above.
    names = [reference_player] if isinstance(reference_player, str) else (reference_player or [])
    for name in names:
        row = profiles[profiles["athlete"] == name]
        if row.empty:
            raise ValueError(f"reference_player {name!r} not found "
                             f"in this window's roster")
        group = row.iloc[0]["group"]
        ref.loc[group, dims] = row.iloc[0][dims].to_numpy(dtype=float)
        ref.loc[group, "reference_kind"] = f"named reference: {name}"
        ref.at[group, "reference_members"] = [name]

    return ref


def slot_gaps(profiles: pd.DataFrame, cfg: Config, depth: pd.DataFrame | None) -> pd.DataFrame:
    """Per-athlete, per-dimension gap to the SPECIFIC starter at his exact
    depth-chart slot -- not the group average (_reference_rows above collapses
    a whole Catapult group, e.g. every "LB", to one number even though the
    depth chart lists three separate starting linebacker spots), and not the
    busiest-player proxy. This is the literal "who is behind whom" reading of
    the depth chart, and needs the depth chart's `slot` column to mean
    anything -- a Catapult "group" alone is too coarse to say which starter a
    given backup actually plays behind.

    Returns empty if there is no depth chart or it has no `slot` column.
    A slot with more than one depth_rank==1 row is a data error (two people
    both marked as the same slot's starter) and is dropped rather than
    guessed at, same principle as _reference_rows never silently picking one
    of several tied "busiest" athletes.
    """
    if depth is None or "slot" not in depth.columns:
        return pd.DataFrame()
    thresholds = cfg["gap"]["effect_thresholds"]

    # Attach each athlete's slot + depth_rank from the depth chart onto his
    # profile row. `how="inner"` means athletes not on the depth chart at all
    # (e.g. deep bench/walk-ons) simply don't appear -- nothing to compare
    # them to.
    merged = profiles.merge(depth[["athlete", "group", "slot", "depth_rank"]],
                            on=["group", "athlete"], how="inner")

    # One starter row per (group, slot), indexed for fast lookup below. A
    # slot with more than one depth_rank==1 athlete is a data entry error
    # (two people both marked "the starter" for the same job) -- drop it
    # rather than silently pick one.
    starters = merged[merged["depth_rank"] == 1].set_index(["group", "slot"])
    starters = starters[~starters.index.duplicated(keep=False)]

    rows = []
    for _, r in merged.iterrows():
        key = (r["group"], r["slot"])
        # No valid (non-duplicate) starter for this athlete's slot -- skip him,
        # nothing to compare against.
        if key not in starters.index:
            continue
        starter = starters.loc[key]
        # Same gap_sd = value - reference calculation as dimension_gaps()
        # above, just against THIS SPECIFIC slot's starter instead of a
        # whole-group reference.
        for dim in dimension_columns(cfg):
            col = f"{dim}_shrunk"
            value = r.get(col, np.nan)
            target = starter.get(col, np.nan)
            gap = value - target
            rows.append({
                "group": r["group"], "slot": r["slot"], "athlete": r["athlete"],
                "depth_rank": int(r["depth_rank"]), "n_sessions": r["n_sessions"],
                "dimension": dim.replace("dim_", ""), "value": value,
                "reference": target, "gap_sd": gap,
                "gap_label": label_effect(gap, thresholds),
                "starter": starter["athlete"],
            })
    return pd.DataFrame(rows)


def label_effect(d: float, thresholds: dict) -> str:
    a = abs(d)
    if not np.isfinite(a):
        return "unknown"
    if a < thresholds["trivial"]:
        return "trivial"
    if a < thresholds["small"]:
        return "small"
    if a < thresholds["moderate"]:
        return "moderate"
    if a < thresholds["large"]:
        return "large"
    return "very large"


def dimension_gaps(profiles: pd.DataFrame, cfg: Config,
                   depth: pd.DataFrame | None = None,
                   reference_player: str | list[str] | None = None,
                   mode: str | None = None,
                   value_suffix: str = "shrunk") -> pd.DataFrame:
    """Per-athlete, per-dimension distance from the reference.

    Reported three ways because three audiences need three things:
      gap_sd   standardised difference, for the analyst -- comparable across
               dimensions and labelled with Hopkins' thresholds
      gap_pct  percentage of the reference, for the coach
      gap_raw  the difference itself, for the record

    `value_suffix`: see _reference_rows -- "shrunk" (default) or "mean" (the
    same gap, computed with no empirical-Bayes shrinkage applied at all).
    """
    ref = _reference_rows(profiles, cfg, depth, reference_player, mode, value_suffix)
    thresholds = cfg["gap"]["effect_thresholds"]
    rows = []

    for _, r in profiles.iterrows():
        for dim in dimension_columns(cfg):
            col = f"{dim}_{value_suffix}"
            value = r.get(col, np.nan)
            target = ref.loc[r["group"], col] if r["group"] in ref.index else np.nan
            # Dimensions are already z-scored within group, so one unit is one
            # within-group SD and the difference is itself a standardised effect.
            gap = value - target
            rows.append({
                "group": r["group"], "athlete": r["athlete"],
                "n_sessions": r["n_sessions"],
                "dimension": dim.replace("dim_", ""),
                "value": value, "reference": target,
                "gap_sd": gap,
                "gap_label": label_effect(gap, thresholds),
                "reference_kind": ref.loc[r["group"], "reference_kind"] if r["group"] in ref.index else None,
                "reference_members": ref.loc[r["group"], "reference_members"] if r["group"] in ref.index else None,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------- multivariate gap --
def multivariate_gap(profiles: pd.DataFrame, cfg: Config,
                     depth: pd.DataFrame | None = None,
                     reference_player: str | list[str] | None = None,
                     mode: str | None = None,
                     value_suffix: str = "shrunk") -> pd.DataFrame:
    """One gap number per athlete, weights chosen by the data's covariance.

    Mahalanobis distance from the athlete's profile x to the reference profile m,
    using the within-group covariance S:

        D = sqrt( (x - m)' S^-1 (x - m) )

    Why this rather than a weighted sum: the dimensions are correlated, and any
    hand-chosen weight vector therefore does not mean what it appears to mean --
    two correlated dimensions given 0.2 each behave like one dimension with 0.4.
    S^-1 removes exactly that double-counting. It also makes the result invariant
    to how each dimension was scaled, so the answer does not depend on arbitrary
    units.

    Because a single distance is not actionable on its own, the per-dimension
    contributions are returned alongside it: contribution_j = (x-m)_j * [S^-1 (x-m)]_j,
    which sums to D^2 and says which dimension is driving the gap.

    Small groups make S unstable, so it is ridge-regularised toward a diagonal
    matrix; with fewer athletes than dimensions the shrinkage is heavy and the
    result tends toward a plain sum of squared standardised gaps.
    """
    dims = [f"{d}_{value_suffix}" for d in dimension_columns(cfg)]
    names = [d.replace("dim_", "") for d in dimension_columns(cfg)]
    ref = _reference_rows(profiles, cfg, depth, reference_player, mode, value_suffix)
    out = []

    # Work one position group at a time -- the whole point is that
    # "correlated" means correlated WITHIN a group, so the covariance matrix
    # below has to be computed per group, not across the whole roster.
    for group, g in profiles.groupby("group"):
        X = g[dims].to_numpy(dtype=float)
        ok = ~np.isnan(X).any(axis=1)          # rows with no missing dimension
        if ok.sum() < 3 or group not in ref.index:
            continue                            # too few complete rows to get a covariance at all
        m = ref.loc[group, dims].to_numpy(dtype=float)   # this group's reference point

        # --- Build this group's 5x5 covariance matrix S (how the five
        # dimensions move together within this group), then regularise it so
        # it's always safely invertible:
        S = np.cov(X[ok], rowvar=False)
        S = np.atleast_2d(S)
        # Ridge toward the diagonal: keeps S invertible when the group is small.
        # Small groups (fewer athletes than 3x the dimension count) get a
        # heavier ridge, since their raw covariance estimate is noisier.
        lam = 0.20 if ok.sum() >= 3 * len(dims) else 0.50
        S = (1 - lam) * S + lam * np.diag(np.clip(np.diag(S), 1e-9, None))
        S += np.eye(len(dims)) * 1e-9          # tiny nudge, avoids a singular matrix
        Sinv = np.linalg.pinv(S)               # the inverse covariance -- this is what does the "don't double-count correlated dimensions" work

        # --- Now compute one distance per athlete in this group:
        for (_, row), keep in zip(g.iterrows(), ok):
            if not keep:
                continue
            d = row[dims].to_numpy(dtype=float) - m   # this athlete's 5-dimension vector minus the reference
            q = Sinv @ d
            d2 = float(d @ q)                          # squared Mahalanobis distance

            # Distance is symmetric: an athlete who does MORE than the reference
            # is also "far" from it, which is not a readiness problem. The
            # deficit distance keeps only the dimensions where he is BELOW the
            # reference, and is the number to use for "is he ready".
            deficit = np.minimum(d, 0.0)
            q_def = Sinv @ deficit
            d2_def = float(deficit @ q_def)
            # Per-dimension pieces that sum to d2_def -- "which dimension is
            # driving this athlete's DEFICIT". Built from the same clamped
            # vector as deficit_distance itself, not the two-sided d/q above,
            # so a dimension where he's actually ahead of the reference can
            # never be named as the driver of a deficit he doesn't have.
            contrib = deficit * q_def

            rec = {"group": group, "athlete": row["athlete"],
                   "n_sessions": row["n_sessions"],
                   "mahalanobis": float(np.sqrt(max(d2, 0.0))),
                   "deficit_distance": float(np.sqrt(max(d2_def, 0.0))),
                   "n_dimensions_below": int((d < 0).sum()),
                   "reference_kind": ref.loc[group, "reference_kind"],
                   "reference_members": ref.loc[group, "reference_members"]}
            # Express each dimension's contribution as a share of the total
            # deficit (so they read as e.g. "62% of his deficit is collision"),
            # guarding against a zero total -- nothing to be behind on at all
            # (the reference athlete himself, or anyone at/above reference on
            # every dimension).
            total = contrib.sum()
            for nm, c in zip(names, contrib):
                rec[f"share_{nm}"] = float(c / total) if total > 0 else np.nan
            out.append(rec)

    res = pd.DataFrame(out)
    if res.empty:
        return res

    # --- Flag list: rank each athlete's deficit_distance within his OWN
    # group (0-1 percentile), then flag anyone above the configured
    # threshold (e.g. top 20%) -- "flagged" is what drives the Flag List
    # section of the report.
    q = cfg["gap"]["flag_percentile"]
    # Flagging uses the deficit distance, not the symmetric one.
    res["group_percentile"] = res.groupby("group")["deficit_distance"].rank(pct=True)
    res["flagged"] = res["group_percentile"] >= q

    # --- "driven_by": whichever dimension's share_<dim> column is largest
    # for that row, i.e. the single biggest contributor to his distance.
    # Which dimension is doing the most work in this athlete's gap. The
    # reference athlete himself has zero gap and therefore no driver.
    share_cols = [c for c in res.columns if c.startswith("share_")]
    has_share = res[share_cols].notna().any(axis=1)
    res["driven_by"] = pd.NA
    if has_share.any():
        res.loc[has_share, "driven_by"] = (
            res.loc[has_share, share_cols].idxmax(axis=1).str.replace("share_", "", regex=False))
    return res.sort_values(["group", "mahalanobis"], ascending=[True, False])


def bootstrap_gap_ci(profiles: pd.DataFrame, cfg: Config, dim: str,
                     n_boot: int | None = None) -> pd.DataFrame:
    """Resample athletes within group to put an interval on the group-level gap.

    Groups here hold 6 to 16 athletes. A ratio computed on six people carries a
    lot of uncertainty, and reporting it bare invites a decision the data cannot
    support.
    """
    rng = np.random.default_rng(cfg["gap"]["random_seed"])
    n_boot = n_boot or cfg["gap"]["bootstrap_iterations"]
    col = f"dim_{dim}_shrunk"
    rows = []
    for group, g in profiles.groupby("group"):
        v = g[col].dropna().to_numpy()
        if len(v) < 4:
            continue
        # Classic bootstrap: resample this group's athletes WITH replacement
        # n_boot times, recompute "median minus the top performer" each time,
        # and take the 2.5th/97.5th percentile of that distribution as a 95%
        # confidence interval on the real statistic.
        stats = [np.median(s) - s.max() for s in
                 (rng.choice(v, len(v), replace=True) for _ in range(n_boot))]
        lo, hi = np.percentile(stats, [2.5, 97.5])
        rows.append({"group": group, "dimension": dim, "n_athletes": len(v),
                     "median_minus_top_sd": float(np.median(v) - v.max()),
                     "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows)
