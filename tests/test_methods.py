"""
Tests against data with a KNOWN answer -- a failure here means a formula is
wrong, not that some external dataset changed. No real data of any kind is
used anywhere in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clean, features, gap          # noqa: E402
from src.config import Config                   # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return Config.load()


# ------------------------------------------------------------------ helpers
def synthetic(n_athletes=12, n_sessions=25, between_sd=1.0, within_sd=1.0, seed=0):
    """Athletes differing by a known amount, sessions differing by a known amount."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0, between_sd, n_athletes)
    rows = []
    for a in range(n_athletes):
        for s in range(n_sessions):
            rows.append({
                "athlete": f"A{a:02d}", "group": "WR",
                "date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=s),
                "value": true[a] + rng.normal(0, within_sd),
            })
    return pd.DataFrame(rows), true


# --------------------------------------------------------------------- unit
@pytest.mark.parametrize("between,within,expected", [
    (1.0, 1.0, 0.50),   # equal variance -> ICC 0.5
    (2.0, 1.0, 0.80),   # 4x between-variance -> ICC 0.8
    (0.0, 1.0, 0.00),   # no real differences -> ICC 0
])
def test_icc_recovers_known_value(between, within, expected):
    df, _ = synthetic(n_athletes=40, n_sessions=40, between_sd=between,
                      within_sd=within, seed=1)
    got = gap.icc_and_reliability(df, "value")["icc"]
    assert got == pytest.approx(expected, abs=0.06)


def test_spearman_brown_is_monotonic_and_bounded():
    df, _ = synthetic(n_athletes=30, n_sessions=30, between_sd=0.6, seed=2)
    r = gap.icc_and_reliability(df, "value")
    icc, need = r["icc"], r["sessions_for_080"]
    assert 0 < icc < 1 and need is not None
    rel = [k * icc / (1 + (k - 1) * icc) for k in range(1, 60)]
    assert all(b >= a for a, b in zip(rel, rel[1:]))          # never decreases
    assert rel[need - 1] >= 0.80 and rel[max(need - 2, 0)] < 0.80


def test_zscore_is_computed_within_group_not_across():
    """The core guarantee: a group with a huge mean must not drag another group.

    This is what makes comparing a 320 lb lineman to a 175 lb corner impossible
    by construction rather than by discipline.
    """
    df = pd.DataFrame({
        "group": ["OL"] * 5 + ["WR"] * 5,
        "m": [1000, 1100, 1200, 1300, 1400, 10, 11, 12, 13, 14],
    })
    out = features.zscore_within_group(df, ["m"])
    for g in ("OL", "WR"):
        sub = out.loc[out["group"] == g, "z_m"]
        assert sub.mean() == pytest.approx(0, abs=1e-9)
        assert sub.std() == pytest.approx(1, abs=1e-9)


def test_shrinkage_pulls_thin_samples_harder():
    """An athlete seen 3 times must move toward the group more than one seen 40."""
    df, _ = synthetic(n_athletes=20, n_sessions=30, between_sd=0.8, seed=3)
    df = df.rename(columns={"value": "dim_volume"})
    df["dim_volume_n_metrics"] = 1
    prof = (df.groupby(["group", "athlete"])
              .agg(dim_volume_mean=("dim_volume", "mean"), n_sessions=("date", "nunique"))
              .reset_index())
    prof.loc[0, "n_sessions"] = 3
    prof.loc[1, "n_sessions"] = 40

    cfg = Config.load()
    cfg.raw["dimensions"] = {"volume": {"label": "V", "question": "", "metrics": ["x"]}}
    out = gap.shrink(prof, df, cfg)
    assert out.loc[0, "dim_volume_weight"] < out.loc[1, "dim_volume_weight"]

    gm = prof["dim_volume_mean"].mean()
    moved_thin = abs(out.loc[0, "dim_volume_shrunk"] - out.loc[0, "dim_volume_mean"])
    moved_thick = abs(out.loc[1, "dim_volume_shrunk"] - out.loc[1, "dim_volume_mean"])
    assert moved_thin > moved_thick
    # and it moves toward the group mean, never past it
    assert min(out.loc[0, "dim_volume_mean"], gm) - 1e-9 <= out.loc[0, "dim_volume_shrunk"] \
        <= max(out.loc[0, "dim_volume_mean"], gm) + 1e-9


def test_effect_labels_match_hopkins_thresholds(cfg):
    t = cfg["gap"]["effect_thresholds"]
    assert gap.label_effect(0.1, t) == "trivial"
    assert gap.label_effect(0.4, t) == "small"
    assert gap.label_effect(0.9, t) == "moderate"
    assert gap.label_effect(1.5, t) == "large"
    assert gap.label_effect(3.0, t) == "very large"
    assert gap.label_effect(-1.5, t) == "large"        # sign-independent
    assert gap.label_effect(np.nan, t) == "unknown"


def test_indoor_detection_finds_planted_sessions(cfg):
    """A day where the squad records zero velocity is indoors, and must be caught
    without hard-coding the date."""
    rows = []
    for d in range(10):
        for a in range(30):
            indoor = d in (3, 7)
            rows.append({"athlete": f"A{a}", "position_raw": "WR",
                         "date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=d),
                         "total_player_load": 300.0,
                         "max_velocity": 0.0 if indoor else 17.0})
    df = clean.flag_participation(pd.DataFrame(rows), cfg)
    out = clean.flag_indoor(df, cfg)
    assert set(out.attrs["qc"]["indoor_sessions"]) == {"2024-08-04", "2024-08-08"}
    assert out.loc[out["date"] == pd.Timestamp("2024-08-04"), "indoor"].all()
    assert not out.loc[out["date"] == pd.Timestamp("2024-08-01"), "indoor"].any()


def test_velocity_spikes_removed_and_indoor_gps_blanked(cfg):
    df = pd.DataFrame({
        "athlete": ["A", "B", "C"], "position_raw": ["WR"] * 3,
        "date": [pd.Timestamp("2024-08-01")] * 3,
        "total_player_load": [300.0, 300.0, 300.0],
        "max_velocity": [18.0, 439.0, 0.0],
        "high_speed_distance": [100.0, 100.0, 100.0],
        "indoor": [False, False, True],
    })
    out = clean.clean_velocity(df, cfg)
    assert out.loc[0, "max_velocity"] == 18.0
    assert pd.isna(out.loc[1, "max_velocity"])            # spike removed
    assert pd.isna(out.loc[2, "high_speed_distance"])      # indoor GPS blanked
    assert out.attrs["qc"]["impossible_velocity_rows"] == 1


def test_mahalanobis_ignores_dimension_rescaling():
    """The property that makes it preferable to a weighted sum: the answer must
    not depend on the units each dimension happens to be in."""
    rng = np.random.default_rng(5)
    n = 40
    base = rng.normal(size=(n, 3))
    prof = pd.DataFrame({
        "group": ["WR"] * n, "athlete": [f"A{i}" for i in range(n)],
        "n_sessions": 20,
        "dim_volume_shrunk": base[:, 0],
        "dim_mechanical_shrunk": base[:, 1],
        "dim_collision_shrunk": base[:, 2],
    })
    cfg = Config.load()
    cfg.raw["dimensions"] = {k: {"label": k, "question": "", "metrics": ["x"]}
                             for k in ("volume", "mechanical", "collision")}
    cfg.raw["gap"]["reference"] = "group_median"

    a = gap.multivariate_gap(prof, cfg).set_index("athlete")["mahalanobis"]
    scaled = prof.copy()
    scaled["dim_mechanical_shrunk"] *= 1000          # change the units
    b = gap.multivariate_gap(scaled, cfg).set_index("athlete")["mahalanobis"]
    assert np.allclose(a.sort_index(), b.sort_index(), rtol=1e-6)


def test_deficit_distance_ignores_doing_more_than_the_reference():
    """An athlete above the reference on every dimension is not a readiness risk,
    so his deficit distance must be zero even though his plain distance is not."""
    rng = np.random.default_rng(6)
    n = 30
    prof = pd.DataFrame({
        "group": ["WR"] * n, "athlete": [f"A{i}" for i in range(n)],
        "n_sessions": 20,
        "dim_volume_shrunk": rng.normal(size=n),
        "dim_mechanical_shrunk": rng.normal(size=n),
    })
    prof.loc[0, ["dim_volume_shrunk", "dim_mechanical_shrunk"]] = [9.0, 9.0]   # way above
    prof.loc[1, ["dim_volume_shrunk", "dim_mechanical_shrunk"]] = [-9.0, -9.0] # way below

    cfg = Config.load()
    cfg.raw["dimensions"] = {k: {"label": k, "question": "", "metrics": ["x"]}
                             for k in ("volume", "mechanical")}
    cfg.raw["gap"]["reference"] = "group_median"
    res = gap.multivariate_gap(prof, cfg).set_index("athlete")

    assert res.loc["A0", "mahalanobis"] > 3            # far from the reference
    assert res.loc["A0", "deficit_distance"] == pytest.approx(0.0, abs=1e-9)
    assert res.loc["A1", "deficit_distance"] > 3       # genuinely behind
    assert not res.loc["A0", "flagged"]
    assert res.loc["A1", "flagged"]


def test_canonical_position_resolves_multi_label_athletes(cfg):
    """A real roster export often has some athletes logged under more than one
    position label across their sessions. Group membership has to be decided
    once, or every within-group number is computed on a shifting set."""
    df = pd.DataFrame({
        "athlete": ["A"] * 10 + ["B"] * 10,
        "position_raw": ["CB"] * 9 + ["SAF"] + ["DB"] * 10,
        "date": list(pd.date_range("2024-08-01", periods=10)) * 2,
    })
    out = clean.canonical_position(df, cfg)
    assert out.loc[out["athlete"] == "A", "position"].nunique() == 1
    assert out.loc[out["athlete"] == "A", "position"].iat[0] == "CB"
    assert out.loc[out["athlete"] == "B", "position"].iat[0] == "CB"   # DB aliased
    assert out.attrs["qc"]["athletes_with_multiple_labels"] == 1
