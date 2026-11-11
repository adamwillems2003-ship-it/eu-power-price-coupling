import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from coupling import config as cfg  # not `C`: patsy's C() is used in a formula below
from coupling import events, metrics, models


def test_acer_threshold_is_strict_and_nan_propagates():
    s = pd.Series([0.0, 0.99, 1.0, -0.5, -1.2, np.nan])
    out = metrics.converged(s, cfg.TOL_ACER)
    assert out.iloc[:5].tolist() == [1, 1, 0, 1, 0]
    assert np.isnan(out.iloc[5])


def test_region_convergence_requires_all_zones():
    p = pd.DataFrame({"A": [10, 10, 10], "B": [10.5, 12, np.nan], "C": [10.2, 10, 10]})
    out = metrics.region_convergence(p, ["A", "B", "C"])
    assert out.iloc[0] == 1 and out.iloc[1] == 0 and np.isnan(out.iloc[2])


def test_quarter_hour_divergence_hidden_by_hourly_mean():
    idx = pd.date_range(cfg.MTU15_GOLIVE, periods=8, freq="15min").tz_convert("UTC")
    a = [50, 50, 50, 50, 60, 40, 50, 50]
    b = [50, 50, 50, 50, 50, 50, 50, 50]
    q = pd.DataFrame({"A": a, "B": b}, index=idx, dtype=float)
    out = metrics.quarter_hour_convergence(q, "A", "B")
    assert out["conv_hourly_mean"].tolist() == [1.0, 1.0]
    assert out["conv_all_quarters"].tolist() == [1.0, 0.0]
    assert out["intrahour_spread_range"].iloc[1] == pytest.approx(20)


def test_border_start_masks_pre_commissioning_hours():
    idx = pd.date_range("2020-10-01", "2021-01-01", freq="h", tz="UTC", inclusive="left")
    flows = pd.DataFrame({"X-Y": np.where(idx >= pd.Timestamp("2020-11-18", tz="UTC"), 500.0, 0.0)}, index=idx)
    start = metrics.border_start(flows)["X-Y"]
    assert start == pd.Timestamp("2020-11-01", tz=cfg.MARKET_TZ)


def test_ntc_utilisation_uses_capacity_in_flow_direction():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    flow = pd.Series([500.0, -300.0], index=idx)
    ntc = pd.DataFrame({"A->B": [1000.0, 1000.0], "B->A": [600.0, 600.0]}, index=idx)
    util = models.ntc_utilisation(flow, ntc, "A", "B")
    assert util.tolist() == [0.5, 0.5]


def test_twfe_demean_matches_dummy_regression_on_unbalanced_panel():
    rng = np.random.default_rng(1)
    n_b, n_d = 6, 40
    df = pd.DataFrame([(b, d) for b in range(n_b) for d in range(n_d)], columns=["b", "d"])
    df = df.sample(frac=0.8, random_state=0).reset_index(drop=True)  # unbalanced
    df["x"] = rng.normal(size=len(df)) + df["b"] * 0.3
    df["y"] = 2.5 * df["x"] + df["b"] + 0.1 * df["d"] + rng.normal(size=len(df))
    g1, g2 = df["b"].to_numpy(), df["d"].to_numpy()
    yt, xt = events._twfe_demean(df["y"].to_numpy(), g1, g2), events._twfe_demean(df["x"].to_numpy(), g1, g2)
    beta_fwl = xt @ yt / (xt @ xt)
    beta_ols = smf.ols("y ~ x + C(b) + C(d)", df).fit().params["x"]
    assert beta_fwl == pytest.approx(beta_ols, rel=1e-6)


def synthetic_daily(effect: float, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    treated = ["T1", "T2", "T3"]
    control = ["C1", "C2", "C3", "C4", "C5"]
    golive = pd.Timestamp("2022-06-09", tz=cfg.MARKET_TZ)
    dates = pd.date_range("2021-06-09", "2023-06-08", freq="D")
    shock = pd.Series(rng.normal(0, 0.1, len(dates)), index=dates)  # common shock, e.g. the crisis
    rows = []
    for i, b in enumerate(treated + control):
        level = 0.3 + 0.05 * i
        for d in dates:
            post = d.tz_localize(cfg.MARKET_TZ) >= golive
            y = level + shock[d] + (effect if (b in treated and post) else 0) + rng.normal(0, 0.02)
            rows.append((b, d, y))
    df = pd.DataFrame(rows, columns=["border", "date", "conv"])
    design = dict(golive=golive, treated=treated, control=control,
                  window=(pd.Timestamp("2021-06-09", tz=cfg.MARKET_TZ), pd.Timestamp("2023-06-09", tz=cfg.MARKET_TZ)))
    return df, design


def test_did_recovers_effect_despite_common_shock_and_ranks_first_in_permutation():
    df, design = synthetic_daily(effect=-0.15)
    res = events.did(df, design)
    assert res["beta"] == pytest.approx(-0.15, abs=0.005)
    assert res["n_perm"] == 56  # C(8, 3)
    assert res["p_perm"] == pytest.approx(1 / 56)


def test_did_null_effect_is_not_significant():
    df, design = synthetic_daily(effect=0.0, seed=3)
    res = events.did(df, design)
    assert abs(res["beta"]) < 0.005
    assert res["p_perm"] > 0.1


def test_event_study_flat_pre_period_and_step_after():
    df, design = synthetic_daily(effect=0.2)
    es = events.event_study(df, design)
    pre = es.loc[es.index < -1, "coef"]
    post = es.loc[es.index >= 0, "coef"]
    assert pre.abs().max() < 0.01
    assert post.mean() == pytest.approx(0.2, abs=0.01)
