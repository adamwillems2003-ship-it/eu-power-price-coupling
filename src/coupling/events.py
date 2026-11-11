"""Event layer.

1. The 15-minute MTU switch (1 Oct 2025). It applied to every zone at once, so there is no control group.
   We compare 12 months before with 12 months after, which matches the seasons. Uncertainty comes from a
   bootstrap over delivery days. Descriptive only.
2. The flow-based go-lives (Core 9 Jun 2022, Nordic 30 Oct 2024). Border-level difference-in-differences
   on daily convergence shares, with border and date fixed effects. Inference uses an exact permutation
   test over treatment assignment, because there are too few borders for clustered standard errors to
   be reliable. An event study (monthly leads and lags) checks pre-trends.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import config as C
from .metrics import market_time_features


def daily(panel: pd.DataFrame, col: str = "conv", min_hours: int = 20) -> pd.DataFrame:
    """Border x market-time-day mean of `col`. Days with fewer than `min_hours` valid hours are dropped."""
    date = market_time_features(panel.index)["date"].to_numpy()
    d = (panel.assign(date=date).groupby(["border", "date"])[col]
         .agg(["mean", "count"]).reset_index())
    return d[d["count"] >= min_hours].rename(columns={"mean": col}).drop(columns="count")


# --------------------------------------------------------------------------- 15-minute MTU switch
def mtu15_before_after(panel: pd.DataFrame, months: int = 12, n_boot: int = 2000, seed: int = 0,
                       col: str = "conv") -> pd.DataFrame:
    """Per border: `col` share in the `months` before and after go-live, the difference, and a
    95% day-bootstrap CI for the difference. Also reports the median |spread| before and after."""
    rng = np.random.default_rng(seed)
    t0 = C.MTU15_GOLIVE
    pre = (t0 - pd.DateOffset(months=months), t0)
    post = (t0, t0 + pd.DateOffset(months=months))
    rows = []
    d = daily(panel, col)
    d_abs = daily(panel, "abs_spread")
    for border, g in d.groupby("border"):
        dates = pd.to_datetime(g["date"]).dt.tz_localize(C.MARKET_TZ)
        a = g.loc[(dates >= pre[0]) & (dates < pre[1]), col].to_numpy()
        b = g.loc[(dates >= post[0]) & (dates < post[1]), col].to_numpy()
        if len(a) < 30 or len(b) < 30:
            continue
        boot = (rng.choice(b, (n_boot, len(b))).mean(1) - rng.choice(a, (n_boot, len(a))).mean(1))
        ga = d_abs[d_abs["border"] == border]
        gdates = pd.to_datetime(ga["date"]).dt.tz_localize(C.MARKET_TZ)
        rows.append(dict(
            border=border, pre=a.mean(), post=b.mean(), diff=b.mean() - a.mean(),
            ci_lo=np.quantile(boot, 0.025), ci_hi=np.quantile(boot, 0.975),
            pre_days=len(a), post_days=len(b),
            median_abs_spread_pre=ga.loc[(gdates >= pre[0]) & (gdates < pre[1]), "abs_spread"].median(),
            median_abs_spread_post=ga.loc[(gdates >= post[0]) & (gdates < post[1]), "abs_spread"].median(),
        ))
    return pd.DataFrame(rows).set_index("border")


# --------------------------------------------------------------------------- DiD machinery
def _twfe_demean(x: np.ndarray, g1: np.ndarray, g2: np.ndarray, tol: float = 1e-10, max_iter: int = 500):
    """Two-way within transformation by alternating projections. Handles unbalanced panels."""
    x = x.astype(float).copy()
    n1, n2 = g1.max() + 1, g2.max() + 1
    c1, c2 = np.bincount(g1, minlength=n1), np.bincount(g2, minlength=n2)
    for _ in range(max_iter):
        x -= (np.bincount(g1, x, n1) / c1)[g1]
        m2 = (np.bincount(g2, x, n2) / c2)[g2]
        x -= m2
        if np.abs(m2).max() < tol:
            break
    return x


def _prepare(daily_df: pd.DataFrame, design: dict, col: str) -> pd.DataFrame:
    lo, hi = design["window"]
    keep = design["treated"] + design["control"]
    d = daily_df[daily_df["border"].isin(keep)].copy()
    dates = pd.to_datetime(d["date"]).dt.tz_localize(C.MARKET_TZ)
    d = d[(dates >= lo) & (dates < hi)].copy()
    d["post"] = (pd.to_datetime(d["date"]).dt.tz_localize(C.MARKET_TZ) >= design["golive"]).astype(float)
    d["treated"] = d["border"].isin(design["treated"]).astype(float)
    d["g_border"] = pd.factorize(d["border"])[0]
    d["g_date"] = pd.factorize(d["date"])[0]
    return d.dropna(subset=[col])


def did(daily_df: pd.DataFrame, design: dict, col: str = "conv") -> dict:
    """TWFE DiD: y_bd = a_b + g_d + beta * treated_b * post_d. Returns beta, a permutation p-value over
    all assignments of the treated label to borders, and raw pre/post means by group."""
    d = _prepare(daily_df, design, col)
    g1, g2 = d["g_border"].to_numpy(), d["g_date"].to_numpy()
    y = _twfe_demean(d[col].to_numpy(), g1, g2)

    def beta_for(treated_set) -> float:
        dpost = (d["border"].isin(treated_set).to_numpy() * d["post"].to_numpy()).astype(float)
        x = _twfe_demean(dpost, g1, g2)
        return float(x @ y / (x @ x))

    beta = beta_for(design["treated"])
    borders = sorted(d["border"].unique())
    k = len([b for b in design["treated"] if b in borders])
    perm = np.array([beta_for(set(s)) for s in combinations(borders, k)])
    p_value = float(np.mean(np.abs(perm) >= abs(beta) - 1e-12))
    means = d.groupby(["treated", "post"])[col].mean().unstack()
    return dict(beta=beta, p_perm=p_value, n_perm=len(perm), perm=perm, means=means,
                n_borders=len(borders), n_obs=len(d))


def event_study(daily_df: pd.DataFrame, design: dict, col: str = "conv", ref: int = -1) -> pd.DataFrame:
    """Monthly leads and lags of treated x month-relative-to-go-live, with border and date FE.
    Coefficients are relative to month `ref`. Returns coef and the permutation 2.5/97.5% band."""
    d = _prepare(daily_df, design, col)
    dates = pd.to_datetime(d["date"]).dt.tz_localize(C.MARKET_TZ)
    rel = ((dates - design["golive"]).dt.days // 30).astype(int)
    d["rel"] = rel.to_numpy()
    # 30-day bins leave slivers at the window edges. Drop bins with fewer than 20 days.
    days_per_bin = d.groupby("rel")["date"].nunique()
    d = d[d["rel"].isin(days_per_bin[days_per_bin >= 20].index)].copy()
    d["g_border"] = pd.factorize(d["border"])[0]
    d["g_date"] = pd.factorize(d["date"])[0]
    bins = sorted(b for b in d["rel"].unique() if b != ref)
    g1, g2 = d["g_border"].to_numpy(), d["g_date"].to_numpy()
    y = _twfe_demean(d[col].to_numpy(), g1, g2)

    def coefs(treated_set) -> np.ndarray:
        tr = d["border"].isin(treated_set).to_numpy()
        X = np.column_stack([_twfe_demean((tr & (d["rel"].to_numpy() == b)).astype(float), g1, g2)
                             for b in bins])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    est = coefs(design["treated"])
    borders = sorted(d["border"].unique())
    k = len([b for b in design["treated"] if b in borders])
    perms = np.array([coefs(set(s)) for s in combinations(borders, k)])
    return pd.DataFrame({"coef": est, "perm_lo": np.quantile(perms, 0.025, axis=0),
                         "perm_hi": np.quantile(perms, 0.975, axis=0)}, index=pd.Index(bins, name="month_rel"))
