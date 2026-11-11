"""Explanatory layer: what predicts non-convergence and the size of the spread on a border-hour.

Two nested specifications, both linear (LPM for the 0/1 outcome, OLS for |spread|):

* A ("fundamentals"): day-ahead forecasts of load and wind+solar on both sides, the gas price, and
  hour, weekday, month and border fixed effects. These are known before gate closure, so they are
  predetermined relative to the auction.
* B ("+ congestion"): A plus a flow-utilisation proxy. Congestion is the mechanism through which prices
  separate, so B is descriptive. It shows how much of the variation the congestion proxy absorbs; it is
  not a causal estimate.

Standard errors are clustered by delivery day, because all hours of a day clear in one auction.
Continuous regressors are z-scored, so each coefficient is the change in the outcome per 1 SD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Not imported as `C`: patsy formulas below use C(...) for categoricals and evaluate names in this module.
from . import config as cfg
from .metrics import market_time_features, res_share


def flow_utilisation(net_flow: pd.Series) -> pd.Series:
    """|net flow| scaled by the border's 99th-percentile |flow| within each market-time quarter.

    Flow-based borders have no NTC, so no ex-ante capacity series exists across the whole sample.
    The quarterly p99 is a uniform stand-in for the maximum usable capacity, and lets that capacity
    change over time (new lines, the 70% minRAM ramp).
    """
    a = net_flow.abs()
    q = a.index.tz_convert(cfg.MARKET_TZ).tz_localize(None).to_period("Q")
    cap = a.groupby(q).transform(lambda x: x.quantile(0.99))
    return (a / cap).clip(upper=1.5)


def ntc_utilisation(net_flow: pd.Series, ntc: pd.DataFrame, a: str, b: str) -> pd.Series:
    """|net flow| / day-ahead NTC in the direction of the flow (NTC borders only)."""
    cap = ntc[f"{a}->{b}"].where(net_flow >= 0, ntc[f"{b}->{a}"])
    return (net_flow.abs() / cap.where(cap > 0)).clip(upper=1.5)


def load_relative(load: pd.Series) -> pd.Series:
    """Load relative to the zone's calendar-year mean. Removes demand trends and zone size."""
    year = load.index.tz_convert(cfg.MARKET_TZ).year
    return load / load.groupby(year).transform("mean")


def regression_frame(prices, fundamentals, flows, ntc, gas: pd.Series, borders=cfg.BORDERS,
                     starts: pd.Series | None = None) -> pd.DataFrame:
    month_start = prices.index.tz_convert(cfg.MARKET_TZ).tz_localize(None).to_period("M").to_timestamp()
    gas_naive = gas.copy()
    gas_naive.index = gas_naive.index.tz_localize(None)
    ttf = pd.Series(month_start.map(gas_naive).to_numpy(dtype=float), index=prices.index)
    feats = market_time_features(prices.index)
    frames = []
    for b in borders:
        spread = prices[b.a] - prices[b.b]
        if starts is not None and pd.notna(starts.get(b.name)):
            spread = spread.where(spread.index >= starts[b.name])
        df = pd.DataFrame({
            "border": b.name,
            "spread": spread,
            "abs_spread": spread.abs(),
            "nonconv": (spread.abs() >= cfg.TOL_ACER).astype(float).where(spread.notna()),
            "res_a": res_share(fundamentals, b.a),
            "res_b": res_share(fundamentals, b.b),
            "load_rel_a": load_relative(fundamentals[b.a]["load_fc"]),
            "load_rel_b": load_relative(fundamentals[b.b]["load_fc"]),
            "flow_util": flow_utilisation(flows[b.name]),
            "ntc_util": ntc_utilisation(flows[b.name], ntc, b.a, b.b),
            "log_ttf": np.log(ttf),
        })
        frames.append(df.join(feats))
    out = pd.concat(frames)
    out.index.name = "ts_utc"
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["spread"])


def zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df


FE = "C(border) + C(hour) + C(dow) + C(month)"
SPEC_A = "res_a + res_b + load_rel_a + load_rel_b + log_ttf"
SPEC_B = SPEC_A + " + flow_util"


def fit(frame: pd.DataFrame, outcome: str, spec: str, fe: str = FE):
    cols = [c.strip() for c in spec.split("+")]
    d = frame.dropna(subset=[outcome, *cols]).copy()
    d = zscore(d, cols)
    d["date_cluster"] = pd.factorize(d["date"])[0]
    model = smf.ols(f"{outcome} ~ {spec} + {fe}", data=d)
    return model.fit(cov_type="cluster", cov_kwds={"groups": d["date_cluster"].to_numpy()})


def coef_table(results: dict[str, object], terms: list[str]) -> pd.DataFrame:
    """Side-by-side coefficients (SE) for the named fitted models."""
    rows = {}
    for name, r in results.items():
        col = {}
        for t in terms:
            if t in r.params.index:
                col[t] = f"{r.params[t]:.3f} ({r.bse[t]:.3f})"
            else:
                col[t] = ""
        col["R2"] = f"{r.rsquared:.3f}"
        col["N"] = f"{int(r.nobs):,}"
        rows[name] = col
    return pd.DataFrame(rows)


def per_border(frame: pd.DataFrame, outcome: str = "nonconv", spec: str = SPEC_B) -> pd.DataFrame:
    """Re-estimate the spec separately for each border (border FE dropped)."""
    rows = []
    terms = [c.strip() for c in spec.split("+")]
    for name, d in frame.groupby("border"):
        try:
            r = fit(d, outcome, spec, fe="C(hour) + C(dow) + C(month)")
        except (ValueError, np.linalg.LinAlgError):
            continue
        row = {"border": name, "N": int(r.nobs), "mean_outcome": d[outcome].mean(), "R2": r.rsquared}
        for t in terms:
            row[t] = r.params.get(t, np.nan)
            row[f"{t}_se"] = r.bse.get(t, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("border")
