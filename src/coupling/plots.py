"""Figures. Every function returns the matplotlib Figure and, if `save` is given, writes figures/<save>.png."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C

ZONE_LABEL = {z: z.replace("_", "") for z in C.ZONES}


def _save(fig, save: str | None):
    if save:
        C.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(C.FIG_DIR / f"{save}.png", dpi=150, bbox_inches="tight")
    return fig


def heatmap(matrix: pd.DataFrame, title: str, save: str | None = None):
    m = matrix.rename(index=ZONE_LABEL, columns=ZONE_LABEL)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(m * 100, annot=True, fmt=".0f", cmap="viridis", vmin=0, vmax=100,
                mask=np.eye(len(m), dtype=bool), cbar_kws={"label": "% hours |Δp| < 1 €/MWh"}, ax=ax)
    ax.set_title(title)
    return _save(fig, save)


def quarterly_lines(q: pd.DataFrame, title: str, save: str | None = None, events: bool = True):
    """q: long frame with columns border, quarter, share."""
    wide = q.pivot(index="quarter", columns="border", values="share") * 100
    wide.index = wide.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 5))
    wide.plot(ax=ax, lw=1.6)
    ax.set_ylabel("% hours converged (ACER, <1 €/MWh)")
    ax.set_ylim(0, 100)
    if events:
        for t, lab in [(C.CRISIS_START, "crisis"), (C.CORE_FB_GOLIVE, "Core FB"),
                       (C.NORDIC_FB_GOLIVE, "Nordic FB"), (C.MTU15_GOLIVE, "15-min MTU")]:
            ax.axvline(t.tz_localize(None), color="grey", ls=":", lw=1)
            ax.text(t.tz_localize(None), 101, lab, rotation=0, fontsize=8, ha="center", va="bottom")
        ax.axvspan(C.CRISIS_START.tz_localize(None), C.CRISIS_END.tz_localize(None), color="grey", alpha=0.08)
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.set_title(title, pad=18)
    ax.set_xlabel("")
    return _save(fig, save)


def spread_histograms(panel: pd.DataFrame, borders: list[str], save: str | None = None):
    n = len(borders)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), sharex=True)
    bins = np.concatenate([-np.logspace(3, -1, 40), [0], np.logspace(-1, 3, 40)])
    for ax, name in zip(axes.flat, borders):
        s = panel.loc[panel["border"] == name, "spread"].clip(-999, 999)
        ax.hist(s, bins=bins, color="C0")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.axvspan(-C.TOL_ACER, C.TOL_ACER, color="C2", alpha=0.2)
        ax.set_title(f"{name}  ({(s.abs() < C.TOL_ACER).mean():.0%} converged)", fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.supxlabel("price spread a − b (€/MWh, symlog; green = |Δp| < 1)")
    fig.tight_layout()
    return _save(fig, save)


def hour_profile(shares: pd.DataFrame, save: str | None = None):
    """shares: columns border, hour, share. One line per border, showing non-convergence by hour."""
    wide = shares.pivot(index="hour", columns="border", values="share")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ((1 - wide) * 100).plot(ax=ax, lw=1.4)
    ax.set_ylabel("% hours NOT converged")
    ax.set_xlabel("hour of day (market time)")
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, save)


def regime_bars(tbl: pd.DataFrame, save: str | None = None):
    """tbl: index border, columns regimes, values non-convergence share."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    (tbl * 100).plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("% hours NOT converged")
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, save)


def coef_plot(per_border: pd.DataFrame, terms: list[str], save: str | None = None):
    fig, axes = plt.subplots(1, len(terms), figsize=(3.2 * len(terms), 4.5), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), terms):
        y = np.arange(len(per_border))
        ax.errorbar(per_border[t], y, xerr=1.96 * per_border[f"{t}_se"], fmt="o", ms=4)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(t, fontsize=10)
        ax.set_yticks(y, per_border.index)
    fig.supxlabel("Δ P(non-converged) per 1 SD (95% CI, day-clustered)")
    fig.tight_layout()
    return _save(fig, save)


def event_study_plot(es: pd.DataFrame, title: str, save: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(es.index, es["perm_lo"] * 100, es["perm_hi"] * 100, color="grey", alpha=0.25,
                    label="permutation 95% band")
    ax.plot(es.index, es["coef"] * 100, "o-", label="treated − control")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="C3", ls="--", lw=1)
    ax.set_xlabel("months relative to go-live (ref = −1)")
    ax.set_ylabel("Δ convergence (pp)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, save)


def mtu15_plot(tbl: pd.DataFrame, save: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = tbl.sort_values("diff")
    y = np.arange(len(t))
    ax.errorbar(t["diff"] * 100, y, xerr=[(t["diff"] - t["ci_lo"]) * 100, (t["ci_hi"] - t["diff"]) * 100],
                fmt="o")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("Δ convergence share, 12m after vs 12m before 15-min MTU (pp, 95% day-bootstrap CI)")
    return _save(fig, save)
