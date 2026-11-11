# How well are European day-ahead power prices coupled, and where do they break apart?

ENTSO-E day-ahead prices for 9 bidding zones (NL, BE, DE-LU, FR, ES, PL, DK1, SE3, NO2), from
1 Oct 2018, the first delivery day after the DE-AT-LU split, to the last complete month.

**Questions**
1. How often do neighbouring bidding zones clear at the same price?
2. Does non-convergence cluster in specific hours and regimes (the 2021-23 energy crisis, high-RES
   hours, winter peaks)?
3. Did the switch to 15-minute day-ahead pricing (delivery 1 Oct 2025) change anything?

## Findings

_Status: the pipeline and notebook are complete and tested. The findings are filled in from
`results/summary.json` after the first full ENTSO-E pull (see "Reproduce")._

## What drives price differences: congested cross-zonal capacity

In the Single Day-Ahead Coupling (SDAC), one algorithm (Euphemia) clears every bidding zone at once.
It maximises total welfare subject to the cross-zonal capacity the TSOs make available. If no
network constraint between two zones binds, power flows from the cheaper zone to the dearer one
until their prices are equal, so they converge. Prices separate only when a constraint binds.
Under NTC coupling (e.g. FR-ES, DK1-DE), that constraint is the border's own capacity. Under
flow-based coupling (Core since 2022, Nordics since 2024), it is a critical network element whose
flows, weighted by power-transfer distribution factors, reach their limit. Such an element can sit
anywhere in the meshed grid, so two zones can decouple even when the line between them is not full.
The price spread times the scheduled flow is the congestion income the TSOs collect. Whatever pushes
large flows against limited capacity therefore drives non-convergence: wind or solar surges on one
side, demand peaks, outages, and loop flows that eat into available capacity. In the regressions,
the congestion proxy (flow utilisation) is the mechanism itself, not a separate cause.

## Method

| Step | Choice | Why |
|---|---|---|
| Convergence | \|p_a − p_b\| < 1 €/MWh; exact (< 0.01) as a sensitivity check | ACER MMR "full price convergence" |
| Time | Everything in UTC; hour-of-day and delivery day in Europe/Brussels | No DST duplicates or gaps; one local clock for all CET zones |
| 15-min prices | Hourly value = mean of the 4 quarter-hours; only complete hours kept | Like-for-like before/after comparison |
| Zones | DE-LU from 1 Oct 2018 only; interconnectors that start mid-sample (ALEGrO, COBRA, NordLink) begin at their first month of physical flow | No stitched or non-existent borders |
| Gaps, units | Strict hourly completeness, conflicting duplicates voided, currency and unit read from the raw XML (asserted EUR/MWh), negative prices kept and counted | entsoe-py's parser drops currency |
| Crisis regime | Sep 2021 to Feb 2023: months with TTF > 3× its 2015-20 mean (asserted in the notebook) | Rule-based, not eyeballed |
| Explanatory | Border-hour LPM (non-converged) and OLS (\|Δp\|) on day-ahead load and RES forecasts on both sides plus log TTF, then + flow utilisation; border, hour, weekday and month FE; SEs clustered by delivery day | Forecasts are predetermined at gate closure |
| 15-min switch | 12 months before vs 12 after, day-bootstrap CIs, plus intra-hour divergence that hourly means hide | No control group, so descriptive only |
| Flow-based (exploratory) | Border-day TWFE DiD and monthly event study; exact permutation p-values over treatment assignment | About 10 borders, too few for clustered SEs |

Verified dates: Core FB delivery 9 Jun 2022 (trading day 8 Jun); Nordic FB delivery 30 Oct 2024
(trading day 29 Oct); 15-min MTU in SDAC delivery 1 Oct 2025 (trading day 30 Sep); PL-DE implicit
coupling via Interim Coupling, delivery 18 Jun 2021.

## Limitations

* **Correlation, not causation**, except in the exploratory DiD, and even that is fragile. Core FB
  went live mid-crisis and during the French nuclear outage, and its treated CWE borders were already
  flow-based, so the treatment is a domain extension, not a switch from NTC to flow-based.
* **The 15-minute comparison has no counterfactual.** Fuel prices, RES capacity and Nordic FB (inside
  the pre-window for the Nordic borders) also changed.
* **The congestion proxy is approximate.** Flow-based borders have no ex-ante capacity series, so
  utilisation is |net flow| relative to the border's quarterly 99th percentile. NTC borders get a
  robustness check against the published day-ahead NTC.
* **Gas is monthly.** IMF/FRED TTF (USD/MMBtu). Daily TTF futures (Yahoo `TTF=F`) currently return
  no history.
* **Pairwise, not network.** A pair can diverge because of a constraint elsewhere in the grid (see
  above), and the border-level analysis does not identify which element binds.

## Reproduce

```bash
uv venv && uv pip install -e '.[dev]'
export ENTSOE_API_KEY=...                 # or: echo TOKEN > ~/.entsoe_key
python -m coupling.fetch                  # cached per zone-year in data/raw/ (not committed)
python -m coupling.clean                  # data/processed/*.parquet + price_quality.csv
jupyter nbconvert --execute --to notebook --inplace notebooks/analysis.ipynb
pytest
```

Layout: `src/coupling/` has `config` (zones, borders, verified dates), `fetch`, `clean`, `metrics`,
`models`, `events` and `plots`. `notebooks/analysis.ipynb` holds the narrative. `results/` holds the
CSV/JSON numbers behind the README, and `figures/` holds the PNGs.

Data: ENTSO-E Transparency Platform (re-use subject to ENTSO-E's terms, attribution required);
IMF Primary Commodity Prices via FRED (`PNGASEUUSDM`).
