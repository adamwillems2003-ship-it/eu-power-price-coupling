"""Study design constants: zones, borders, dates, thresholds.

Every date here was checked against a primary source. The source is cited next to each one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

# Local market time of SDAC. Every zone in the study uses CET/CEST, so one local clock serves for
# hour-of-day and delivery-day features.
MARKET_TZ = "Europe/Brussels"

# Sample: start on the first delivery day after the DE-AT-LU split, so the DE-LU series stays
# like-for-like (ENTSO-E area 10Y1001A1001A82H).
SAMPLE_START = pd.Timestamp("2018-10-01", tz=MARKET_TZ)

# Core zones, peripheral zones, and the two Nordic zones that the Nordic flow-based DiD needs.
CORE_ZONES = ["NL", "BE", "DE_LU", "FR"]
PERIPHERAL_ZONES = ["ES", "PL", "DK_1"]
NORDIC_AUX_ZONES = ["SE_3", "NO_2"]
ZONES = CORE_ZONES + PERIPHERAL_ZONES + NORDIC_AUX_ZONES

# ACER MMR: "full price convergence" means a price difference below 1 EUR/MWh.
# Source: ACER 2024 MMR, cross-zonal capacities volume, p. 10 note.
TOL_ACER = 1.0
# Sensitivity check: identical prices (ENTSO-E publishes prices to 2 decimals).
TOL_EXACT = 0.01

# SDAC switched to a 15-minute MTU on trading day 30 Sep 2025, for delivery day 1 Oct 2025.
# Source: SDAC press release, 12 Sep 2025, confirmed 1 Oct 2025.
MTU15_GOLIVE = pd.Timestamp("2025-10-01", tz=MARKET_TZ)

# Core flow-based market coupling: trading day 8 Jun 2022, delivery 9 Jun 2022 (JAO/Core announcement).
CORE_FB_GOLIVE = pd.Timestamp("2022-06-09", tz=MARKET_TZ)
# Nordic flow-based market coupling: trading day 29 Oct 2024, delivery 30 Oct 2024 (Nord Pool, Statnett).
NORDIC_FB_GOLIVE = pd.Timestamp("2024-10-30", tz=MARKET_TZ)
# Interim Coupling put the PL-DE border into implicit NTC coupling: delivery 18 Jun 2021 (PSE press release).
INTERIM_COUPLING_GOLIVE = pd.Timestamp("2021-06-18", tz=MARKET_TZ)

# Energy-crisis regime: the contiguous months in which the monthly TTF price (FRED PNGASEUUSDM) was
# more than 3x its 2015-2020 mean, i.e. Sep 2021 to Feb 2023. The notebook recomputes the rule and
# asserts these bounds.
CRISIS_TTF_MULTIPLE = 3.0
CRISIS_START = pd.Timestamp("2021-09-01", tz=MARKET_TZ)
CRISIS_END = pd.Timestamp("2023-03-01", tz=MARKET_TZ)  # exclusive

# Winter peak: Dec-Feb, working days, 17:00-20:59 local.
WINTER_MONTHS = (12, 1, 2)
PEAK_HOURS = range(17, 21)

# Gas price: IMF "Natural gas, EU" (TTF), USD/MMBtu, monthly, via FRED. No key needed.
# Daily TTF futures (Yahoo TTF=F) return no history as of Nov 2026.
FRED_GAS_SERIES = "PNGASEUUSDM"


@dataclass(frozen=True)
class Border:
    a: str
    b: str
    ccr: str  # capacity calculation region
    regime: str  # coupling regime at sample start

    @property
    def name(self) -> str:
        return f"{self.a}-{self.b}"


# Bidding-zone borders between study zones that have a physical interconnector.
# Interconnectors commissioned during the sample (BE-DE_LU ALEGrO, NL-DK_1 COBRA, DE_LU-NO_2 NordLink)
# are not hard-coded: the first month with non-zero physical flow, taken from the data, sets their start.
BORDERS = [
    Border("NL", "BE", "Core", "CWE flow-based"),
    Border("NL", "DE_LU", "Core", "CWE flow-based"),
    Border("BE", "FR", "Core", "CWE flow-based"),
    Border("BE", "DE_LU", "Core", "CWE flow-based"),
    Border("FR", "DE_LU", "Core", "CWE flow-based"),
    Border("DE_LU", "PL", "Core", "explicit -> NTC implicit (2021-06-18)"),
    Border("FR", "ES", "SWE", "NTC"),
    Border("DE_LU", "DK_1", "Hansa", "NTC"),
    Border("NL", "DK_1", "Hansa", "NTC"),
    Border("NL", "NO_2", "Hansa/other", "NTC"),
    Border("DE_LU", "NO_2", "Hansa/other", "NTC"),
    Border("DK_1", "SE_3", "Nordic", "NTC"),
    Border("DK_1", "NO_2", "Nordic", "NTC"),
]
BORDER_BY_NAME = {b.name: b for b in BORDERS}

# DiD designs. Treated and control sets exclude borders whose regime changed for another reason
# inside the window.
CORE_DID = dict(
    golive=CORE_FB_GOLIVE,
    # CWE borders moved from CWE flow-based to Core flow-based. That is a domain extension,
    # not a switch from NTC to flow-based.
    treated=["NL-BE", "NL-DE_LU", "BE-FR", "BE-DE_LU", "FR-DE_LU"],
    # DE_LU-PL is excluded: it entered implicit coupling on 2021-06-18, inside the pre-period.
    control=["FR-ES", "DE_LU-DK_1", "NL-DK_1", "NL-NO_2", "DE_LU-NO_2", "DK_1-SE_3", "DK_1-NO_2"],
    window=(pd.Timestamp("2021-06-09", tz=MARKET_TZ), pd.Timestamp("2023-06-09", tz=MARKET_TZ)),
)
NORDIC_DID = dict(
    golive=NORDIC_FB_GOLIVE,
    treated=["DK_1-SE_3", "DK_1-NO_2"],
    # Hansa borders (DK_1-DE_LU, DK_1-NL, NO_2-*) are excluded: Nordic flow-based brought them into
    # advanced hybrid coupling, so they are partially treated.
    control=["NL-BE", "NL-DE_LU", "BE-FR", "BE-DE_LU", "FR-DE_LU", "DE_LU-PL", "FR-ES"],
    # The window ends before the 15-minute MTU switch.
    window=(pd.Timestamp("2023-10-30", tz=MARKET_TZ), MTU15_GOLIVE),
)
