# ============================================================================
# Data models for the optimization engine
# ============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================================
# Costants for data processing and optimization
# ============================================================================

INTERVENTION_TYPES = ["stunting", "wasting", "severe_wasting"]

COUNT_COLS = {
    "stunting":       "Count_stunting",
    "wasting":        "Count_wasting",
    "severe_wasting": "Count_severe_wasting",
}
COST_COLS = {
    "stunting":       "Cost_stunting",
    "wasting":        "Cost_wasting",
    "severe_wasting": "Cost_severe_wasting",
}

# ============================================================================
# Scaling factors for numerical stability in optimization
# ============================================================================

COUNT_SCALE_FACTOR = 1000
BUDGET_SCALE_FACTOR = 1e6
COST_SCALE_FACTOR = 1000  # Costs must be scaled consistently: 1e6 / 1000 = 1000

# ============================================================================
# Dataclasses for problem definition and stakeholder preferences
# (avoid __init__ for better readability and immutability)
# ============================================================================

@dataclass
class AllocationProblem:
    df: pd.DataFrame
    total_budget: float
    countries: Optional[List[str]] = None
    demographic_filter: Optional[List[str]] = None
    filtered_df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        df = self.df.copy()
        # Scale counts and costs for numerical stability
        for col in COUNT_COLS.values():
            if col in df.columns:
                df[col] = df[col] / COUNT_SCALE_FACTOR
        for col in COST_COLS.values():
            if col in df.columns:
                df[col] = df[col] / COST_SCALE_FACTOR
        # Scale budget
        self.total_budget = self.total_budget / BUDGET_SCALE_FACTOR
        
        if self.countries:
            df = df[df["ISO3"].isin(self.countries)]
        if self.demographic_filter:
            df = df[df["Demographic_group"].isin(self.demographic_filter)]
        self.filtered_df = df.reset_index(drop=True)



@dataclass
class StakeholderPreferences:
    """Preference bundle for a single stakeholder."""

    metric_weights: Dict[str, float]
    demographic_constraints: Dict[str, float]
    fairness_mode: str  # utilitarian | max-min | proportional
    min_coverage_share: Optional[float] = None


@dataclass
class StakeholderProfile:
    """One stakeholder plus influence weight for aggregation."""

    name: str
    preferences: StakeholderPreferences
    influence: float = 1.0