"""
Constrained Optimization Engine for Fairness-Aware Budget Allocation.

Skeleton-only version: implementation intentionally left as TODOs.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pulp


# ============================================================================
# 1. DATA STRUCTURES & INPUTS
# ============================================================================
INTERVENTION_TYPES = ["stunting", "wasting", "severe_wasting"]

COUNT_COLS = {
    "stunting":       "Count_Stunting",
    "wasting":        "Count_Wasting",
    "severe_wasting": "Count_Severe_Wasting",
}
COST_COLS = {
    "stunting":       "Cost_stunting",
    "wasting":        "Cost_wasting",
    "severe_wasting": "Cost_severe_wasting",
}

@dataclass
class AllocationProblem:
    df: pd.DataFrame
    total_budget: float
    countries: Optional[List[str]] = None
    demographic_filter: Optional[List[str]] = None
    filtered_df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        df = self.df.copy()
        if self.countries:
            df = df[df["ISO3"].isin(self.countries)]
        if self.demographic_filter:
            df = df[df["Demographic_Group"].isin(self.demographic_filter)]
        self.filtered_df = df.reset_index(drop=True)



@dataclass
class StakeholderPreferences:
    """Preference bundle for a single stakeholder."""

    metric_weights: Dict[str, float]
    demographic_constraints: Dict[str, float]
    fairness_mode: str  # utilitarian | max-min | proportional | weighted-log


@dataclass
class StakeholderProfile:
    """One stakeholder plus influence weight for aggregation."""

    name: str
    preferences: StakeholderPreferences
    influence: float = 1.0


# ============================================================================
# 2. BASELINE UTILITARIAN MODEL
# ============================================================================


class UtilitarianOptimizer:
    """
    Standard LP baseline using PuLP with HiGHS solver.

    Goal:
        maximize total treated children
    Subject to:
        global budget constraint
        non-negative allocation variables
    """

    def __init__(self, problem: AllocationProblem):
        self.problem = problem
        self.model = pulp.LpProblem("Utilitarian_Allocation", pulp.LpMaximize)
        self.allocation_vars = {}
        self.cost_map = {}  # Pre-computed cost lookup: (iso3, demographic_group, itype) → cost

    def _build_cost_map(self):
        """Pre-compute cost mapping to avoid repeated DataFrame lookups."""
        df = self.problem.filtered_df
        for row in df.to_dict('records'):
            iso3 = row["ISO3"]
            demographic_group = row["Demographic_Group"]
            for itype in INTERVENTION_TYPES:
                key = (iso3, demographic_group, itype)
                self.cost_map[key] = float(pd.to_numeric(row[COST_COLS[itype]], errors="coerce"))

    def setup_variables(self):
        """Create one LpVariable per (iso3, demographic_group, intervention).
        Key structure: (iso3, demographic_group, itype) → LpVariable
        upBound = available children → encodes capacity without a separate constraint."""
        df = self.problem.filtered_df
        for row in df.to_dict('records'):
            iso3 = row["ISO3"]
            demographic_group = row["Demographic_Group"]
            for itype in INTERVENTION_TYPES:
                key = (iso3, demographic_group, itype)
                self.allocation_vars[key] = pulp.LpVariable(
                    name=f"x_{iso3}_{demographic_group}_{itype}",
                    lowBound=0,
                    upBound=float(row[COUNT_COLS[itype]]),
                    cat=pulp.LpContinuous,
                )

    def add_objective(self):
        """Maximize total treated children (unweighted sum across all allocation variables)."""
        self.model += pulp.lpSum(
            self.allocation_vars[key]
            for key in self.allocation_vars
        ), "Total_Treated_Children"

    def add_budget_constraint(self):
        """Enforce sum(x[key] * cost[key]) <= total_budget."""
        constraint_expr = pulp.lpSum(
            self.allocation_vars[key] * self.cost_map[key]
            for key in self.allocation_vars
        )
        self.model += constraint_expr <= self.problem.total_budget, "Global_Budget"

    def solve(self) -> Dict:
        """Run optimization using PuLP with HiGHS solver and return structured solution."""
        self.setup_variables()
        self._build_cost_map()  # Pre-compute costs before building constraints
        self.add_objective()
        self.add_budget_constraint()
        
        # Solve using HiGHS solver
        self.model.solve(pulp.HiGHS(msg=False))
        
        return self._extract_solution()

    def _extract_solution(self) -> Dict:
        """Convert solver output to a summary dict + per-row allocation DataFrame."""
        df = self.problem.filtered_df
        rows = []
        for base_row in df.to_dict('records'):
            iso3 = base_row["ISO3"]
            demographic_group = base_row["Demographic_Group"]
            record = {"ISO3": base_row["ISO3"], "Country": base_row["Country"], "Demographic_Group": base_row["Demographic_Group"]}
            
            for itype in INTERVENTION_TYPES:
                key = (iso3, demographic_group, itype)
                treated = pulp.value(self.allocation_vars[key]) or 0.0
                record[f"treated_{itype}"] = treated
                record[f"spend_{itype}"] = treated * self.cost_map[key]
            
            record["total_treated"] = sum(record[f"treated_{t}"] for t in INTERVENTION_TYPES)
            record["total_spend"] = sum(record[f"spend_{t}"] for t in INTERVENTION_TYPES)
            rows.append(record)

        allocation_df = pd.DataFrame(rows)
        total_spend = allocation_df["total_spend"].sum()

        return {
            "status": pulp.LpStatus[self.model.status],
            "total_treated": pulp.value(self.model.objective) or 0.0,
            "total_spend": total_spend,
            "budget": self.problem.total_budget,
            "budget_utilisation_pct": 100 * total_spend / self.problem.total_budget if self.problem.total_budget > 0 else 0,
            "allocation_df": allocation_df,
        }


class ConstrainedUtilitarianOptimizer(UtilitarianOptimizer):
    """
    Constrained utilitarian optimizer with equity budget caps and minimum allocations.

    Goal:
        maximize total treated children (same as UtilitarianOptimizer)
    Subject to:
        global budget constraint
        country-level budget caps (e.g., no country gets >60% of budget)
        demographic-level budget caps (optional)
        demographic-level minimum allocations (optional, e.g., rural must get >=30%)
        non-negative allocation variables

    This prevents greedy allocation while still maximizing efficiency.
    """

    def __init__(
        self,
        problem: AllocationProblem,
        country_cap: float = 0.5,
        demographic_cap: Optional[float] = None,
        demographic_min_share: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters:
            problem: AllocationProblem instance
            country_cap: Maximum fraction of total budget per country (default 0.5 = 50%)
            demographic_cap: Maximum fraction of total budget per demographic group (optional)
            demographic_min_share: Dict of {demographic_group: min_fraction} constraints.
                                 E.g., {"Rural": 0.30} means rural must get >=30% of budget
        """
        super().__init__(problem)
        self.country_cap = country_cap
        self.demographic_cap = demographic_cap
        self.demographic_min_share = demographic_min_share or {}
        self.model = pulp.LpProblem("Constrained_Utilitarian_Allocation", pulp.LpMaximize)

    def add_country_budget_caps(self):
        """Enforce per-country budget cap: sum(spend by country) <= cap * total_budget."""
        df = self.problem.filtered_df
        # Pre-compute mapping from iso3 to country
        iso3_to_country = df[["ISO3", "Country"]].drop_duplicates().set_index("ISO3")["Country"].to_dict()
        
        for country in df["Country"].unique():
            country_vars = [key for key in self.allocation_vars if iso3_to_country.get(key[0]) == country]
            
            if country_vars:
                constraint_expr = pulp.lpSum(
                    self.allocation_vars[key] * self.cost_map[key]
                    for key in country_vars
                )
                self.model += constraint_expr <= self.country_cap * self.problem.total_budget, f"Country_Cap_{country}"

    def add_demographic_budget_caps(self):
        """Enforce per-demographic budget cap: sum(spend by demographic) <= cap * total_budget."""
        if self.demographic_cap is None:
            return
        
        for demographic in self.problem.filtered_df["Demographic_Group"].unique():
            demographic_vars = [key for key in self.allocation_vars if key[1] == demographic]
            
            if demographic_vars:
                constraint_expr = pulp.lpSum(
                    self.allocation_vars[key] * self.cost_map[key]
                    for key in demographic_vars
                )
                self.model += constraint_expr <= self.demographic_cap * self.problem.total_budget, f"Demographic_Cap_{demographic}"

    def add_demographic_minimum_constraints(self):
        """Enforce per-demographic minimum: sum(spend by demographic) >= min_share * total_budget."""
        for demographic, min_share in self.demographic_min_share.items():
            demographic_vars = [key for key in self.allocation_vars if key[1] == demographic]
            
            if demographic_vars:
                constraint_expr = pulp.lpSum(
                    self.allocation_vars[key] * self.cost_map[key]
                    for key in demographic_vars
                )
                self.model += constraint_expr >= min_share * self.problem.total_budget, f"Demographic_Min_{demographic}"

    def solve(self) -> Dict:
        """Run constrained optimization with budget caps and demographic constraints."""
        self.setup_variables()
        self._build_cost_map()  # Pre-compute costs before building constraints
        self.add_objective()
        self.add_budget_constraint()
        self.add_country_budget_caps()
        self.add_demographic_budget_caps()
        self.add_demographic_minimum_constraints()
        
        # Solve using HiGHS solver
        self.model.solve(pulp.HiGHS(msg=False))
        
        return self._extract_solution()


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "processed" / "master_df_with_counts_and_costs.csv"
    df = pd.read_csv(data_path)

    # Use all countries in the dataset
    problem = AllocationProblem(df=df, total_budget=10_000_000)
    print(f"Testing with {df['Country'].nunique()} countries\n")
    print(df.groupby("Country")[["Cost_stunting", "Cost_wasting", "Cost_severe_wasting"]].mean())
    print(df.head(10))
    print(df.columns)
    print(df[['Country', 'Count_Stunting', 'Cost_stunting', 'Count_Wasting', 'Cost_wasting']].head())
    
    # Run baseline utilitarian (no constraints)
    print("=" * 70)
    print("BASELINE UTILITARIAN OPTIMIZER (No Constraints)")
    print("=" * 70)
    solution_baseline = UtilitarianOptimizer(problem).solve()
    
    print(f"Status            : {solution_baseline['status']}")
    print(f"Total treated     : {solution_baseline['total_treated']:,.0f}")
    print(f"Total spend       : ${solution_baseline['total_spend']:,.0f}")
    print(f"Budget utilisation: {solution_baseline['budget_utilisation_pct']:.1f}%")
    
    print("\n--- Summary by Country ---")
    by_country_baseline = solution_baseline["allocation_df"].groupby("Country").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    by_country_baseline["pct_spend"] = 100 * by_country_baseline["total_spend"] / solution_baseline["total_spend"]
    by_country_baseline["cost_per_child"] = by_country_baseline["total_spend"] / by_country_baseline["total_treated"]
    by_country_baseline = by_country_baseline.sort_values("total_spend", ascending=False)
    print(by_country_baseline[["Country", "total_treated", "total_spend", "pct_spend", "cost_per_child"]].head(10).to_string(index=False))
    
    # Run constrained utilitarian (10% cap per country - very tight for visible tradeoff)
    print("\n" + "=" * 70)
    print("CONSTRAINED UTILITARIAN OPTIMIZER (10% Budget Cap per Country)")
    print("=" * 70)
    solution_constrained = ConstrainedUtilitarianOptimizer(problem, country_cap=0.1).solve()
    
    print(f"Status            : {solution_constrained['status']}")
    print(f"Total treated     : {solution_constrained['total_treated']:,.0f}")
    print(f"Total spend       : ${solution_constrained['total_spend']:,.0f}")
    print(f"Budget utilisation: {solution_constrained['budget_utilisation_pct']:.1f}%")
    
    print("\n--- Summary by Country ---")
    by_country_constrained = solution_constrained["allocation_df"].groupby("Country").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    by_country_constrained["pct_spend"] = 100 * by_country_constrained["total_spend"] / solution_constrained["total_spend"]
    by_country_constrained["cost_per_child"] = by_country_constrained["total_spend"] / by_country_constrained["total_treated"]
    by_country_constrained = by_country_constrained.sort_values("total_spend", ascending=False)
    print(by_country_constrained[["Country", "total_treated", "total_spend", "pct_spend", "cost_per_child"]].to_string(index=False))
    
    # Run with demographic minimum constraint (Rural >= 30%)
    print("\n" + "=" * 70)
    print("CONSTRAINED UTILITARIAN + DEMOGRAPHIC MINIMUM (Rural >= 30%, 10% Country Cap)")
    print("=" * 70)
    solution_demographic_constrained = ConstrainedUtilitarianOptimizer(
        problem, 
        country_cap=0.1,
        demographic_min_share={"Rural": 0.30}
    ).solve()
    
    print(f"Status            : {solution_demographic_constrained['status']}")
    print(f"Total treated     : {solution_demographic_constrained['total_treated']:,.0f}")
    print(f"Total spend       : ${solution_demographic_constrained['total_spend']:,.0f}")
    print(f"Budget utilisation: {solution_demographic_constrained['budget_utilisation_pct']:.1f}%")
    
    print("\n--- Allocation by Demographic Group ---")
    by_demographic = solution_demographic_constrained["allocation_df"].groupby("Demographic_Group").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    by_demographic["pct_spend"] = 100 * by_demographic["total_spend"] / solution_demographic_constrained["total_spend"]
    by_demographic["cost_per_child"] = by_demographic["total_spend"] / by_demographic["total_treated"]
    by_demographic = by_demographic.sort_values("total_spend", ascending=False)
    print(by_demographic.to_string(index=False))
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Country-only Caps vs. Country + Demographic Minimum")
    print("=" * 70)
    efficiency_loss_demographic = solution_constrained["total_treated"] - solution_demographic_constrained["total_treated"]
    efficiency_loss_demographic_pct = 100 * efficiency_loss_demographic / solution_constrained["total_treated"]
    print(f"Efficiency loss (children not treated): {efficiency_loss_demographic:,.0f} ({efficiency_loss_demographic_pct:.1f}%)")
    print(f"\nCountry-only caps: {solution_constrained['total_treated']:,.0f} children treated")
    print(f"With rural minimum: {solution_demographic_constrained['total_treated']:,.0f} children treated")
    print(f"\nNote: Forcing allocation to Rural (higher-cost demographic) reduces overall efficiency.")
    
    # Overall comparison
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON: All Three Scenarios")
    print("=" * 70)
    baseline_loss = solution_baseline["total_treated"] - solution_constrained["total_treated"]
    demographic_loss = solution_constrained["total_treated"] - solution_demographic_constrained["total_treated"]
    total_loss = solution_baseline["total_treated"] - solution_demographic_constrained["total_treated"]
    
    print(f"Baseline (no constraints)              : {solution_baseline['total_treated']:>12,.0f} children")
    print(f"Country-level constraints              : {solution_constrained['total_treated']:>12,.0f} children (loss: {baseline_loss:>8,.0f})")
    print(f"Country + Demographic constraints      : {solution_demographic_constrained['total_treated']:>12,.0f} children (loss: {total_loss:>8,.0f})")
    print(f"\nCost of fairness constraints:")
    print(f"  Country redistribution alone         : {100*baseline_loss/solution_baseline['total_treated']:.2f}%")
    print(f"  Additional demographic constraint    : {100*demographic_loss/solution_constrained['total_treated']:.2f}%")
    
    # Debug: check if constraints are actually binding
    print("\n" + "=" * 70)
    print("DEBUG: Constraint Analysis")
    print("=" * 70)
    print(f"Baseline budget utilisation: {solution_baseline['budget_utilisation_pct']:.1f}%")
    print(f"Constrained budget utilisation: {solution_constrained['budget_utilisation_pct']:.1f}%")
    print("\nBaseline - Max country spend as % of total:")
    print(by_country_baseline[["Country", "pct_spend"]].sort_values("pct_spend", ascending=False).head(3).to_string(index=False))
    print("\nConstrained - Max country spend as % of total:")
    print(by_country_constrained[["Country", "pct_spend"]].sort_values("pct_spend", ascending=False).head(3).to_string(index=False))
    
    # Check cost efficiency by demographic
    print("\n" + "=" * 70)
    print("DEMOGRAPHIC COST ANALYSIS")
    print("=" * 70)
    
    print("\nBaseline allocation by demographic (top 5):")
    dem_baseline = solution_baseline["allocation_df"].groupby("Demographic_Group").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    dem_baseline["cost_per_child"] = dem_baseline["total_spend"] / dem_baseline["total_treated"]
    dem_baseline = dem_baseline.sort_values("total_spend", ascending=False)
    print(dem_baseline[["Demographic_Group", "total_treated", "cost_per_child"]].head(5).to_string(index=False))
    
    print("\nConstrained allocation by demographic (top 5):")
    dem_constrained = solution_constrained["allocation_df"].groupby("Demographic_Group").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    dem_constrained["cost_per_child"] = dem_constrained["total_spend"] / dem_constrained["total_treated"]
    dem_constrained = dem_constrained.sort_values("total_spend", ascending=False)
    print(dem_constrained[["Demographic_Group", "total_treated", "cost_per_child"]].head(5).to_string(index=False))
    
    print("\nWith Rural >=30% minimum:")
    dem_demographic = solution_demographic_constrained["allocation_df"].groupby("Demographic_Group").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    dem_demographic["cost_per_child"] = dem_demographic["total_spend"] / dem_demographic["total_treated"]
    dem_demographic = dem_demographic.sort_values("total_spend", ascending=False)
    print(dem_demographic[dem_demographic["total_spend"] > 0][["Demographic_Group", "total_treated", "cost_per_child"]].to_string(index=False))


# ============================================================================
# 3. PREFERENCE ELICITATION MODULE
# ============================================================================

class PreferenceElicitor:
    """
    Helpers for loading, validating, and normalizing stakeholder preferences.

    Expected responsibilities:
      - load preferences from JSON
      - validate metric weights
      - validate demographic/fairness constraints
      - normalize input into StakeholderPreferences-ready structures

    Constraint convention used here:
      demographic_constraints = {
          "rural_min_share": 0.40,
          "urban_max_share": 0.60,
          "female_min_share": 0.45,
          "male_max_share": 0.55,
          "poorest_quintile_min_share": 0.25,
      }

    Each key follows the pattern:
        <group_value>_<min|max>_share

    Examples:
        rural_min_share = 0.40
        female_min_share = 0.50
        q1_min_share = 0.20
    """

    ALLOWED_FAIRNESS_MODES = {"utilitarian", "weighted-log", "max-min", "proportional"}

    def __init__(
        self,
        metric_weights: Dict[str, float],
        demographic_constraints: Optional[Dict[str, float]] = None,
        fairness_mode: str = "utilitarian",
    ):
        self.metric_weights = metric_weights
        self.demographic_constraints = demographic_constraints or {}
        self.fairness_mode = fairness_mode

    @staticmethod
    def from_json(filepath: str) -> "PreferenceElicitor":
        """
        Load preferences from JSON.

        Supported JSON shape:
        {
          "metric_weights": {
            "stunting": 1.0,
            "wasting": 1.2,
            "severe_wasting": 1.5
          },
          "demographic_constraints": {
            "rural_min_share": 0.4,
            "urban_max_share": 0.6,
            "female_min_share": 0.45
          },
          "fairness_mode": "weighted-log"
        }
        """
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Preference JSON must contain a top-level object.")

        return PreferenceElicitor(
            metric_weights=payload.get("metric_weights", {}),
            demographic_constraints=payload.get("demographic_constraints", {}),
            fairness_mode=payload.get("fairness_mode", "utilitarian"),
        )

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _parse_constraint_key(key: str) -> Dict[str, str]:
        """
        Parse keys like:
            rural_min_share
            urban_max_share
            female_min_share
            q1_min_share

        Returns:
            {
              "group_value": "rural",
              "bound_type": "min",
              "quantity": "share"
            }
        """
        parts = key.split("_")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid constraint key '{key}'. Expected pattern <group_value>_<min|max>_share."
            )

        quantity = parts[-1]
        bound_type = parts[-2]
        group_value = "_".join(parts[:-2])

        if not group_value:
            raise ValueError(f"Invalid constraint key '{key}': missing group value.")
        if bound_type not in {"min", "max"}:
            raise ValueError(
                f"Invalid constraint key '{key}': bound must be 'min' or 'max'."
            )
        if quantity != "share":
            raise ValueError(
                f"Invalid constraint key '{key}': currently only '_share' constraints are supported."
            )

        return {
            "group_value": group_value,
            "bound_type": bound_type,
            "quantity": quantity,
        }

    def validate(self) -> bool:
        """
        Validate weights, constraint ranges, and fairness mode.

        Rules:
          - fairness_mode must be one of the supported modes
          - metric_weights must be a non-empty dict
          - each weight must be numeric and >= 0
          - at least one metric weight must be strictly positive
          - demographic constraint values must be numeric and in [0, 1]
          - min_share and max_share for the same group cannot conflict
        """
        if self.fairness_mode not in self.ALLOWED_FAIRNESS_MODES:
            raise ValueError(
                f"Unsupported fairness_mode '{self.fairness_mode}'. "
                f"Allowed values: {sorted(self.ALLOWED_FAIRNESS_MODES)}"
            )

        if not isinstance(self.metric_weights, dict) or not self.metric_weights:
            raise ValueError("metric_weights must be a non-empty dictionary.")

        positive_weight_found = False
        for metric, weight in self.metric_weights.items():
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("Each metric name must be a non-empty string.")
            if not self._is_number(weight):
                raise ValueError(f"Metric weight for '{metric}' must be numeric.")
            if weight < 0:
                raise ValueError(f"Metric weight for '{metric}' cannot be negative.")
            if weight > 0:
                positive_weight_found = True

        if not positive_weight_found:
            raise ValueError("At least one metric weight must be strictly positive.")

        if not isinstance(self.demographic_constraints, dict):
            raise ValueError("demographic_constraints must be a dictionary.")

        grouped_bounds: Dict[str, Dict[str, float]] = {}

        for key, value in self.demographic_constraints.items():
            parsed = self._parse_constraint_key(key)

            if not self._is_number(value):
                raise ValueError(f"Constraint '{key}' must have a numeric value.")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"Constraint '{key}' must be between 0 and 1.")

            group_value = parsed["group_value"]
            bound_type = parsed["bound_type"]
            grouped_bounds.setdefault(group_value, {})[bound_type] = float(value)

        for group_value, bounds in grouped_bounds.items():
            if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
                raise ValueError(
                    f"Conflicting constraints for '{group_value}': "
                    f"min_share ({bounds['min']}) cannot exceed max_share ({bounds['max']})."
                )

        return True

    def normalized_metric_weights(self) -> Dict[str, float]:
        """Return metric weights normalized to sum to 1 when possible."""
        total = float(sum(self.metric_weights.values()))
        if total <= 0:
            raise ValueError("Cannot normalize metric weights: total weight must be > 0.")
        return {metric: float(weight) / total for metric, weight in self.metric_weights.items()}

    def normalized_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert flat constraint keys into a structured representation.

        Example input:
            {
              "rural_min_share": 0.40,
              "urban_max_share": 0.60,
              "female_min_share": 0.45
            }

        Example output:
            {
              "rural": {"type": "share", "min": 0.40},
              "urban": {"type": "share", "max": 0.60},
              "female": {"type": "share", "min": 0.45}
            }
        """
        structured: Dict[str, Dict[str, Any]] = {}

        for key, value in self.demographic_constraints.items():
            parsed = self._parse_constraint_key(key)
            group_value = parsed["group_value"]
            bound_type = parsed["bound_type"]
            quantity = parsed["quantity"]

            structured.setdefault(group_value, {"type": quantity})
            structured[group_value][bound_type] = float(value)

        return structured

    def to_preferences(self) -> StakeholderPreferences:
        """
        Validate and convert to StakeholderPreferences dataclass.

        We store:
          - normalized metric weights
          - raw flat demographic constraints
          - fairness mode

        The flat constraint format is kept for compatibility with downstream
        optimizer code, while `normalized_constraints()` is available if a
        structured form is preferred during constraint generation.
        """
        self.validate()
        return StakeholderPreferences(
            metric_weights=self.normalized_metric_weights(),
            demographic_constraints={
                key: float(value) for key, value in self.demographic_constraints.items()
            },
            fairness_mode=self.fairness_mode,
        )

# ============================================================================
# 4. FAIRNESS ALLOCATION ALGORITHMS
# ============================================================================


class FairnessOptimizer:
    """
    Multi-objective fairness-aware optimizer.

    Expected modes:
      - utilitarian
      - weighted-log (proxy if staying LP)
      - max-min fairness
      - proportional fairness
    """

    def __init__(self, problem: AllocationProblem, preferences: StakeholderPreferences):
        self.problem = problem
        self.preferences = preferences
        self.model = pulp.LpProblem("Fairness_Allocation", pulp.LpMaximize)
        self.allocation_vars = {}

    def setup_variables(self):
        data = self.problem.data 
        data = data[:20] # ONLY AFGHANISTAN, TO REMOVE

        for index, row in data.iterrows():
            for idx, metric in enumerate(["stunting", "wasting", "severe_wasting"]):
                # self.allocation_vars[f"{row.iloc[0]}_{row.iloc[2]}_{metric}"] = pulp.LpVariable(f"{row.iloc[0]}_{row.iloc[2]}_{metric}", lowBound=0, upBound=row.iloc[7 + idx] * row.iloc[10 + idx], cat='continuous')
                self.allocation_vars[(index, metric)] = pulp.LpVariable(f"{row.iloc[0]}_{row.iloc[2]}_{metric}", lowBound=0, upBound=row.iloc[7 + idx] * row.iloc[10 + idx], cat='continuous')

    def add_weighted_log_objective(self):
        """TODO: add weighted-log objective (or LP-compatible approximation)."""
        pass

    def add_max_min_fairness(self):
        """TODO: add max-min constraints/objective."""
        pass

    def add_proportional_fairness(self):
        """TODO: constrain allocation proportional to burden shares."""
        pass

    def add_demographic_constraints(self):
        """TODO: apply stakeholder constraints (e.g., rural_minimum)."""
        pass

    def solve(self) -> Dict:
        """Dispatch by fairness mode and return solution payload."""
        self.setup_variables()
        self.model += pulp.lpSum(self.allocation_vars.values()) <= self.problem.total_budget

        if self.preferences.fairness_mode == "utilitarian":
            self.model += pulp.lpSum([
                self.allocation_vars[(i, metric)] * 
                (1 / float(pd.to_numeric(self.problem.data.at[i, f"Cost_{metric}"], errors="coerce"))) *
                self.preferences.metric_weights[metric]
                for (i, metric) in self.allocation_vars
            ]), "Total_Treated_Children"
        elif self.preferences.fairness_mode == "weighted-log":
            self.add_weighted_log_objective()
        elif self.preferences.fairness_mode == "max-min":
            self.add_max_min_fairness()
        elif self.preferences.fairness_mode == "proportional":
            self.add_proportional_fairness()

        self.add_demographic_constraints()
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))
        return self._extract_solution()

    def _extract_solution(self) -> Dict:
        """TODO: convert solver output to dictionary/DataFrame payload."""
        pass


# ============================================================================
# 5. FAIRNESS METRICS
# ============================================================================


class FairnessMetrics:
    """Metrics and comparison helpers for baseline vs fairness allocations."""

    KEY_COLS = ["ISO3", "Country", "Demographic_Group"]

    @staticmethod
    def _allocation_df(allocation: Dict) -> pd.DataFrame:
        if not isinstance(allocation, dict):
            raise ValueError("allocation must be a dictionary result returned by an optimizer.")
        allocation_df = allocation.get("allocation_df")
        if allocation_df is None or not isinstance(allocation_df, pd.DataFrame):
            raise ValueError("allocation must contain an 'allocation_df' pandas DataFrame.")
        if allocation_df.empty:
            raise ValueError("allocation_df is empty.")
        return allocation_df.copy()

    @staticmethod
    def _base_problem_df(problem: AllocationProblem) -> pd.DataFrame:
        if not hasattr(problem, "filtered_df"):
            raise ValueError("problem must expose a filtered_df attribute.")
        base_df = problem.filtered_df.copy()
        if base_df.empty:
            raise ValueError("problem.filtered_df is empty.")
        return base_df

    @classmethod
    def _merged_allocation_view(cls, allocation: Dict, problem: AllocationProblem) -> pd.DataFrame:
        allocation_df = cls._allocation_df(allocation)
        base_df = cls._base_problem_df(problem)

        required_alloc_cols = set(cls.KEY_COLS + ["total_treated", "total_spend"])
        missing_alloc = required_alloc_cols - set(allocation_df.columns)
        if missing_alloc:
            raise ValueError(
                f"allocation_df is missing required columns: {sorted(missing_alloc)}"
            )

        working = base_df.copy()
        working["need_total"] = working.apply(
            lambda row: sum(float(pd.to_numeric(row[COUNT_COLS[t]], errors="coerce") or 0.0) for t in INTERVENTION_TYPES),
            axis=1,
        )

        merged = working.merge(
            allocation_df[cls.KEY_COLS + ["total_treated", "total_spend"]],
            on=cls.KEY_COLS,
            how="left",
            validate="one_to_one",
        )

        merged["total_treated"] = pd.to_numeric(merged["total_treated"], errors="coerce").fillna(0.0)
        merged["total_spend"] = pd.to_numeric(merged["total_spend"], errors="coerce").fillna(0.0)
        merged["coverage_ratio"] = merged.apply(
            lambda row: float(row["total_treated"]) / float(row["need_total"])
            if float(row["need_total"]) > 0
            else 0.0,
            axis=1,
        )
        return merged

    @staticmethod
    def _gini(values: List[float]) -> float:
        clean = [float(v) for v in values if pd.notna(v)]
        if not clean:
            return 0.0
        if any(v < 0 for v in clean):
            raise ValueError("Gini coefficient is undefined for negative values.")
        if all(v == 0 for v in clean):
            return 0.0

        sorted_vals = sorted(clean)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        weighted_sum = sum((idx + 1) * val for idx, val in enumerate(sorted_vals))
        return (2 * weighted_sum) / (n * total) - (n + 1) / n

    @staticmethod
    def total_lives_impacted(allocation: Dict, problem: AllocationProblem) -> float:
        """Efficiency metric: total treated children across all rows."""
        allocation_df = FairnessMetrics._allocation_df(allocation)
        if "total_treated" not in allocation_df.columns:
            raise ValueError("allocation_df must contain a 'total_treated' column.")
        return float(pd.to_numeric(allocation_df["total_treated"], errors="coerce").fillna(0.0).sum())

    @staticmethod
    def gini_coefficient(allocation: Dict, problem: AllocationProblem) -> float:
        """Inequality of achieved coverage across rows (0 = perfectly equal coverage)."""
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)
        return float(FairnessMetrics._gini(merged["coverage_ratio"].tolist()))

    @staticmethod
    def max_min_ratio(allocation: Dict, problem: AllocationProblem) -> float:
        """Best-off / worst-off achieved coverage ratio across rows with positive need."""
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)
        valid = merged[merged["need_total"] > 0].copy()
        if valid.empty:
            return 1.0

        min_cov = float(valid["coverage_ratio"].min())
        max_cov = float(valid["coverage_ratio"].max())

        if math.isclose(max_cov, 0.0, abs_tol=1e-12):
            return 1.0
        if math.isclose(min_cov, 0.0, abs_tol=1e-12):
            return float("inf")
        return max_cov / min_cov
    
    """ !!! non passato ne risultati, da valtare come chiamarlo se interessa   """
    @staticmethod
    def demographic_coverage_gap(
        allocation: Dict,
        problem: AllocationProblem,
        group_col: str = "Demographic_Group",
    ) -> Dict[str, Any]:
        """
        Per ogni paese, misura il gap di copertura tra gruppi demografici.

        Returns:
            {
            "per_country": {
                "AFG": {"max_gap": 0.32, "best_group": "urban", "worst_group": "rural"},
                ...
            },
            "weighted_mean_gap": 0.18,   # media pesata sul need totale
            "unweighted_mean_gap": 0.21,
            }
        """
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)

        if group_col not in merged.columns:
            raise ValueError(
                f"Column '{group_col}' not found in data. "
                f"Available columns: {list(merged.columns)}"
            )

        per_country = {}
        weighted_gap_sum = 0.0
        total_need_sum = 0.0

        for country, country_df in merged.groupby("ISO3"):
            if country_df.shape[0] < 2:
                # Un solo gruppo demografico: gap non definito
                continue

            # Escludiamo righe senza bisogno
            valid = country_df[country_df["need_total"] > 0].copy()
            if valid.shape[0] < 2:
                continue

            idx_min = valid["coverage_ratio"].idxmin()
            idx_max = valid["coverage_ratio"].idxmax()
            gap = float(valid.at[idx_max, "coverage_ratio"] - valid.at[idx_min, "coverage_ratio"])
            country_need = float(valid["need_total"].sum())

            per_country[country] = {
                "max_gap": gap,
                "best_group": valid.at[idx_max, group_col],
                "worst_group": valid.at[idx_min, group_col],
                "n_groups": int(valid.shape[0]),
            }

            weighted_gap_sum += gap * country_need
            total_need_sum += country_need

        if not per_country:
            return {
                "per_country": {},
                "weighted_mean_gap": 0.0,
                "unweighted_mean_gap": 0.0,
            }

        unweighted_mean = float(
            sum(v["max_gap"] for v in per_country.values()) / len(per_country)
        )
        weighted_mean = float(weighted_gap_sum / total_need_sum) if total_need_sum > 0 else 0.0

        return {
            "per_country": per_country,
            "weighted_mean_gap": weighted_mean,
            "unweighted_mean_gap": unweighted_mean,
        }

    """ !!! passato nei risultati, da valtare se interessa   """
    @staticmethod
    def demographic_parity_violation(
        allocation: Dict,
        problem: AllocationProblem,
        group_col: str = "Demographic_Group",
    ) -> float:
        """
        Misura quanto la distribuzione di copertura per gruppo demografico
        si discosta dalla distribuzione proporzionale al bisogno.

        Metrica: total variation distance tra distribuzione di copertura
        osservata e distribuzione ideale burden-proportional, aggregata
        su tutti i paesi e pesata sul need.

        0.0 -> copertura perfettamente proporzionale al bisogno per ogni gruppo
        1.0 -> massima deviazione possibile
        """
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)

        if group_col not in merged.columns:
            raise ValueError(
                f"Column '{group_col}' not found in data. "
                f"Available columns: {list(merged.columns)}"
            )

        violations = []
        weights = []

        for country, country_df in merged.groupby("ISO3"):
            valid = country_df[country_df["need_total"] > 0].copy()
            if valid.empty:
                continue

            country_need_total = float(valid["need_total"].sum())
            country_treated_total = float(valid["total_treated"].sum())

            if math.isclose(country_need_total, 0.0, abs_tol=1e-12):
                continue
            if math.isclose(country_treated_total, 0.0, abs_tol=1e-12):
                # Nessuno trattato: tutta la deviazione è massima
                violations.append(1.0)
                weights.append(country_need_total)
                continue

            # Distribuzione ideale: proporzionale al bisogno
            burden_share = valid["need_total"] / country_need_total

            # Distribuzione osservata: proporzionale ai trattati
            treated_share = valid["total_treated"] / country_treated_total

            # Total variation distance per questo paese
            tvd = float(0.5 * (treated_share - burden_share).abs().sum())
            violations.append(tvd)
            weights.append(country_need_total)

        if not violations:
            return 0.0

        total_weight = sum(weights)
        if math.isclose(total_weight, 0.0, abs_tol=1e-12):
            return 0.0

        return float(sum(v * w for v, w in zip(violations, weights)) / total_weight)

    @staticmethod
    def proportionality_violation(allocation: Dict, problem: AllocationProblem) -> float:
        """
        Distance between budget shares and burden shares.

        Returns the total variation distance:
            0   -> perfectly burden-proportional spending
            1   -> maximal deviation
        """
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)
        total_need = float(merged["need_total"].sum())
        total_spend = float(merged["total_spend"].sum())

        if math.isclose(total_need, 0.0, abs_tol=1e-12) or math.isclose(total_spend, 0.0, abs_tol=1e-12):
            return 0.0

        burden_share = merged["need_total"] / total_need
        spend_share = merged["total_spend"] / total_spend
        return float(0.5 * (spend_share - burden_share).abs().sum())

    @staticmethod
    def build_summary(allocation: Dict, problem: AllocationProblem, label: str = "solution") -> Dict[str, float]:
        """Return a compact metric summary for one allocation result."""
        return {
            "label": label,
            "status": allocation.get("status", "UNKNOWN"),
            "total_lives_impacted": FairnessMetrics.total_lives_impacted(allocation, problem),
            "total_spend": float(allocation.get("total_spend", 0.0) or 0.0),
            "demographic_parity_violation": FairnessMetrics.demographic_parity_violation(allocation, problem), """messo io assieme agli altri cosi"""
            "budget_utilisation_pct": float(allocation.get("budget_utilisation_pct", 0.0) or 0.0),
            "gini_coefficient": FairnessMetrics.gini_coefficient(allocation, problem),
            "max_min_ratio": FairnessMetrics.max_min_ratio(allocation, problem),
            "proportionality_violation": FairnessMetrics.proportionality_violation(allocation, problem),
        }

    @staticmethod
    def compare_allocations(
        baseline: Dict,
        fairness: Dict,
        problem: AllocationProblem,
        fairness_label: str = "fairness",
    ) -> Dict[str, Any]:
        """
        Compare baseline and fairness allocation outputs.

        Returns:
          - one summary dict per solution
          - a comparison DataFrame
          - price_of_fairness_pct
        """
        baseline_summary = FairnessMetrics.build_summary(baseline, problem, label="baseline")
        fairness_summary = FairnessMetrics.build_summary(fairness, problem, label=fairness_label)

        comparison_df = pd.DataFrame(
            {
                "metric": [
                    "total_lives_impacted",
                    "total_spend",
                    "budget_utilisation_pct",
                    "gini_coefficient",
                    "max_min_ratio",
                    "proportionality_violation",
                ],
                "baseline": [
                    baseline_summary["total_lives_impacted"],
                    baseline_summary["total_spend"],
                    baseline_summary["budget_utilisation_pct"],
                    baseline_summary["gini_coefficient"],
                    baseline_summary["max_min_ratio"],
                    baseline_summary["proportionality_violation"],
                ],
                fairness_label: [
                    fairness_summary["total_lives_impacted"],
                    fairness_summary["total_spend"],
                    fairness_summary["budget_utilisation_pct"],
                    fairness_summary["gini_coefficient"],
                    fairness_summary["max_min_ratio"],
                    fairness_summary["proportionality_violation"],
                ],
            }
        )
        comparison_df["absolute_delta"] = comparison_df[fairness_label] - comparison_df["baseline"]
        comparison_df["relative_delta_pct"] = comparison_df.apply(
            lambda row: 100.0 * row["absolute_delta"] / row["baseline"]
            if not math.isclose(float(row["baseline"]), 0.0, abs_tol=1e-12)
            else float("nan"),
            axis=1,
        )

        baseline_lives = float(baseline_summary["total_lives_impacted"])
        fairness_lives = float(fairness_summary["total_lives_impacted"])
        price_of_fairness_pct = (
            100.0 * (baseline_lives - fairness_lives) / baseline_lives
            if not math.isclose(baseline_lives, 0.0, abs_tol=1e-12)
            else 0.0
        )

        return {
            "baseline_summary": baseline_summary,
            "fairness_summary": fairness_summary,
            "comparison_df": comparison_df,
            "price_of_fairness_pct": price_of_fairness_pct,
        }


# ============================================================================
# 6. EVALUATION & VISUALIZATION
# ============================================================================


class ParetoBoundary:
    """Generate and plot efficiency-equity trade-off points."""

    def __init__(self, problem: AllocationProblem):
        self.problem = problem
        self.frontier = []

    def generate_solutions(
        self,
        fairness_modes: List[str],
        budget_fractions: List[float] = [0.5, 0.75, 1.0],
    ):
        """TODO: run batches and store (efficiency, equity, mode, budget_frac)."""
        pass

    def plot(self, filepath: Optional[str] = None):
        """TODO: scatter-plot Pareto points and optionally save figure."""
        pass


# ============================================================================
# 7. ORCHESTRATION
# ============================================================================


class AllocationEngine:
    """
    High-level orchestrator for baseline, fairness, and simulations.
    """

    def __init__(self, data_path: str, total_budget: float):
        self.data = pd.read_csv(data_path)
        self.problem = AllocationProblem(df=self.data, total_budget=total_budget)
        self.results = {}

    def run_baseline(self) -> Dict:
        """Run baseline utilitarian optimization."""
        optimizer = UtilitarianOptimizer(self.problem)
        baseline = optimizer.solve()
        self.results["baseline"] = baseline
        return baseline

    def run_fairness(self, preferences: StakeholderPreferences) -> Dict:
        """Run fairness-aware optimization for one preference profile."""
        optimizer = FairnessOptimizer(self.problem, preferences)
        fair_solution = optimizer.solve()
        self.results["fairness"] = fair_solution
        return fair_solution

    def aggregate_preferences(self, stakeholders: List[StakeholderProfile]) -> StakeholderPreferences:
        """TODO: aggregate multiple stakeholder preferences into one consensus profile."""
        pass

    def run_fairness_simulations(
        self,
        stakeholders: List[StakeholderProfile],
        include_consensus: bool = True,
    ) -> Dict[str, Dict]:
        """TODO: run one simulation per stakeholder (+ optional consensus)."""
        pass

    def compare_solutions(self) -> Dict[str, Any]:
        """Compare baseline vs fairness using efficiency/equity metrics."""
        baseline = self.results.get("baseline")
        fairness = self.results.get("fairness")

        if baseline is None:
            raise ValueError("Baseline result not found. Run run_baseline() first.")
        if fairness is None:
            raise ValueError("Fairness result not found. Run run_fairness() first.")

        comparison = FairnessMetrics.compare_allocations(
            baseline=baseline,
            fairness=fairness,
            problem=self.problem,
            fairness_label="fairness",
        )
        self.results["comparison"] = comparison
        return comparison

    def generate_pareto_frontier(self, filepath: Optional[str] = None):
        """Generate and plot Pareto front."""
        pareto = ParetoBoundary(self.problem)
        pareto.generate_solutions(["utilitarian", "proportional", "weighted-log", "max-min"])
        pareto.plot(filepath=filepath)
