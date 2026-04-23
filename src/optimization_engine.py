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
    "stunting":       "Count_stunting",
    "wasting":        "Count_wasting",
    "severe_wasting": "Count_severe_wasting",
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
            df = df[df["Demographic_group"].isin(self.demographic_filter)]
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
            demographic_group = row["Demographic_group"]
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
            demographic_group = row["Demographic_group"]
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
            demographic_group = base_row["Demographic_group"]
            record = {"ISO3": base_row["ISO3"], "Country": base_row["Country"], "Demographic_group": base_row["Demographic_group"]}
            
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
        
        for demographic in self.problem.filtered_df["Demographic_group"].unique():
            demographic_vars = [key for key in self.allocation_vars if key[1] == demographic]
            
            if demographic_vars:
                constraint_expr = pulp.lpSum(
                    self.allocation_vars[key] * self.cost_map[key]
                    for key in demographic_vars
                )
                self.model += constraint_expr <= self.demographic_cap * self.problem.total_budget, f"Demographic_Cap_{demographic}"

    def add_demographic_minimum_constraints(self):
        """Enforce per-demographic minimum: sum(spend by demographic group) >= min_share * total_budget.
        
        Supports pattern matching: constraint keys can be:
        - Exact group name: "Wealth Quintile 1 Rural" (matches only that group)
        - Substring pattern: "Rural" (matches all groups containing "Rural")
        - Wealth quintile: "Wealth Quintile 1" (matches all Q1 groups: Q1 Rural + Q1 Urban)
        """
        for demographic_pattern, min_share in self.demographic_min_share.items():
            # Find all allocation variables matching this pattern
            demographic_vars = [
                key for key in self.allocation_vars 
                if demographic_pattern in key[1]  # key[1] is the demographic_group
            ]
            
            if demographic_vars:
                constraint_expr = pulp.lpSum(
                    self.allocation_vars[key] * self.cost_map[key]
                    for key in demographic_vars
                )
                self.model += constraint_expr >= min_share * self.problem.total_budget, f"Demographic_Min_{demographic_pattern}"
            else:
                print(f"⚠️ Warning: No demographic groups found matching pattern '{demographic_pattern}'")
                print(f"   Available groups: {sorted(set(key[1] for key in self.allocation_vars))}")


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

    # ALLOWED_FAIRNESS_MODES = {"utilitarian", "weighted-log", "max-min", "proportional"}
    ALLOWED_FAIRNESS_MODES = {"utilitarian", "max-min", "proportional"}

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
      - max-min fairness
      - proportional fairness
    """

    def __init__(self, problem: AllocationProblem, preferences: StakeholderPreferences):
        self.problem = problem
        self.preferences = preferences
        self.model = pulp.LpProblem("Fairness_Allocation", pulp.LpMaximize)
        self.allocation_vars = {}

    def setup_variables(self):
        data = self.problem.df
        # data = data[:20] # TODO: ONLY AFGHANISTAN, TO REMOVE

        for index, row in data.iterrows():
            for idx, metric in enumerate(INTERVENTION_TYPES):
                # self.allocation_vars[(index, row['ISO3'], row['Demographic_Group'], metric)] = pulp.LpVariable(f"{row['ISO3']}_{row.at['Demographic_Group']}_{metric}", lowBound=0, upBound=row.iloc[7 + idx] * row.iloc[10 + idx], cat='continuous')
                self.allocation_vars[(index, row['ISO3'], row['Demographic_Group'], metric)] = pulp.LpVariable(f"{row['ISO3']}_{row.at['Demographic_Group']}_{metric}", lowBound=0, upBound=row.at[f"Count_{metric}"], cat='continuous')

    def add_max_min_fairness(self):
        min_coverage = pulp.LpVariable("min_coverage", lowBound=0, cat=pulp.LpContinuous)
        self.model += min_coverage, "Maximize_Min_Coverage"

        for iso3, group in self.problem.df.groupby("ISO3"):
            burden = sum([ 
                float(pd.to_numeric(group.iloc[0]["Population_U5"], errors="coerce")) * self.preferences.metric_weights[metric] 
                for metric in INTERVENTION_TYPES 
            ])

            # ​coverage_g = ∑i∈{g,m}​​ (x{i,m}​​​/c{i,m})

            national_rows = group[group["Demographic_Group"].str.lower() == "national"]
            if national_rows.empty: continue
            national_idx = national_rows.index[0]

            self.model += pulp.lpSum([ 
                self.allocation_vars[(national_idx, iso3, "National", metric)] # *
                # (1 / float(pd.to_numeric(group["Demographic_Group" == "National"][f"Count_{metric}"])))
                for metric in INTERVENTION_TYPES
            ]) >= min_coverage * burden # coverage_g >= z * burden_g 

    def add_proportional_fairness(self):
        # Keep utilitarian-style objective.
        self.model += pulp.lpSum(
            self.allocation_vars[(idx, iso3, demo, metric)] * self.preferences.metric_weights[metric]
            for (idx, iso3, demo, metric) in self.allocation_vars
        ), "Total_Treated_Children"
    
        # Build normalized country burden shares.
        countries_burden = {}
        for iso3, group in self.problem.df.groupby("ISO3"):
            burden = sum(
                float(pd.to_numeric(group.iloc[0]["Population_U5"], errors="coerce") or 0.0)
                * float(self.preferences.metric_weights[metric])
                for metric in INTERVENTION_TYPES
            )
            countries_burden[iso3] = burden
    
        total_burden = sum(countries_burden.values())
        if total_burden <= 0:
            return
    
        for iso3 in list(countries_burden.keys()):
            countries_burden[iso3] = countries_burden[iso3] / total_burden
    
        # Country-specific minimum spend share.
        for iso3, burden_share in countries_burden.items():
            country_spend_expr = pulp.lpSum(
                self.allocation_vars[(idx, var_iso3, demo, metric)] *
                float(pd.to_numeric(self.problem.df.at[idx, COST_COLS[metric]], errors="coerce") or 0.0)
                for (idx, var_iso3, demo, metric) in self.allocation_vars
                if var_iso3 == iso3
            )
            self.model += country_spend_expr >= burden_share * self.problem.total_budget, f"Proportional_fairness_{iso3}"

    def add_demographic_constraints(self):
        '''
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
        '''

        for constraint, value in self.preferences.demographic_constraints.items():
            parts = constraint.split("_")
            if len(parts) < 3:
                continue

            group_value = "_".join(parts[:-2]).lower()
            bound_type = parts[-2].lower()
            quantity = parts[-1].lower()

            if quantity != "share" or bound_type not in {"min", "max"}:
                continue

            constrained_keys = [
                key for key in self.allocation_vars
                if str(key[2]).lower() == group_value
            ]

            # If no matching demographic rows exist, skip instead of creating impossible constraints.
            if not constrained_keys:
                continue

            spend_expr = pulp.lpSum(
                self.allocation_vars[(idx, iso3, demo, metric)] *
                float(pd.to_numeric(self.problem.df.at[idx, COST_COLS[metric]], errors="coerce") or 0.0)
                for (idx, iso3, demo, metric) in constrained_keys
            )

            if bound_type == "min":
                self.model += spend_expr >= float(value) * self.problem.total_budget
            else:
                self.model += spend_expr <= float(value) * self.problem.total_budget

    def solve(self) -> Dict:
        """Dispatch by fairness mode and return solution payload."""
        self.setup_variables()

        self.model += pulp.lpSum([
            self.allocation_vars[(index, iso3, group, metric)] * 
            float(pd.to_numeric(self.problem.df.at[index, f"Cost_{metric}"], errors="coerce"))
            for (index, iso3, group, metric) in self.allocation_vars
        ]) <= self.problem.total_budget, "Max_Budget"

        if self.preferences.fairness_mode == "utilitarian":
            self.model += pulp.lpSum([
                self.allocation_vars[(index, iso3, group, metric)] * 
                self.preferences.metric_weights[metric]
                for (index, iso3, group, metric) in self.allocation_vars
            ]), "Total_Treated_Children"
        elif self.preferences.fairness_mode == "max-min":
            self.add_max_min_fairness()
        elif self.preferences.fairness_mode == "proportional":
            self.add_proportional_fairness()

        self.add_demographic_constraints()
        self.model.solve(pulp.HiGHS(msg=False))
        return self._extract_solution()

    def _extract_solution(self) -> Dict:
        """
        Convert solver output to:
          - aggregate summary
          - per (ISO3, Demographic_Group) allocation DataFrame
        """
        df = self.problem.filtered_df.copy()
    
        # Build quick lookup for Country and costs by (ISO3, Demographic_Group, metric)
        country_lookup = (
            df[["ISO3", "Demographic_Group", "Country"]]
            .drop_duplicates()
            .set_index(["ISO3", "Demographic_Group"])["Country"]
            .to_dict()
        )
    
        cost_lookup = {}
        for row in df.to_dict("records"):
            iso3 = row["ISO3"]
            demo = row["Demographic_Group"]
            for metric in INTERVENTION_TYPES:
                cost_lookup[(iso3, demo, metric)] = float(
                    pd.to_numeric(row[COST_COLS[metric]], errors="coerce")
                )
    
        # Aggregate variable values into row-level output
        records = {}
        for (_, iso3, demo, metric), var in self.allocation_vars.items():
            treated = float(pulp.value(var) or 0.0)
            cost = cost_lookup.get((iso3, demo, metric), 0.0)
            spend = treated * cost
    
            key = (iso3, demo)
            if key not in records:
                records[key] = {
                    "ISO3": iso3,
                    "Country": country_lookup.get((iso3, demo), None),
                    "Demographic_Group": demo,
                    "total_treated": 0.0,
                    "total_spend": 0.0,
                }
                for t in INTERVENTION_TYPES:
                    records[key][f"treated_{t}"] = 0.0
                    records[key][f"spend_{t}"] = 0.0
    
            records[key][f"treated_{metric}"] += treated
            records[key][f"spend_{metric}"] += spend
            records[key]["total_treated"] += treated
            records[key]["total_spend"] += spend
    
        allocation_df = pd.DataFrame(list(records.values()))
        if allocation_df.empty:
            allocation_df = pd.DataFrame(
                columns=[
                    "ISO3", "Country", "Demographic_Group",
                    "treated_stunting", "treated_wasting", "treated_severe_wasting",
                    "spend_stunting", "spend_wasting", "spend_severe_wasting",
                    "total_treated", "total_spend",
                ]
            )
    
        total_spend = float(allocation_df["total_spend"].sum()) if not allocation_df.empty else 0.0

        return {
            "status": pulp.LpStatus[self.model.status],
            "objective_value": float(pulp.value(self.model.objective) or 0.0),
            "total_treated": float(allocation_df["total_treated"].sum()) if not allocation_df.empty else 0.0,
            "total_spend": total_spend,
            "budget": float(self.problem.total_budget),
            "budget_utilisation_pct": (
                100.0 * total_spend / float(self.problem.total_budget)
                if float(self.problem.total_budget) > 0 else 0.0
            ),
            "allocation_df": allocation_df,
        }


# ============================================================================
# 5. FAIRNESS METRICS
# ============================================================================


class FairnessMetrics:
    """Metrics and comparison helpers for baseline vs fairness allocations."""

    KEY_COLS = ["ISO3", "Country", "Demographic_Group"]

    # Validate the optimizer output and return its allocation DataFrame.
    # Input: allocation result dict.
    # Output: a non-empty copy of allocation_df.
    # Purpose: ensure downstream metrics always work on a valid allocation table.
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

    # Validate the original problem data and return the filtered base DataFrame.
    # Input: AllocationProblem instance.
    # Output: a non-empty copy of problem.filtered_df.
    # Purpose: provide the reference data needed to evaluate an allocation.
    @staticmethod
    def _base_problem_df(problem: AllocationProblem) -> pd.DataFrame:
        if not hasattr(problem, "filtered_df"):
            raise ValueError("problem must expose a filtered_df attribute.")
        base_df = problem.filtered_df.copy()
        if base_df.empty:
            raise ValueError("problem.filtered_df is empty.")
        return base_df

    # Build the main evaluation view by merging base data with allocation results.
    # Input: optimizer result dict + AllocationProblem.
    # Output: merged DataFrame with need, treated, spend, and coverage ratio.
    # Purpose: create the common table used by most fairness metrics.
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

    # Compute the Gini coefficient of a list of non-negative values.
    # Input: numeric list, typically coverage ratios.
    # Output: inequality score from 0 (equal) upward.
    # Purpose: summarize how unevenly benefits are distributed.
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

    # Measure total efficiency as the number of children treated overall.
    # Input: allocation result dict.
    # Output: total treated across all rows.
    # Purpose: quantify the aggregate impact of the solution.
    @staticmethod
    def total_lives_impacted(allocation: Dict, problem: AllocationProblem) -> float:
        """Efficiency metric: total treated children across all rows."""
        allocation_df = FairnessMetrics._allocation_df(allocation)
        if "total_treated" not in allocation_df.columns:
            raise ValueError("allocation_df must contain a 'total_treated' column.")
        return float(pd.to_numeric(allocation_df["total_treated"], errors="coerce").fillna(0.0).sum())

    # Measure inequality in achieved coverage across rows.
    # Input: allocation result dict + AllocationProblem.
    # Output: Gini coefficient on coverage ratios.
    # Purpose: check whether coverage is distributed evenly or not.
    @staticmethod
    def gini_coefficient(allocation: Dict, problem: AllocationProblem) -> float:
        """Inequality of achieved coverage across rows (0 = perfectly equal coverage)."""
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)
        return float(FairnessMetrics._gini(merged["coverage_ratio"].tolist()))

    # Compare the best-served and worst-served groups in terms of coverage.
    # Input: allocation result dict + AllocationProblem.
    # Output: max coverage divided by min coverage.
    # Purpose: capture worst-case disparity in a simple fairness indicator.
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
    
    # Measure within-country coverage gaps between demographic groups.
    # Input: allocation result dict + AllocationProblem.
    # Output: per-country gap details and mean gap summaries.
    # Purpose: detect whether some demographic groups are systematically underserved.
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

    # Compare observed demographic coverage with the ideal burden-based distribution.
    # Input: allocation result dict + AllocationProblem.
    # Output: violation score between 0 and 1.
    # Purpose: measure how far treatment distribution departs from demographic need.
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

    # Compare budget shares with need shares across the allocation.
    # Input: allocation result dict + AllocationProblem.
    # Output: total variation distance between spend and burden distributions.
    # Purpose: assess whether money follows need proportionally.
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

    # Build a compact summary of the main efficiency and fairness metrics.
    # Input: allocation result dict + AllocationProblem.
    # Output: dictionary of key evaluation indicators for one solution.
    # Purpose: make a single allocation easy to inspect and report.
    @staticmethod
    def build_summary(allocation: Dict, problem: AllocationProblem, label: str = "solution") -> Dict[str, float]:
        """Return a compact metric summary for one allocation result."""
        return {
            "label": label,
            "status": allocation.get("status", "UNKNOWN"),
            "total_lives_impacted": FairnessMetrics.total_lives_impacted(allocation, problem),
            "total_spend": float(allocation.get("total_spend", 0.0) or 0.0),
            "budget_utilisation_pct": float(allocation.get("budget_utilisation_pct", 0.0) or 0.0),
            "gini_coefficient": FairnessMetrics.gini_coefficient(allocation, problem),
            "max_min_ratio": FairnessMetrics.max_min_ratio(allocation, problem),
            "demographic_parity_violation": FairnessMetrics.demographic_parity_violation(allocation, problem),
            "proportionality_violation": FairnessMetrics.proportionality_violation(allocation, problem),
        }

    # Compare baseline and fairness-aware solutions side by side.
    # Input: two allocation result dicts + AllocationProblem.
    # Output: summaries, comparison table, and price of fairness.
    # Purpose: quantify the trade-off between efficiency and equity.
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
                    # "budget_utilisation_pct",
                    "gini_coefficient",
                    "max_min_ratio",
                    "proportionality_violation",
                ],
                "baseline": [
                    baseline_summary["total_lives_impacted"],
                    baseline_summary["total_spend"],
                    # baseline_summary["budget_utilisation_pct"],
                    baseline_summary["gini_coefficient"],
                    baseline_summary["max_min_ratio"],
                    baseline_summary["proportionality_violation"],
                ],
                fairness_label: [
                    fairness_summary["total_lives_impacted"],
                    fairness_summary["total_spend"],
                    # fairness_summary["budget_utilisation_pct"],
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
        """
        Run one fairness optimization per stakeholder profile and optionally one
        additional consensus profile.
    
        Returns:
            {
                "<stakeholder_name>": <optimizer solution dict>,
                "consensus": <optimizer solution dict>  # optional
            }
        """
        if not stakeholders:
            raise ValueError("stakeholders must contain at least one StakeholderProfile.")
    
        simulations: Dict[str, Dict] = {}
        used_names = set()
    
        for i, stakeholder in enumerate(stakeholders, start=1):
            name = (stakeholder.name or "").strip() or f"stakeholder_{i}"
            if name in used_names:
                name = f"{name}_{i}"
            used_names.add(name)
    
            optimizer = FairnessOptimizer(self.problem, stakeholder.preferences)
            result = optimizer.solve()
            result["stakeholder_name"] = name
            result["influence"] = float(stakeholder.influence)
            simulations[name] = result
    
        if include_consensus:
            consensus_preferences = self.aggregate_preferences(stakeholders)
            if consensus_preferences is not None:
                consensus_optimizer = FairnessOptimizer(self.problem, consensus_preferences)
                consensus_result = consensus_optimizer.solve()
                consensus_result["stakeholder_name"] = "consensus"
                consensus_result["influence"] = float(sum(s.influence for s in stakeholders))
                simulations["consensus"] = consensus_result
    
        self.results["simulations"] = simulations
        return simulations
    
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
