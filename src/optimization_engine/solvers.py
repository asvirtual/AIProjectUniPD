# ============================================================================
# OPTIMIZATION SOLVERS
# ============================================================================

from typing import Dict, Optional

import pandas as pd
import pulp

from .models import (
    AllocationProblem,
    StakeholderPreferences,
    INTERVENTION_TYPES,
    COUNT_COLS,
    COST_COLS,
    COUNT_SCALE_FACTOR,
    COST_SCALE_FACTOR,
    BUDGET_SCALE_FACTOR,
)


def _solve_with_fallback(
    model: pulp.LpProblem,
    primary_solver,
    fallback_solver=None,
) -> None:
    """Solve a PuLP model with a deterministic fallback solver.

    This shields the optimization flow from environment-specific solver issues
    (for example HiGHS_CMD output parsing errors on some local installs).
    """
    solvers_to_try = [primary_solver]
    if fallback_solver is not None:
        solvers_to_try.append(fallback_solver)

    last_error = None

    for solver in solvers_to_try:
        available_method = getattr(solver, "available", None)
        if callable(available_method):
            try:
                if not solver.available():
                    continue
            except Exception:
                # If availability probing fails, try solving anyway.
                pass

        try:
            model.solve(solver)
            return
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    model.solve()


#=======================================================================
# Standard Utilitarian Optimizer
#=======================================================================

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
        
        _solve_with_fallback(
            self.model,
            primary_solver=pulp.HiGHS_CMD(msg=False),
            fallback_solver=pulp.PULP_CBC_CMD(msg=False),
        )

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
                # Unscale treated counts to original scale
                record[f"treated_{itype}"] = treated * COUNT_SCALE_FACTOR
                # Unscale spend: spend(scaled) * COUNT_SCALE_FACTOR * COST_SCALE_FACTOR
                record[f"spend_{itype}"] = treated * self.cost_map[key] * COUNT_SCALE_FACTOR * COST_SCALE_FACTOR
            
            record["total_treated"] = sum(record[f"treated_{t}"] for t in INTERVENTION_TYPES)
            record["total_spend"] = sum(record[f"spend_{t}"] for t in INTERVENTION_TYPES)
            rows.append(record)

        allocation_df = pd.DataFrame(rows)
        total_spend = allocation_df["total_spend"].sum()
        # Unscale budget to original scale
        original_budget = self.problem.total_budget * BUDGET_SCALE_FACTOR

        return {
            "status": pulp.LpStatus[self.model.status],
            "total_treated": (pulp.value(self.model.objective) or 0.0) * COUNT_SCALE_FACTOR,
            "total_spend": total_spend,
            "budget": original_budget,
            "budget_utilisation_pct": 100 * total_spend / original_budget if original_budget > 0 else 0,
            "allocation_df": allocation_df,
        }


#=======================================================================
# Constrained Utilitarian Optimizer with Equity Constraints
#=======================================================================

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
        
        _solve_with_fallback(
            self.model,
            primary_solver=pulp.HiGHS_CMD(msg=False),
            fallback_solver=pulp.PULP_CBC_CMD(msg=False),
        )
        
        return self._extract_solution()

# ========================================================================
# Fairness-Aware Optimizer (multi-objective optimization with max-min or proportional fairness)
# ========================================================================

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
        self.cost_map = {}

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
        """Create one LpVariable per (iso3, demographic_group, metric).
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

    def add_max_min_fairness(self):
        """Maximize the minimum treated-share across all filtered rows.

        Each row represents a country/demographic segment. We constrain the
        treated total in every row to be at least min_coverage times that row's
        total need. This is the most stable max-min formulation for this data
        shape and avoids depending on special labels like 'National'.
        """
        min_coverage = pulp.LpVariable("min_coverage", lowBound=0, upBound=1, cat=pulp.LpContinuous)

        working_df = self.problem.filtered_df
        added_constraint = False

        for iso3, demo in working_df[["ISO3", "Demographic_group"]].drop_duplicates().values:
            row = working_df[(working_df["ISO3"] == iso3) & 
                             (working_df["Demographic_group"] == demo)].iloc[0]
            row_need = sum(
                float(pd.to_numeric(row[COUNT_COLS[metric]], errors="coerce") or 0.0)
                for metric in INTERVENTION_TYPES
            )

            if row_need <= 0:
                continue

            row_treated_expr = pulp.lpSum(
                self.allocation_vars[(iso3, demo, metric)]
                for metric in INTERVENTION_TYPES
            )

            self.model += row_treated_expr >= min_coverage * row_need, f"Min_Coverage_{iso3}_{demo}"
            added_constraint = True

        if not added_constraint:
            raise ValueError("No valid positive-need rows found for max-min fairness optimization.")

        self.model.setObjective(min_coverage)

    def add_proportional_fairness(self):
        # Keep utilitarian-style objective.
        self.model += pulp.lpSum(
            self.allocation_vars[(iso3, demo, metric)] * self.preferences.metric_weights[metric]
            for (iso3, demo, metric) in self.allocation_vars
        ), "Total_Treated_Children"
    
        # Build normalized country burden shares.
        countries_burden = {}
        for iso3, group in self.problem.filtered_df.groupby("ISO3"):
            burden = sum(
                float(pd.to_numeric(group.iloc[0]["Population_u5"], errors="coerce") or 0.0)
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
                self.allocation_vars[(var_iso3, demo, metric)] * self.cost_map[(var_iso3, demo, metric)]
                for (var_iso3, demo, metric) in self.allocation_vars
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
                if group_value in str(key[1]).lower()
            ]

            # If no matching demographic rows exist, skip instead of creating impossible constraints.
            if not constrained_keys:
                continue

            spend_expr = pulp.lpSum(
                self.allocation_vars[(iso3, demo, metric)] * self.cost_map[(iso3, demo, metric)]
                for (iso3, demo, metric) in constrained_keys
            )

            if bound_type == "min":
                self.model += spend_expr >= float(value) * self.problem.total_budget
            else:
                self.model += spend_expr <= float(value) * self.problem.total_budget

    def add_coverage_floor(self):
        """Apply a per-stratum minimum treated-share floor when requested.

        Preference key:
            min_coverage_share (float in [0, 1])

        Semantics:
            For each (ISO3, Demographic_group) with positive need,
            treated_in_stratum >= min_coverage_share * need_in_stratum.
        """
        min_coverage_share = self.preferences.min_coverage_share
        if min_coverage_share is None:
            return

        min_coverage_share = float(min_coverage_share)
        if min_coverage_share <= 0:
            return

        working_df = self.problem.filtered_df
        for iso3, demo in working_df[["ISO3", "Demographic_group"]].drop_duplicates().values:
            row = working_df[
                (working_df["ISO3"] == iso3) & (working_df["Demographic_group"] == demo)
            ].iloc[0]
            row_need = sum(
                float(pd.to_numeric(row[COUNT_COLS[metric]], errors="coerce") or 0.0)
                for metric in INTERVENTION_TYPES
            )

            if row_need <= 0:
                continue

            row_treated_expr = pulp.lpSum(
                self.allocation_vars[(iso3, demo, metric)]
                for metric in INTERVENTION_TYPES
            )
            self.model += (
                row_treated_expr >= min_coverage_share * row_need,
                f"Coverage_Floor_{iso3}_{demo}",
            )

    def solve(self) -> Dict:
        """Dispatch by fairness mode and return solution payload."""
        self.setup_variables()
        self._build_cost_map()  # Pre-compute costs before building constraints

        self.model += pulp.lpSum([
            self.allocation_vars[(iso3, group, metric)] * self.cost_map[(iso3, group, metric)]
            for (iso3, group, metric) in self.allocation_vars
        ]) <= self.problem.total_budget, "Max_Budget"

        if self.preferences.fairness_mode == "utilitarian":
            self.model += pulp.lpSum([
                self.allocation_vars[(iso3, group, metric)] * self.preferences.metric_weights[metric]
                for (iso3, group, metric) in self.allocation_vars
            ]), "Total_Treated_Children"
        elif self.preferences.fairness_mode == "max-min":
            self.add_max_min_fairness()
        elif self.preferences.fairness_mode == "proportional":
            self.add_proportional_fairness()

        self.add_demographic_constraints()
        self.add_coverage_floor()
        solver = (
            pulp.PULP_CBC_CMD(msg=False)
            if self.preferences.fairness_mode == "max-min"
            else pulp.HiGHS_CMD(msg=False)
        )
        fallback_solver = (
            pulp.HiGHS_CMD(msg=False)
            if self.preferences.fairness_mode == "max-min"
            else pulp.PULP_CBC_CMD(msg=False)
        )
        _solve_with_fallback(self.model, primary_solver=solver, fallback_solver=fallback_solver)
        return self._extract_solution()

    def _extract_solution(self) -> Dict:
        """
        Convert solver output to:
          - aggregate summary
          - per (ISO3, Demographic_group) allocation DataFrame
        """
        df = self.problem.filtered_df.copy()
    
        # Build quick lookup for Country by (ISO3, Demographic_group)
        country_lookup = (
            df[["ISO3", "Demographic_group", "Country"]]
            .drop_duplicates()
            .set_index(["ISO3", "Demographic_group"])["Country"]
            .to_dict()
        )
    
        # Aggregate variable values into row-level output
        records = {}
        for (iso3, demo, metric), var in self.allocation_vars.items():
            treated = float(pulp.value(var) or 0.0)
            # Unscale treated counts to original scale
            treated_unscaled = treated * COUNT_SCALE_FACTOR
            cost = self.cost_map.get((iso3, demo, metric), 0.0)
            # Unscale spend: spend(scaled) * COUNT_SCALE_FACTOR * COST_SCALE_FACTOR
            spend = treated * cost * COUNT_SCALE_FACTOR * COST_SCALE_FACTOR
    
            key = (iso3, demo)
            if key not in records:
                records[key] = {
                    "ISO3": iso3,
                    "Country": country_lookup.get((iso3, demo), None),
                    "Demographic_group": demo,
                    "total_treated": 0.0,
                    "total_spend": 0.0,
                }
                for t in INTERVENTION_TYPES:
                    records[key][f"treated_{t}"] = 0.0
                    records[key][f"spend_{t}"] = 0.0
    
            records[key][f"treated_{metric}"] += treated_unscaled
            records[key][f"spend_{metric}"] += spend
            records[key]["total_treated"] += treated_unscaled
            records[key]["total_spend"] += spend
    
        allocation_df = pd.DataFrame(list(records.values()))
        if allocation_df.empty:
            allocation_df = pd.DataFrame(
                columns=[
                    "ISO3", "Country", "Demographic_group",
                    "treated_stunting", "treated_wasting", "treated_severe_wasting",
                    "spend_stunting", "spend_wasting", "spend_severe_wasting",
                    "total_treated", "total_spend",
                ]
            )
    
        total_spend = float(allocation_df["total_spend"].sum()) if not allocation_df.empty else 0.0
        # Unscale budget to original scale
        original_budget = float(self.problem.total_budget) * BUDGET_SCALE_FACTOR

        return {
            "status": pulp.LpStatus[self.model.status],
            "objective_value": float(pulp.value(self.model.objective) or 0.0),
            "total_treated": float(allocation_df["total_treated"].sum()) if not allocation_df.empty else 0.0,
            "total_spend": total_spend,
            "budget": original_budget,
            "budget_utilisation_pct": (
                100.0 * total_spend / original_budget
                if original_budget > 0 else 0.0
            ),
            "allocation_df": allocation_df,
        }

# ========================================================================
# Lexicographic Consensus Optimizer
# ========================================================================

class LexicographicConsensusOptimizer:
    """
    Two-stage lexicographic optimizer for multi-stakeholder consensus.

    Stage 1 — efficiency ceiling:
        Run a plain UtilitarianOptimizer to find E*, the maximum number of
        children treatable within budget.

    Stage 2 — fairness within tolerance:
        Run FairnessOptimizer with the blended consensus preferences, but add
        a hard lower bound: total_treated >= (1 - tolerance) * E*.

    This guarantees the consensus solution is never worse than
    (1 - tolerance) of the utilitarian ceiling, regardless of how
    conflicting the blended preferences are.

    Args:
        problem     : AllocationProblem instance.
        preferences : StakeholderPreferences produced by aggregate_preferences.
        efficiency_tolerance: maximum acceptable efficiency loss as a fraction
            of E*. Default 0.05 = allow up to 5% fewer children treated in
            exchange for improved fairness.
    """

    def __init__(
        self,
        problem: AllocationProblem,
        preferences: StakeholderPreferences,
        efficiency_tolerance: float = 0.05,
    ):
        if not (0.0 <= efficiency_tolerance < 1.0):
            raise ValueError("efficiency_tolerance must be in [0, 1).")
        self.problem = problem
        self.preferences = preferences
        self.efficiency_tolerance = efficiency_tolerance

    def _build_stage2(
        self,
        efficiency_floor: float,
        include_demographic_constraints: bool,
    ) -> "FairnessOptimizer":
        """
        Construct and return a configured FairnessOptimizer for stage 2.
        Kept separate so we can retry without demographic constraints on infeasibility.
        """
        fairness_opt = FairnessOptimizer(self.problem, self.preferences)
        fairness_opt.setup_variables()
        fairness_opt._build_cost_map()

        # Budget constraint
        fairness_opt.model += pulp.lpSum([
            fairness_opt.allocation_vars[k] * fairness_opt.cost_map[k]
            for k in fairness_opt.allocation_vars
        ]) <= fairness_opt.problem.total_budget, "Max_Budget"

        # Efficiency floor — scale back to solver units
        floor_scaled = efficiency_floor / COUNT_SCALE_FACTOR
        fairness_opt.model += pulp.lpSum([
            fairness_opt.allocation_vars[k]
            for k in fairness_opt.allocation_vars
        ]) >= floor_scaled, "Efficiency_Floor"

        # Fairness objective
        if self.preferences.fairness_mode == "utilitarian":
            fairness_opt.model += pulp.lpSum([
                fairness_opt.allocation_vars[(iso3, group, metric)]
                * self.preferences.metric_weights[metric]
                for (iso3, group, metric) in fairness_opt.allocation_vars
            ]), "Weighted_Treated"
        elif self.preferences.fairness_mode == "max-min":
            fairness_opt.add_max_min_fairness()
        elif self.preferences.fairness_mode == "proportional":
            fairness_opt.add_proportional_fairness()

        fairness_opt.add_coverage_floor()

        if include_demographic_constraints:
            fairness_opt.add_demographic_constraints()

        return fairness_opt
    
    def solve(self) -> Dict:
        """
        Run both stages and return the stage-2 solution with ceiling metadata.

        Fallback strategy: if stage 2 is infeasible with demographic constraints
        (e.g. rural_min_share conflicts with the efficiency floor at new costs),
        retry without them and attach a warning to the result so callers know
        which constraints were relaxed.
        """
        # ------------------------------------------------------------------
        # Stage 1: utilitarian ceiling
        # ------------------------------------------------------------------
        stage1 = UtilitarianOptimizer(self.problem).solve()
        efficiency_ceiling = float(stage1.get("total_treated", 0.0))
        efficiency_floor   = efficiency_ceiling * (1.0 - self.efficiency_tolerance)

        solver = (
            pulp.PULP_CBC_CMD(msg=False)
            if self.preferences.fairness_mode == "max-min"
            else pulp.HiGHS_CMD(msg=False)
        )
        fallback_solver = (
            pulp.HiGHS_CMD(msg=False)
            if self.preferences.fairness_mode == "max-min"
            else pulp.PULP_CBC_CMD(msg=False)
        )

        # ------------------------------------------------------------------
        # Stage 2a: with demographic constraints
        # ------------------------------------------------------------------
        fairness_opt = self._build_stage2(efficiency_floor, include_demographic_constraints=True)
        _solve_with_fallback(
            fairness_opt.model,
            primary_solver=solver,
            fallback_solver=fallback_solver,
        )
        demographic_constraints_applied = True

        # ------------------------------------------------------------------
        # Stage 2b: fallback — relax demographic constraints if infeasible
        # ------------------------------------------------------------------
        if pulp.LpStatus[fairness_opt.model.status] != "Optimal":
            print(
                f"[WARN] Consensus infeasible with demographic constraints "
                f"(mode={self.preferences.fairness_mode}, "
                f"tolerance={self.efficiency_tolerance}). "
                f"Retrying without demographic constraints."
            )
            fairness_opt = self._build_stage2(efficiency_floor, include_demographic_constraints=False)
            _solve_with_fallback(
                fairness_opt.model,
                primary_solver=solver,
                fallback_solver=fallback_solver,
            )
            demographic_constraints_applied = False

        result = fairness_opt._extract_solution()
        result["efficiency_ceiling"]              = efficiency_ceiling
        result["efficiency_floor"]                = efficiency_floor
        result["efficiency_tolerance"]            = self.efficiency_tolerance
        result["demographic_constraints_applied"] = demographic_constraints_applied
        return result