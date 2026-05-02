# ============================================================================
# Allocation Engine: integrates data processing, preference elicitation, optimization, and evaluation
# ============================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import AllocationProblem, StakeholderPreferences, StakeholderProfile
from .solvers import ConstrainedUtilitarianOptimizer, UtilitarianOptimizer, FairnessOptimizer, LexicographicConsensusOptimizer
from .metrics import FairnessMetrics
from .visualization import ParetoBoundary

# ============================================================================
# Main engine class that orchestrates the entire optimization workflow
# ============================================================================

class AllocationEngine:
    """
    High-level orchestrator for baseline, fairness, and simulations.
    """

    def __init__(self, data_path: str, total_budget: float):
        # Convert relative path to absolute path based on script location
        file_path = Path(data_path)
        if not file_path.is_absolute():
            # Remove leading ../ to get the relative path from project root
            clean_path = data_path.lstrip("../").lstrip(".\\")
            file_path = Path(__file__).parent.parent.parent / clean_path
        self.data = pd.read_csv(file_path)
        self.problem = AllocationProblem(df=self.data, total_budget=total_budget)
        self.results = {}
        self._last_constraint_baseline_constraints: Dict[str, Any] = {}
        self._last_fairness_preferences: Optional[StakeholderPreferences] = None
        self._last_stakeholders: Optional[List[StakeholderProfile]] = None
        self._last_include_consensus: bool = True
        self._last_consensus_tolerance: float = 0.05

    def run_baseline(self) -> Dict:
        """Run baseline utilitarian optimization."""
        optimizer = UtilitarianOptimizer(self.problem)
        baseline = optimizer.solve()
        self.results["baseline"] = baseline
        return baseline
    
    def run_constraint_baseline(self, constraints: Optional[Dict[str, Any]] = None) -> Dict:
        """Run constrained utilitarian optimization (e.g. with per-country budget cap)."""
        constraints = constraints or {}
        self._last_constraint_baseline_constraints = dict(constraints)
        optimizer = ConstrainedUtilitarianOptimizer(
            self.problem,
            country_cap=constraints.get("country_cap", 0.5),
            demographic_cap=constraints.get("demographic_cap", None),
            demographic_min_share=constraints.get("demographic_min_share", {}),
        )
        constrained_baseline = optimizer.solve()
        self.results["constrained_baseline"] = constrained_baseline
        return constrained_baseline

    def run_fairness(self, preferences: StakeholderPreferences) -> Dict:
        """Run fairness-aware optimization for one preference profile."""
        self._last_fairness_preferences = preferences
        optimizer = FairnessOptimizer(self.problem, preferences)
        fair_solution = optimizer.solve()
        self.results["fairness"] = fair_solution
        return fair_solution

    def aggregate_preferences(self, stakeholders: List[StakeholderProfile]) -> StakeholderPreferences:
        """Aggregate stakeholder preferences with influence-weighted averaging.

        Three aggregation rules:
        - metric_weights: influence-weighted mean across all stakeholders.
        - demographic_constraints: only constraints backed by >50% of total
          influence are kept, averaged over the stakeholders who hold them.
          Constraints held by a minority are dropped to avoid imposing a
          binding floor nobody meaningfully voted for.
        - fairness_mode: plurality vote weighted by influence; ties broken
          toward the less restrictive mode (utilitarian > proportional > max-min).
        """
        if not stakeholders:
            raise ValueError("stakeholders must contain at least one StakeholderProfile.")

        total_influence = float(sum(max(float(s.influence), 0.0) for s in stakeholders))
        if total_influence <= 0:
            total_influence = float(len(stakeholders))

        metric_totals: Dict[str, float] = {}
        constraint_weighted_sum: Dict[str, float] = {}
        constraint_influence_sum: Dict[str, float] = {}
        fairness_mode_votes: Dict[str, float] = {}
        coverage_weighted_sum = 0.0
        coverage_influence_sum = 0.0

        for stakeholder in stakeholders:
            weight = max(float(stakeholder.influence), 0.0)
            if weight == 0:
                continue

            fairness_mode_votes[stakeholder.preferences.fairness_mode] = (
                fairness_mode_votes.get(stakeholder.preferences.fairness_mode, 0.0) + weight
            )
            for metric, value in stakeholder.preferences.metric_weights.items():
                metric_totals[metric] = metric_totals.get(metric, 0.0) + weight * float(value)
            for key, value in stakeholder.preferences.demographic_constraints.items():
                constraint_weighted_sum[key] = (
                    constraint_weighted_sum.get(key, 0.0) + weight * float(value)
                )
                constraint_influence_sum[key] = (
                    constraint_influence_sum.get(key, 0.0) + weight
                )
            if stakeholder.preferences.min_coverage_share is not None:
                coverage_weighted_sum += weight * float(stakeholder.preferences.min_coverage_share)
                coverage_influence_sum += weight

        metric_weights = {
            metric: total / total_influence for metric, total in metric_totals.items()
        }

        # Only keep constraints backed by a majority (>50%) of total influence.
        demographic_constraints = {
            key: constraint_weighted_sum[key] / constraint_influence_sum[key]
            for key in constraint_weighted_sum
            if constraint_influence_sum[key] / total_influence > 0.5
        }

        min_coverage_share = None
        if coverage_influence_sum / total_influence > 0.5:
            min_coverage_share = coverage_weighted_sum / coverage_influence_sum

        # Plurality vote; tie-break toward less restrictive mode.
        mode_priority = {"utilitarian": 0, "proportional": 1, "max-min": 2}
        fairness_mode = (
            min(
                fairness_mode_votes.items(),
                key=lambda item: (-item[1], mode_priority.get(item[0], 99)),
            )[0]
            if fairness_mode_votes
            else "utilitarian"
        )

        return StakeholderPreferences(
            metric_weights=metric_weights,
            demographic_constraints=demographic_constraints,
            fairness_mode=fairness_mode,
            min_coverage_share=min_coverage_share,
        )
    
    def build_lexicographic_consensus(
        self,
        stakeholders: List[StakeholderProfile],
        efficiency_tolerance: float = 0.05,
    ) -> Dict:
        """Aggregate preferences then run LexicographicConsensusOptimizer."""
        consensus_prefs = self.aggregate_preferences(stakeholders)
        optimizer = LexicographicConsensusOptimizer(
            self.problem,
            consensus_prefs,
            efficiency_tolerance=efficiency_tolerance,
        )
        return optimizer.solve()

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

        self._last_stakeholders = list(stakeholders)
        self._last_include_consensus = bool(include_consensus)
    
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
            consensus_result = self.build_lexicographic_consensus(
                stakeholders,
                efficiency_tolerance=0.05,
            )
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

    def generate_pareto_frontier(self, filepath=None):
        constrained_cfgs = []
        if self._last_constraint_baseline_constraints:
            country_cap = self._last_constraint_baseline_constraints.get("country_cap", 0.5)
            constrained_cfgs.append(
                {
                    "label": f"constrained\\n({int(country_cap * 100)}% cap)",
                    "country_cap": country_cap,
                    "demographic_cap": self._last_constraint_baseline_constraints.get("demographic_cap", None),
                    "demographic_min_share": self._last_constraint_baseline_constraints.get("demographic_min_share", {}),
                }
            )

        pareto = ParetoBoundary(self.problem)
        pareto.generate_solutions(
            fairness_preferences=self._last_fairness_preferences,
            stakeholders=self._last_stakeholders,
            include_consensus=self._last_include_consensus,
            efficiency_tolerance=self._last_consensus_tolerance,
            constrained_configs=constrained_cfgs,
        )
        pareto.plot(filepath=filepath)
