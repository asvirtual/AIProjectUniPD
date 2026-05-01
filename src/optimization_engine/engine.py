# ============================================================================
# Allocation Engine: integrates data processing, preference elicitation, optimization, and evaluation
# ============================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import AllocationProblem, StakeholderPreferences, StakeholderProfile
from .solvers import UtilitarianOptimizer, FairnessOptimizer
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
        """Aggregate stakeholder preferences with influence-weighted averaging."""
        if not stakeholders:
            raise ValueError("stakeholders must contain at least one StakeholderProfile.")

        total_influence = float(sum(max(float(stakeholder.influence), 0.0) for stakeholder in stakeholders))
        if total_influence <= 0:
            total_influence = float(len(stakeholders))

        metric_totals: Dict[str, float] = {}
        constraint_totals: Dict[str, float] = {}
        fairness_mode_votes: Dict[str, float] = {}

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
                constraint_totals[key] = constraint_totals.get(key, 0.0) + weight * float(value)

        metric_weights = {
            metric: total / total_influence for metric, total in metric_totals.items()
        }
        demographic_constraints = {
            key: total / total_influence for key, total in constraint_totals.items()
        }

        fairness_mode = (
            max(fairness_mode_votes.items(), key=lambda item: item[1])[0]
            if fairness_mode_votes
            else "utilitarian"
        )

        return StakeholderPreferences(
            metric_weights=metric_weights,
            demographic_constraints=demographic_constraints,
            fairness_mode=fairness_mode,
        )

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
        pareto.generate_solutions(["utilitarian", "proportional", "max-min"])
        pareto.plot(filepath=filepath)
