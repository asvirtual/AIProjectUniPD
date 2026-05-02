# ============================================================================
# Visualization of optimization results
# ============================================================================
from typing import Optional, Dict, List, Any
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .models import AllocationProblem, StakeholderPreferences, StakeholderProfile, BUDGET_SCALE_FACTOR
from .solvers import (
    UtilitarianOptimizer,
    FairnessOptimizer,
    ConstrainedUtilitarianOptimizer,
    LexicographicConsensusOptimizer,
)
from .metrics import FairnessMetrics

# ============================================================================
# Pareto boundary visualization for trade-offs between utilitarian and fairness objectives
# ============================================================================

class ParetoBoundary:
    """Generate and plot efficiency-equity trade-off points."""

    def __init__(self, problem: AllocationProblem):
        self.problem = problem
        self.frontier = []  # list of (efficiency, gini, gini_count, mode_label, frac)

    @staticmethod
    def _aggregate_preferences(stakeholders: List[StakeholderProfile]) -> StakeholderPreferences:
        """Aggregate stakeholder preferences with influence-weighted rules."""
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
        demographic_constraints = {
            key: constraint_weighted_sum[key] / constraint_influence_sum[key]
            for key in constraint_weighted_sum
            if constraint_influence_sum[key] / total_influence > 0.5
        }

        min_coverage_share = None
        if coverage_influence_sum / total_influence > 0.5:
            min_coverage_share = coverage_weighted_sum / coverage_influence_sum

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

    def _append_result(self, result: Dict[str, Any], problem: AllocationProblem, label: str, frac: float):
        efficiency = FairnessMetrics.total_lives_impacted(result, problem)
        gini = FairnessMetrics.gini_coefficient(result, problem)
        gini_count = FairnessMetrics.gini_count(result, problem)
        self.frontier.append((efficiency, gini, gini_count, label, frac))

    def generate_solutions(
        self,
        fairness_preferences: Optional[StakeholderPreferences] = None,
        stakeholders: Optional[List[StakeholderProfile]] = None,
        include_consensus: bool = True,
        efficiency_tolerance: float = 0.05,
        budget_fractions: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        constrained_configs: Optional[List[Dict]] = None,
    ):
        """
        Generate Pareto frontier solutions across budget fractions.

        Args:
            fairness_preferences: profile used by run_fairness() in main.
            stakeholders: profiles used by run_fairness_simulations() in main.
            include_consensus: when True, add consensus lexicographic series.
            efficiency_tolerance: consensus efficiency tolerance.
            budget_fractions: fractions of total budget to sweep over
            constrained_configs: optional list of dicts, each with:
                - "label": display name, e.g. "constrained\n(10% cap)"
                - "country_cap": float, e.g. 0.10
                - "demographic_cap": float or None
                - "demographic_min_share": dict or {}
        """
        self.frontier = []

        # Restore original unscaled budget (AllocationProblem.__post_init__ divides by BUDGET_SCALE_FACTOR)
        original_budget = self.problem.total_budget * BUDGET_SCALE_FACTOR

        for frac in budget_fractions:
            scaled_problem = AllocationProblem(
                df=self.problem.df,
                total_budget=original_budget * frac,
            )

            # Baseline utilitarian at this budget fraction.
            try:
                baseline = UtilitarianOptimizer(scaled_problem).solve()
                self._append_result(baseline, scaled_problem, "baseline", frac)
            except Exception as exc:
                print(f"Skipped baseline@{frac}: {exc}")

            # Constrained utilitarian profile(s) from main config.
            if constrained_configs:
                for cfg in constrained_configs:
                    label = cfg.get("label", "constrained")
                    try:
                        constrained = ConstrainedUtilitarianOptimizer(
                            scaled_problem,
                            country_cap=cfg.get("country_cap", 0.5),
                            demographic_cap=cfg.get("demographic_cap", None),
                            demographic_min_share=cfg.get("demographic_min_share", {}),
                        ).solve()
                        self._append_result(constrained, scaled_problem, label, frac)
                    except Exception as exc:
                        print(f"Skipped {label}@{frac}: {exc}")

            # Main fairness profile.
            if fairness_preferences is not None:
                try:
                    fairness = FairnessOptimizer(scaled_problem, fairness_preferences).solve()
                    self._append_result(fairness, scaled_problem, "fairness", frac)
                except Exception as exc:
                    print(f"Skipped fairness@{frac}: {exc}")

            # Stakeholder profiles from simulations.
            if stakeholders:
                for i, stakeholder in enumerate(stakeholders, start=1):
                    name = (stakeholder.name or "").strip() or f"stakeholder_{i}"
                    try:
                        sim_result = FairnessOptimizer(scaled_problem, stakeholder.preferences).solve()
                        self._append_result(sim_result, scaled_problem, name, frac)
                    except Exception as exc:
                        print(f"Skipped {name}@{frac}: {exc}")

                if include_consensus:
                    try:
                        consensus_prefs = self._aggregate_preferences(stakeholders)
                        consensus = LexicographicConsensusOptimizer(
                            scaled_problem,
                            consensus_prefs,
                            efficiency_tolerance=efficiency_tolerance,
                        ).solve()
                        self._append_result(consensus, scaled_problem, "consensus", frac)
                    except Exception as exc:
                        print(f"Skipped consensus@{frac}: {exc}")

    def plot(self, filepath=None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Style maps — constrained entries keyed by their label
        base_colors  = {"utilitarian": "#2196F3", "proportional": "#FF9800", "max-min": "#4CAF50"}
        base_markers = {"utilitarian": "o",       "proportional": "s",       "max-min": "^"}

        # Extra palette for any constrained configs
        extra_colors  = ["#9C27B0", "#E91E63", "#00BCD4", "#FF5722", "#FFEB3B", "#673AB7", "#009688"]
        extra_markers = ["P", "X", "D", "h", "v", "<", ">", "*", "+", "d"]

        # Build per-mode style, assigning extras to unknown labels
        colors  = dict(base_colors)
        markers = dict(base_markers)
        extra_idx = 0
        for (_, _, _, mode, _) in self.frontier:
            if mode not in colors:
                colors[mode]  = extra_colors[extra_idx % len(extra_colors)]
                markers[mode] = extra_markers[extra_idx % len(extra_markers)]
                extra_idx += 1

        # Group points by mode
        by_mode = defaultdict(list)
        for (eff, gini, gini_count, mode, frac) in self.frontier:
            by_mode[mode].append((eff, gini, gini_count, frac))

        # ─── Panel 1: Efficiency vs Gini Coefficient ──────────────────
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

        scatter_obj1 = None
        for mode, points in by_mode.items():
            points.sort(key=lambda p: p[0])
            xs    = [p[0] for p in points]
            ys    = [p[1] for p in points]
            fracs = [p[3] for p in points]

            ax1.plot(xs, ys, color=colors[mode], alpha=0.4, linewidth=1, linestyle="--")
            scatter_obj1 = ax1.scatter(
                xs, ys,
                c=fracs, cmap="YlOrRd",
                marker=markers[mode], s=80,
                edgecolors=colors[mode], linewidths=1.5,
                label=mode, zorder=3,
                vmin=0.1, vmax=1.0,
            )

        if scatter_obj1 is not None:
            cbar1 = plt.colorbar(scatter_obj1, ax=ax1)
            cbar1.set_label("Budget Fraction", fontsize=10)

        ax1.set_xlabel("Total Lives Impacted (Efficiency)", fontsize=12)
        ax1.set_ylabel("Gini Coefficient (Inequality ↓ better)", fontsize=12)
        ax1.set_title("Pareto Frontier: Efficiency vs Gini Coefficient", fontsize=13)

        ax1.annotate(
            "← Ideal region\n(high efficiency,\nlow inequality)",
            xy=(0.98, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=9, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7),
        )

        ax1.legend(title="Fairness Mode", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ─── Panel 2: Efficiency vs Gini Count ──────────────────────────
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

        scatter_obj2 = None
        for mode, points in by_mode.items():
            points.sort(key=lambda p: p[0])
            xs    = [p[0] for p in points]
            ys    = [p[2] for p in points]  # gini_count
            fracs = [p[3] for p in points]

            ax2.plot(xs, ys, color=colors[mode], alpha=0.4, linewidth=1, linestyle="--")
            scatter_obj2 = ax2.scatter(
                xs, ys,
                c=fracs, cmap="YlOrRd",
                marker=markers[mode], s=80,
                edgecolors=colors[mode], linewidths=1.5,
                label=mode, zorder=3,
                vmin=0.1, vmax=1.0,
            )

        if scatter_obj2 is not None:
            cbar2 = plt.colorbar(scatter_obj2, ax=ax2)
            cbar2.set_label("Budget Fraction", fontsize=10)

        ax2.set_xlabel("Total Lives Impacted (Efficiency)", fontsize=12)
        ax2.set_ylabel("Gini Count (Population inequality ↓ better)", fontsize=12)
        ax2.set_title("Pareto Frontier: Efficiency vs Gini Count", fontsize=13)

        ax2.annotate(
            "← Ideal region\n(high efficiency,\nlow inequality)",
            xy=(0.98, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=9, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7),
        )

        ax2.legend(title="Fairness Mode", fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show()