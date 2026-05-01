# ============================================================================
# Visualization of optimization results
# ============================================================================
from typing import Optional, Dict, List
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .models import AllocationProblem, StakeholderPreferences, BUDGET_SCALE_FACTOR
from .solvers import FairnessOptimizer, ConstrainedUtilitarianOptimizer
from .metrics import FairnessMetrics

# ============================================================================
# Pareto boundary visualization for trade-offs between utilitarian and fairness objectives
# ============================================================================

class ParetoBoundary:
    """Generate and plot efficiency-equity trade-off points."""

    def __init__(self, problem: AllocationProblem):
        self.problem = problem
        self.frontier = []  # list of (efficiency, gini, mode_label, frac)

    def generate_solutions(
        self,
        fairness_modes: List[str],
        budget_fractions: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        constrained_configs: Optional[List[Dict]] = None,
    ):
        """
        Generate Pareto frontier solutions across budget fractions.

        Args:
            fairness_modes: e.g. ["utilitarian", "proportional", "max-min"]
            budget_fractions: fractions of total budget to sweep over
            constrained_configs: optional list of dicts, each with:
                - "label": display name, e.g. "constrained\n(10% cap)"
                - "country_cap": float, e.g. 0.10
                - "demographic_cap": float or None
                - "demographic_min_share": dict or {}
        """
        # Restore original unscaled budget (AllocationProblem.__post_init__ divides by BUDGET_SCALE_FACTOR)
        original_budget = self.problem.total_budget * BUDGET_SCALE_FACTOR

        # --- Standard fairness modes ---
        for mode in fairness_modes:
            for frac in budget_fractions:
                scaled_problem = AllocationProblem(
                    df=self.problem.df,
                    total_budget=original_budget * frac,
                )
                prefs = StakeholderPreferences(
                    metric_weights={"stunting": 1.0, "wasting": 1.0, "severe_wasting": 1.0},
                    demographic_constraints={},
                    fairness_mode=mode,
                )
                try:
                    result = FairnessOptimizer(scaled_problem, prefs).solve()
                    efficiency = FairnessMetrics.total_lives_impacted(result, scaled_problem)
                    gini = FairnessMetrics.gini_coefficient(result, scaled_problem)
                    self.frontier.append((efficiency, gini, mode, frac))
                except Exception as e:
                    print(f"Skipped {mode}@{frac}: {e}")

        # --- Constrained utilitarian configs ---
        if constrained_configs:
            for cfg in constrained_configs:
                label = cfg.get("label", "constrained")
                for frac in budget_fractions:
                    scaled_problem = AllocationProblem(
                        df=self.problem.df,
                        total_budget=original_budget * frac,
                    )
                    optimizer = ConstrainedUtilitarianOptimizer(
                        scaled_problem,
                        country_cap=cfg.get("country_cap", 0.5),
                        demographic_cap=cfg.get("demographic_cap", None),
                        demographic_min_share=cfg.get("demographic_min_share", {}),
                    )
                    try:
                        result = optimizer.solve()
                        efficiency = FairnessMetrics.total_lives_impacted(result, scaled_problem)
                        gini = FairnessMetrics.gini_coefficient(result, scaled_problem)
                        self.frontier.append((efficiency, gini, label, frac))
                    except Exception as e:
                        print(f"Skipped {label}@{frac}: {e}")

    def plot(self, filepath=None):

        fig, ax = plt.subplots(figsize=(12, 7))

        # Style maps — constrained entries keyed by their label
        base_colors  = {"utilitarian": "#2196F3", "proportional": "#FF9800", "max-min": "#4CAF50"}
        base_markers = {"utilitarian": "o",       "proportional": "s",       "max-min": "^"}

        # Extra palette for any constrained configs
        extra_colors  = ["#9C27B0", "#E91E63", "#00BCD4", "#FF5722"]
        extra_markers = ["P", "X", "D", "h"]

        # Build per-mode style, assigning extras to unknown labels
        colors  = dict(base_colors)
        markers = dict(base_markers)
        extra_idx = 0
        for (_, _, mode, _) in self.frontier:
            if mode not in colors:
                colors[mode]  = extra_colors[extra_idx % len(extra_colors)]
                markers[mode] = extra_markers[extra_idx % len(extra_markers)]
                extra_idx += 1

        # Group points by mode
        by_mode = defaultdict(list)
        for (eff, gini, mode, frac) in self.frontier:
            by_mode[mode].append((eff, gini, frac))

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

        scatter_obj = None
        for mode, points in by_mode.items():
            points.sort(key=lambda p: p[0])
            xs    = [p[0] for p in points]
            ys    = [p[1] for p in points]
            fracs = [p[2] for p in points]

            ax.plot(xs, ys, color=colors[mode], alpha=0.4, linewidth=1, linestyle="--")
            scatter_obj = ax.scatter(
                xs, ys,
                c=fracs, cmap="YlOrRd",
                marker=markers[mode], s=80,
                edgecolors=colors[mode], linewidths=1.5,
                label=mode, zorder=3,
                vmin=0.1, vmax=1.0,   # consistent colorbar scale across all series
            )

        if scatter_obj is not None:
            cbar = plt.colorbar(scatter_obj, ax=ax)
            cbar.set_label("Budget Fraction", fontsize=10)

        ax.set_xlabel("Total Lives Impacted (Efficiency)", fontsize=12)
        ax.set_ylabel("Gini Coefficient (Inequality ↓ better)", fontsize=12)
        ax.set_title("Pareto Frontier: Efficiency vs Equity Trade-off", fontsize=14)

        ax.annotate(
            "← Ideal region\n(high efficiency,\nlow inequality)",
            xy=(0.98, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=9, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7),
        )

        ax.legend(title="Fairness Mode", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show()