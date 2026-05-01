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
        self.frontier = []  # list of (efficiency, gini, gini_count, mode_label, frac)

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
                    gini_count = FairnessMetrics.gini_count(result, scaled_problem)
                    self.frontier.append((efficiency, gini, gini_count, mode, frac))
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
                        gini_count = FairnessMetrics.gini_count(result, scaled_problem)
                        self.frontier.append((efficiency, gini, gini_count, label, frac))
                    except Exception as e:
                        print(f"Skipped {label}@{frac}: {e}")

    def plot(self, filepath=None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

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