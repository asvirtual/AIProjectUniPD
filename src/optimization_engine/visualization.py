# ============================================================================
# Visualization of optimization results
# ============================================================================
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .models import AllocationProblem, StakeholderPreferences
from .solvers import FairnessOptimizer
from .metrics import FairnessMetrics

# ============================================================================
# Pareto boundary visualization for trade-offs between utilitarian and fairness objectives
# ============================================================================

class ParetoBoundary:
    """Generate and plot efficiency-equity trade-off points."""

    def __init__(self, problem: AllocationProblem):
        self.problem = problem
        self.frontier = []

    def generate_solutions(self, fairness_modes, budget_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):

        for mode in fairness_modes:
            for frac in budget_fractions:
                scaled_problem = AllocationProblem(
                    df=self.problem.df,
                    total_budget=self.problem.total_budget * frac
                )
                prefs = StakeholderPreferences(
                    metric_weights={"stunting": 1.0, "wasting": 1.0, "severe_wasting": 1.0},
                    demographic_constraints={},
                    fairness_mode=mode
                )
                try:
                    result = FairnessOptimizer(scaled_problem, prefs).solve()
                    efficiency = FairnessMetrics.total_lives_impacted(result, scaled_problem)
                    gini = FairnessMetrics.gini_coefficient(result, scaled_problem)
                    
                    # DEBUG: Check if efficiency is in scaled or unscaled units
                    if mode == "utilitarian" and frac == 1.0:
                        print(f"DEBUG {mode}@{frac}: result['total_treated']={result.get('total_treated', 'N/A')}, efficiency={efficiency}, efficiency/1e6={efficiency/1e6:.3f}M")
                    
                    self.frontier.append((efficiency, gini, mode, frac))
                except Exception as e:
                    print(f"Skipped {mode}@{frac}: {e}")

    def plot(self, filepath=None):
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = {"utilitarian": "#2196F3", "proportional": "#FF9800", "max-min": "#4CAF50"}
        markers = {"utilitarian": "o", "proportional": "s", "max-min": "^"}
        
        # Group by mode
        from collections import defaultdict
        by_mode = defaultdict(list)
        for (eff, gini, mode, frac) in self.frontier:
            by_mode[mode].append((eff, gini, frac))
        
        # Debug: print efficiency values
        if self.frontier:
            sample_effs = [eff for eff, _, _, _ in self.frontier[:3]]
            print(f"DEBUG: Sample efficiency values: {sample_effs}")
            print(f"DEBUG: Min/Max efficiency: {min(eff for eff, _, _, _ in self.frontier)} - {max(eff for eff, _, _, _ in self.frontier)}")
            print(f"DEBUG: All frontier data (first 3): {self.frontier[:3]}")
        
        # Apply formatter BEFORE plotting
        # Efficiency values are already in millions, so just add 'M' suffix
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}M'))
        
        for mode, points in by_mode.items():
            # Sort by efficiency for line connection
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            fracs = [p[2] for p in points]
            
            ax.plot(xs, ys, color=colors[mode], alpha=0.4, linewidth=1, linestyle="--")
            
            sc = ax.scatter(xs, ys, c=fracs, cmap="YlOrRd", 
                            marker=markers[mode], s=80, 
                            edgecolors=colors[mode], linewidths=1.5,
                            label=mode, zorder=3)
        
        # Colorbar for budget fraction
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Budget Fraction", fontsize=10)
        
        ax.set_xlabel("Total Lives Impacted (Efficiency)", fontsize=12)
        ax.set_ylabel("Gini Coefficient (Inequality ↓ better)", fontsize=12)
        ax.set_title("Pareto Frontier: Efficiency vs Equity Trade-off", fontsize=14)
        
        # Annotate ideal corner
        ax.annotate("← Ideal region\n(high efficiency,\nlow inequality)", 
                    xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=9, color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        
        ax.legend(title="Fairness Mode", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show()
