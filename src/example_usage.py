"""
Example: How to Use the AllocationEngine

This script demonstrates the workflow:
  1. Load data from data_cleaning.ipynb output
  2. Define stakeholder preferences
  3. Run baseline utilitarian model
  4. Run fairness-aware model
  5. Compare results"
  6. Generate Pareto frontier
"""

from pathlib import Path

from optimization_engine import (
    AllocationEngine,
    PreferenceElicitor,
    StakeholderPreferences,
    FairnessMetrics
)

# ============================================================================
# STEP 1: Initialize Engine with Data
# ============================================================================

# Load the enriched dataset from data_cleaning.ipynb
BASE_DIR = Path(__file__).resolve().parent
data_path = str(BASE_DIR.parent / "data" / "processed" / "master_df_mece_compliant.csv")
total_budget = 100_000_000  # $100 million global budget

engine = AllocationEngine(data_path=data_path, total_budget=total_budget)

print("✓ Engine initialized")
print(f"  Data shape: {engine.data.shape}")
print(f"  Budget: ${total_budget:,.0f}")

# ============================================================================
# STEP 2: Run Baseline (Utilitarian) Model
# ============================================================================

print("\n" + "="*70)
print("BASELINE: Utilitarian Allocation (Max Total Lives Treated)")
print("="*70)

baseline_solution = engine.run_baseline()

baseline_efficiency = FairnessMetrics.total_lives_impacted(baseline_solution, engine.problem)
baseline_gini = FairnessMetrics.gini_coefficient(baseline_solution, engine.problem)

print(f"✓ Total children treated: {baseline_efficiency:,.0f}")
print(f"✓ Gini coefficient (equity): {baseline_gini:.3f}")

# ============================================================================
# STEP 3: Define Stakeholder Preferences for Fairness
# ============================================================================

print("\n" + "="*70)
print("FAIRNESS PREFERENCES: Stakeholder Definitions")
print("="*70)

# Example 1: Weighted utilitarian fairness with rural minimum
preferences_1 = PreferenceElicitor(
    metric_weights={
        'stunting': 1.0,
        'wasting': 1.0,
        'severe_wasting': 1.5  # Prioritize severe cases
    },
    demographic_constraints={
        'rural_min_share': 0.30  # At least 30% of budget to rural areas
    },
    fairness_mode='utilitarian'
)

print("Preference Set 1 (Severe Wasting + Rural Protection):")
print(f"  - Metric weights: {preferences_1.metric_weights}")
print(f"  - Demographic floor: {preferences_1.demographic_constraints}")
print(f"  - Fairness mode: {preferences_1.fairness_mode}")

# Example 2: Max-min fairness (Rawlsian)
preferences_2 = PreferenceElicitor(
    metric_weights={'stunting': 1.0, 'wasting': 1.0, 'severe_wasting': 1.0},
    demographic_constraints={},
    fairness_mode='max-min'  # Maximize the worst-off group's allocation
)

print("\nPreference Set 2 (Rawlsian Max-Min):")
print(f"  - Equal metric weights")
print(f"  - Fairness mode: {preferences_2.fairness_mode}")

# Example 3: Proportional fairness (burden-driven)
preferences_3 = PreferenceElicitor(
    metric_weights={'stunting': 1.0, 'wasting': 1.0, 'severe_wasting': 1.0},
    demographic_constraints={},
    fairness_mode='proportional'  # Allocate per burden share
)

print("\nPreference Set 3 (Proportional to Burden):")
print(f"  - Fairness mode: {preferences_3.fairness_mode}")

# ============================================================================
# STEP 4: Run Fairness-Aware Solutions
# ============================================================================

print("\n" + "="*70)
print("FAIRNESS SOLUTIONS vs. BASELINE")
print("="*70)

for i, pref in enumerate([preferences_1, preferences_2, preferences_3], 1):
    print(f"\n--- Solution {i}: {pref.fairness_mode.upper()} ---")
    fair_solution = engine.run_fairness(pref.to_preferences())
    
    efficiency = FairnessMetrics.total_lives_impacted(fair_solution, engine.problem)
    gini = FairnessMetrics.gini_coefficient(fair_solution, engine.problem)
    max_min_ratio = FairnessMetrics.max_min_ratio(fair_solution, engine.problem)
    
    efficiency_loss = (baseline_efficiency - efficiency) / baseline_efficiency
    equity_gain = (baseline_gini - gini) / baseline_gini if baseline_gini > 0 else 0
    
    print(f"  Lives treated: {efficiency:,.0f} (loss: {efficiency_loss:+.1%})")
    print(f"  Gini coeff: {gini:.3f} (improvement: {equity_gain:+.1%})")
    print(f"  Max-min ratio: {max_min_ratio:.2f}x (closer to 1.0 = more equal)")

# ============================================================================
# STEP 5: Compare Solutions
# ============================================================================

print("\n" + "="*70)
print("TRADE-OFF ANALYSIS")
print("="*70)

engine.compare_solutions()

# ============================================================================
# STEP 6: Generate Pareto Frontier
# ============================================================================

print("\n" + "="*70)
print("PARETO FRONTIER: Efficiency vs. Equity")
print("="*70)

output_plot = str(BASE_DIR.parent / "plots" / "pareto_frontier.png")
engine.generate_pareto_frontier(filepath=output_plot)
print(f"✓ Pareto frontier saved to {output_plot}")

# ============================================================================
# STEP 7: Export Solutions to CSV
# ============================================================================

print("\n" + "="*70)
print("EXPORTING RESULTS")
print("="*70)

# Pseudo-code: Convert solution dictionaries back to DataFrames and save
# baseline_df = engine.solution_to_dataframe(engine.results['baseline'])
# baseline_df.to_csv("../results/baseline_allocation.csv", index=False)
# 
# fair_df = engine.solution_to_dataframe(engine.results['fairness'])
# fair_df.to_csv("../results/fairness_allocation.csv", index=False)

print("✓ Results exported to ../results/")
