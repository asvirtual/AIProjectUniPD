from optimization_engine import (
    AllocationEngine,
    StakeholderProfile,
    StakeholderPreferences,
    FairnessMetrics
)

"""Example usage of the allocation engine with constraints and fairness preferences.
Steps:
    1. Initialize engine with enriched dataset and budget
    2. Run baseline utilitarian model
    3. Define stakeholder preferences for fairness
    4. Run fairness-aware models with constraints
    5. Compare results on efficiency and equity metrics
    6. Generate Pareto frontier to visualize trade-offs
    7. Analyze and interpret the results
"""

engine = AllocationEngine(
    data_path="../data/processed/master_df_mece_compliant.csv",
    total_budget=500_000_000,  # $100 million global budget
)

print("[OK] Engine initialized")
solution = engine.run_baseline()
print("[OK] Baseline solution computed")

solution_constrained = engine.run_constraint_baseline(constraints={"country_cap": 0.1})
print("[OK] Constrained baseline solution computed")

print("\n=== CONSTRAINED UTILITARIAN RESULTS ===")
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
print(by_country_constrained[["Country", "total_treated", "total_spend", "pct_spend", "cost_per_child"]].head(10).to_string(index=False))

# Compare constrained vs baseline
baseline_summary = FairnessMetrics.build_summary(solution, engine.problem, label="Baseline")
constrained_summary = FairnessMetrics.build_summary(solution_constrained, engine.problem, label="Constrained")
print("\n--- Metrics Comparison: Baseline vs Constrained ---")
print(f"{'Metric':<30} {'Baseline':>15} {'Constrained':>15} {'Δ':>10}")
print("-" * 70)
for metric in ["total_lives_impacted", "gini_coefficient", "gini_count", "max_min_ratio", "proportionality_violation"]:
    baseline_val = baseline_summary.get(metric, 0)
    constrained_val = constrained_summary.get(metric, 0)
    delta = constrained_val - baseline_val if isinstance(baseline_val, (int, float)) else "N/A"
    print(f"{metric:<30} {baseline_val:>15.2f} {constrained_val:>15.2f} {delta:>10.2f}")

preferences = StakeholderPreferences(
    metric_weights={"stunting": 1.0, "wasting": 1.0, "severe_wasting": 1.5},
    demographic_constraints={"rural_min_share": 0.2},
    fairness_mode="proportional",
)

solution_fair = engine.run_fairness(preferences=preferences)
print("[OK] Fairness-aware solution computed")

# Multi-stakeholder profiles: clinical, equity, efficiency
stakeholders = [
    StakeholderProfile(
        name="clinical",
        influence=0.4,
        preferences=StakeholderPreferences(
            metric_weights={"stunting": 0.8, "wasting": 1.0, "severe_wasting": 1.6},
            demographic_constraints={"rural_min_share": 0.15},
            fairness_mode="utilitarian",
        ),
    ),
    StakeholderProfile(
        name="equity",
        influence=0.35,
        preferences=StakeholderPreferences(
            metric_weights={"stunting": 1.0, "wasting": 1.0, "severe_wasting": 1.2},
            demographic_constraints={"rural_min_share": 0.4},
            fairness_mode="max-min",
            min_coverage_share=0.01,
        ),
    ),
    StakeholderProfile(
        name="efficiency",
        influence=0.25,
        preferences=StakeholderPreferences(
            metric_weights={"stunting": 1.1, "wasting": 1.0, "severe_wasting": 1.0},
            demographic_constraints={},
            fairness_mode="proportional",
        ),
    ),
]

sim_results = engine.run_fairness_simulations(stakeholders, include_consensus=True)
print(f"[OK] Ran {len(sim_results)} stakeholder simulations (including consensus)")
for (idx, res) in sim_results.items():
    print(f"\n--- Simulation: {idx} ---")
    print(f"Status            : {res['status']}")
    print(f"Total treated     : {res['total_treated']:,.0f}")
    print(f"Total spend       : ${res['total_spend']:,.0f}")
    print(f"Budget utilisation: {res['budget_utilisation_pct']:.1f}%")

    print("\n--- Summary by Country ---")
    by_country_baseline = res["allocation_df"].groupby("Country").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    by_country_baseline["pct_spend"] = 100 * by_country_baseline["total_spend"] / res["total_spend"]
    by_country_baseline["cost_per_child"] = by_country_baseline["total_spend"] / by_country_baseline["total_treated"]
    by_country_baseline = by_country_baseline.sort_values("total_spend", ascending=False)
    print(by_country_baseline[["Country", "total_treated", "total_spend", "pct_spend", "cost_per_child"]].head(10).to_string(index=False))





consensus = sim_results.get("consensus")
if consensus:
    consensus_eff = FairnessMetrics.total_lives_impacted(consensus, engine.problem)
    consensus_gini = FairnessMetrics.gini_coefficient(consensus, engine.problem)
    consensus_gini_count = FairnessMetrics.gini_count(consensus, engine.problem)
    print(f"[OK] Consensus efficiency: {consensus_eff:,.0f} lives")
    print(f"[OK] Consensus gini: {consensus_gini:.3f}")
    print(f"[OK] Consensus gini_count: {consensus_gini_count:.0f}")
    print(f"[OK] Efficiency ceiling   : {consensus.get('efficiency_ceiling', 0):,.0f}  |  floor: {consensus.get('efficiency_floor', 0):,.0f}  ({consensus.get('efficiency_tolerance', 0)*100:.0f}% tolerance)")
    if not consensus.get('demographic_constraints_applied', True):
        print("[WARN] Demographic constraints were relaxed — efficiency floor + constraints were jointly infeasible")

engine.compare_solutions()
print("[OK] Solutions compared on efficiency and equity metrics")
engine.generate_pareto_frontier()
print("[OK] Pareto frontier generated to visualize trade-offs")
