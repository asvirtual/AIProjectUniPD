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
    data_path="../data/processed/master_df_with_counts_and_costs.csv",
    total_budget=100_000_000,  # $100 million global budget
)

print("✓ Engine initialized")
solution = engine.run_baseline()
print("✓ Baseline solution computed")

preferences = StakeholderPreferences(
    metric_weights={"stunting": 1.0, "wasting": 1.0, "severe_wasting": 1.5},
    demographic_constraints={"rural_minimum": 0.2},
    fairness_mode="proportional",
)

solution_fair = engine.run_fairness(preferences=preferences)
print("✓ Fairness-aware solution computed")

# Multi-stakeholder profiles: clinical, equity, efficiency
stakeholders = [
    StakeholderProfile(
        name="clinical",
        influence=0.4,
        preferences=StakeholderPreferences(
            metric_weights={"stunting": 0.8, "wasting": 1.0, "severe_wasting": 1.6},
            demographic_constraints={"rural_minimum": 0.15},
            fairness_mode="utilitarian",
        ),
    ),
    StakeholderProfile(
        name="equity",
        influence=0.35,
        preferences=StakeholderPreferences(
            metric_weights={"stunting": 1.0, "wasting": 1.0, "severe_wasting": 1.2},
            demographic_constraints={"rural_minimum": 0.4},
            fairness_mode="max-min",
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
print(f"✓ Ran {len(sim_results)} stakeholder simulations (including consensus)")

consensus = sim_results.get("consensus")
if consensus:
    consensus_eff = FairnessMetrics.total_lives_impacted(consensus, engine.problem)
    consensus_gini = FairnessMetrics.gini_coefficient(consensus, engine.problem)
    print(f"✓ Consensus efficiency: {consensus_eff:,.0f} lives")
    print(f"✓ Consensus gini: {consensus_gini:.3f}")

engine.compare_solutions()
print("✓ Solutions compared on efficiency and equity metrics")
engine.generate_pareto_frontier()
print("✓ Pareto frontier generated to visualize trade-offs")
