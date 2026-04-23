"""
Demo: Fair Budget Allocation Across Three Scenarios

Demonstrates the optimization engine with:
  1. Baseline utilitarian (max lives, no constraints)
  2. Country-constrained (10% cap per country, equal distribution)
  3. Demographic-constrained (20% country cap + 40% Rural + 20% poorest quintile)
"""

import pandas as pd
from pathlib import Path
from optimization_engine import AllocationProblem, UtilitarianOptimizer, ConstrainedUtilitarianOptimizer


def main():
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "processed" / "master_df_mece_compliant.csv"
    df = pd.read_csv(data_path)

    # Use all countries in the dataset
    problem = AllocationProblem(df=df, total_budget=100_000_000)
    print(f"Testing with {df['Country'].nunique()} countries\n")

    # ========================================================================
    # SCENARIO 1: Baseline Utilitarian (No Constraints)
    # ========================================================================
    print("=" * 70)
    print("BASELINE UTILITARIAN OPTIMIZER (No Constraints)")
    print("=" * 70)
    solution_baseline = UtilitarianOptimizer(problem).solve()
    
    print(f"Status            : {solution_baseline['status']}")
    print(f"Total treated     : {solution_baseline['total_treated']:,.0f}")
    print(f"Total spend       : ${solution_baseline['total_spend']:,.0f}")
    print(f"Budget utilisation: {solution_baseline['budget_utilisation_pct']:.1f}%")
    
    print("\n--- Summary by Country ---")
    by_country_baseline = solution_baseline["allocation_df"].groupby("Country").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    by_country_baseline["pct_spend"] = 100 * by_country_baseline["total_spend"] / solution_baseline["total_spend"]
    by_country_baseline["cost_per_child"] = by_country_baseline["total_spend"] / by_country_baseline["total_treated"]
    by_country_baseline = by_country_baseline.sort_values("total_spend", ascending=False)
    print(by_country_baseline[["Country", "total_treated", "total_spend", "pct_spend", "cost_per_child"]].head(10).to_string(index=False))

    # ========================================================================
    # SCENARIO 2: Constrained Utilitarian (10% cap per country)
    # ========================================================================
    country_cap_val = 0.1
    print("\n" + "=" * 70)
    print(f"CONSTRAINED UTILITARIAN OPTIMIZER ({100*country_cap_val:.0f}% Budget Cap per Country)")
    print("=" * 70)
    solution_constrained = ConstrainedUtilitarianOptimizer(problem, country_cap=country_cap_val).solve()
    
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
    print(by_country_constrained[["Country", "total_treated", "total_spend", "pct_spend", "cost_per_child"]].to_string(index=False))

    # ========================================================================
    # SCENARIO 3: Demographic + Country Constraints (Fairness-Aware)
    # ========================================================================
    country_cap_demographic = 0.20
    demographic_min_constraints = {
        "Rural": 0.40,                # Minimum 40% to all groups containing "Rural"
        "Wealth Quintile 1": 0.20     # Minimum 20% to all Q1 groups (both Rural and Urban)
    }
    constraint_str = ", ".join([f"{k} >= {100*v:.0f}%" for k, v in demographic_min_constraints.items()])
    print("\n" + "=" * 70)
    print(f"CONSTRAINED UTILITARIAN + DEMOGRAPHIC MINIMUM ({constraint_str}, {100*country_cap_demographic:.0f}% Country Cap)")
    print("=" * 70)
    solution_demographic_constrained = ConstrainedUtilitarianOptimizer(
        problem,
        country_cap=country_cap_demographic,
        demographic_cap=0.50,
        demographic_min_share=demographic_min_constraints
    ).solve()
    
    print(f"Status            : {solution_demographic_constrained['status']}")
    print(f"Total treated     : {solution_demographic_constrained['total_treated']:,.0f}")
    print(f"Total spend       : ${solution_demographic_constrained['total_spend']:,.0f}")
    print(f"Budget utilisation: {solution_demographic_constrained['budget_utilisation_pct']:.1f}%")
    
    print("\n--- Allocation by Demographic Group ---")
    by_demographic = solution_demographic_constrained["allocation_df"].groupby("Demographic_group").agg({
        "total_treated": "sum",
        "total_spend": "sum"
    }).reset_index()
    by_demographic["pct_spend"] = 100 * by_demographic["total_spend"] / solution_demographic_constrained["total_spend"]
    by_demographic["cost_per_child"] = by_demographic["total_spend"] / by_demographic["total_treated"]
    by_demographic = by_demographic.sort_values("total_spend", ascending=False)
    print(by_demographic.to_string(index=False))

    # ========================================================================
    # COMPARISON: Efficiency-Equity Tradeoff
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"COMPARISON: Country-only Caps ({100*country_cap_val:.0f}%) vs. Country + Demographic Minimum ({100*country_cap_demographic:.0f}%)")
    print("=" * 70)
    efficiency_loss_demographic = solution_constrained["total_treated"] - solution_demographic_constrained["total_treated"]
    efficiency_loss_demographic_pct = 100 * efficiency_loss_demographic / solution_constrained["total_treated"]
    print(f"Efficiency loss (children not treated): {efficiency_loss_demographic:,.0f} ({efficiency_loss_demographic_pct:.1f}%)")
    print(f"\nCountry-only caps: {solution_constrained['total_treated']:,.0f} children treated")
    print(f"With demographic minimum: {solution_demographic_constrained['total_treated']:,.0f} children treated")
    print(f"\nNote: Forcing allocation to Rural (higher-cost demographic) reduces overall efficiency.")

    # ========================================================================
    # OVERALL COMPARISON: All Three Scenarios
    # ========================================================================
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON: All Three Scenarios")
    print("=" * 70)
    baseline_loss = solution_baseline["total_treated"] - solution_constrained["total_treated"]
    demographic_loss = solution_constrained["total_treated"] - solution_demographic_constrained["total_treated"]
    total_loss = solution_baseline["total_treated"] - solution_demographic_constrained["total_treated"]
    
    print(f"Baseline (no constraints)              : {solution_baseline['total_treated']:>12,.0f} children")
    print(f"Country-level constraints ({100*country_cap_val:.0f}% cap)      : {solution_constrained['total_treated']:>12,.0f} children (loss: {baseline_loss:>8,.0f})")
    print(f"Country + Demographic constraints      : {solution_demographic_constrained['total_treated']:>12,.0f} children (loss: {total_loss:>8,.0f})")
    print(f"\nCost of fairness constraints:")
    print(f"  Country redistribution alone         : {100*baseline_loss/solution_baseline['total_treated'] if solution_baseline['total_treated'] > 0 else 0:.2f}%")
    print(f"  Additional demographic constraint    : {100*demographic_loss/solution_constrained['total_treated'] if solution_constrained['total_treated'] > 0 else 0:.2f}%")


if __name__ == "__main__":
    main()
