# Constrained Optimization Engine: Implementation Guide

## Overview

This boilerplate provides a modular framework for **multi-objective fairness-aware budget allocation**. The structure separates concerns into:

1. **Data Structures** – Problem definition & preferences
2. **Baseline Model** – Simple utilitarian LP
3. **Preference Module** – Stakeholder input collection
4. **Fairness Optimizers** – Multi-objective solvers for different fairness concepts
5. **Metrics** – Evaluation functions (efficiency, equity, fairness indices)
6. **Visualization** – Pareto frontiers & trade-off plots
7. **Orchestration** – High-level workflow "engine"

---

## Core Modules Explained

### 1. `AllocationProblem` & `StakeholderPreferences`

**What they do:**
- `AllocationProblem`: Wraps your data (from `master_df_with_counts_and_costs.csv`) plus budget constraint
- `StakeholderPreferences`: Captures what a stakeholder cares about (metric weights, demographic floors, fairness concept)

**Implementation note:**
```python
problem = AllocationProblem(
    data=df,  # Loaded CSV
    total_budget=100_000_000,
    metrics=['stunting', 'wasting', 'severe_wasting']
)
```

---

### 2. `UtilitarianOptimizer`

**What it does:**
Simple linear program:
```
Maximize: sum(allocation[i] * count[i])
Subject to: sum(allocation[i] * cost[i]) <= budget
            allocation[i] >= 0
```

**Implementation tasks:**
- `setup_variables()` – Create one LpVariable per (country, demographic_group, metric) tuple
  - Variable name: e.g., `"alloc_AFG_Rural_Quintile1_stunting"`
  - Bounds: each >= 0
  
- `add_objective()` – Use `lpSum()` to sum allocation × count for all variables
  
- `add_budget_constraint()` – Use `lpSum()` to constrain allocation × cost <= budget
  
- `solve()` – Call `self.model.solve(pulp.PULP_CBC_CMD(msg=0))`

**Why this baseline?**
- Establishes efficiency ceiling: max lives you can save
- Reveals cost trade-offs: which regions/metrics are "cheap"
- Exposes inequality: likely allocates heavily to high-burden, low-cost regions

---

### 3. `PreferenceElicitor`

**What it does:**
Collects user preferences in a clean, validated way.

**Implementation tasks:**
- `validate()` – Check weights sum to > 0, demographic constraints ≤ 1.0, fairness_mode is valid
- `from_json()` – Load preferences from a config file

**Example JSON:**
```json
{
  "metric_weights": {"stunting": 1.0, "severe_wasting": 1.5},
  "demographic_constraints": {"Rural": 0.30},
  "fairness_mode": "weighted-log"
}
```

---

### 4. `FairnessOptimizer`

**What it does:**
Solves **four different fairness concepts** via different objectives/constraints.

#### 4a. **Weighted Log-Utility** (`weighted-log`)
```
Maximize: sum(w[i] * log(1 + x[i]))
```
- `w[i]` = metric weight (e.g., 1.5 for severe wasting)
- `x[i]` = allocation to region i
- `log()` encodes diminishing returns (first $ saves most lives, last $ saves fewer)

**Implementation note:**
- PuLP is LP-only → piecewise linear approximation, OR
- Use `CVXPY` for native support of log (convex solver)

#### 4b. **Max-Min Fairness** (`max-min`)
```
Maximize: z
Subject to: z <= (allocation[g] / burden[g]) for all groups g
```
- Maxes the **minimum** coverage % across all demographic groups
- Rawlsian: helps the worst-off

#### 4c. **Proportional Fairness** (`proportional`)
```
Constraints: allocation[i] >= budget * (burden[i] / total_burden)
```
- Each region gets at least its "fair share" of budget
- Fair ≠ equal (per-capita doesn't matter, total burden does)

**Implementation tasks:**
- `add_weighted_log_objective()` – Decide: CVXPY or piecewise LP
- `add_max_min_fairness()` – Introduce slack variable `z`, add constraints
- `add_proportional_fairness()` – For each demographic group, calculate its burden share, enforce floor
- `add_demographic_constraints()` – Layer on user's minimum allocations (e.g., "≥30% rural")

---

### 5. `FairnessMetrics`

**What it does:**
Compute 4 key metrics to evaluate any allocation solution.

**Implement each:**

1. **`total_lives_impacted(allocation, problem)`**
   - Sum of: allocation[i] × count[i] across all demographic groups
   - High = efficient, but might be inequitable

2. **`gini_coefficient(allocation, problem)`**
   - Standard Gini: 0 = perfect equality, 1 = maximum inequality
   - For budget allocation: measure inequality in per-capita coverage
   - Formula: See economics textbooks; `numpy` helpers exist

3. **`max_min_ratio(allocation, problem)`**
   - Ratio: highest coverage % / lowest coverage %
   - 1.0 = perfectly equal (impossible)
   - 2.0 = worst-off region gets half the resources per-capita of best-off
   - Rawlsian preference: minimize this

4. **`proportionality_violation(allocation, problem)`**
   - How much does allocation deviate from burden-proportional?
   - Compute max overage as % of budget

---

### 6. `ParetoBoundary`

**What it does:**
Systematically solve under different fairness modes and budgets, then plot efficiency vs. equity.

**To implement:**
- `generate_solutions()` – For each combo of (fairness_mode, budget_fraction):
  1. Create a StakeholderPreferences with that mode
  2. Solve using FairnessOptimizer
  3. Compute total_lives_impacted + gini_coefficient
  4. Store (efficiency, equity, mode_name, budget_size)

- `plot()` – Scatter plot:
  - X-axis = efficiency (lives impacted)
  - Y-axis = gini or max_min_ratio (equity metric)
  - Color by fairness mode
  - Bubble size by budget level

**Interpretation:**
- Points on the left = worse efficiency but more fair
- Points on the right = higher efficiency but less fair
- Frontier = can't improve both without picking a different fairness concept

---

### 7. `AllocationEngine` Orchestration

**What it does:**
Ties everything together in a clean workflow.

**Implementation:**
- `__init__()` – Load CSV, wrap in AllocationProblem
- `run_baseline()` – UtilitarianOptimizer().solve()
- `run_fairness(pref)` – FairnessOptimizer(...).solve()
- `compare_solutions()` – Compute metrics for both, print side-by-side
- `generate_pareto_frontier()` – Call ParetoBoundary

---

## Implementation Roadmap

### Priority 1: Foundation (Week 1)
- [ ] `AllocationProblem.__init__()` ✓ (already defined)
- [ ] `UtilitarianOptimizer.solve()` ← **Start here**
  - Pick PuLP as default (already in requirements)
  - Debug with a small 5-country subset first
- [ ] `FairnessMetrics.total_lives_impacted()` ← Second
- [ ] `AllocationEngine.run_baseline()` ← Full end-to-end pipeline

### Priority 2: Fairness Models (Week 2)
- [ ] `FairnessOptimizer.add_proportional_fairness()`
- [ ] `FairnessOptimizer.add_max_min_fairness()`
- [ ] `FairnessMetrics.gini_coefficient()` + `max_min_ratio()`
- [ ] `AllocationEngine.run_fairness()` + `compare_solutions()`

### Priority 3: Advanced (Week 3)
- [ ] `add_weighted_log_objective()` – Use CVXPY for convex solver
- [ ] `ParetoBoundary.generate_solutions()` + `plot()`
- [ ] `PreferenceElicitor.from_json()` + validation

### Priority 4: Polish (Week 4)
- [ ] Export solutions to CSV
- [ ] Add sensitivity analysis (what if budget ±10%?)
- [ ] Documentation + example outputs

---

## Quick Start: Fill-in Template

For each `pass` statement, replace with implementation logic. Here's a pattern:

```python
def add_objective(self):
    """Objective: maximize total children treated."""
    # Step 1: Identify decision variables
    # Step 2: Identify objective coefficients (count per child)
    # Step 3: Build lpSum and assign to model.objective
    
    from pulp import lpSum
    objective = lpSum([
        self.allocation_vars[key] * self.problem.data.loc[key, 'Count_Stunting']
        for key in self.allocation_vars
    ])
    self.model += objective
```

---

## Dependencies

Already in `requirements.txt`:
- `pandas` – data handling
- `pulp` – linear programming (install CBC solver separately if needed)
- `matplotlib` / `seaborn` – plotting

For weighted log-utility (optional):
- `cvxpy` – convex optimization (handles log natively)
- `scipy` – nonlinear optimization fallback

---

## Testing & Debugging

Start small:
```python
# Use only 2 countries, 3 demographic groups
small_problem = AllocationProblem(
    data=engine.data[engine.data['ISO3'].isin(['AFG', 'MLI'])],
    total_budget=1_000_000
)
baseline = UtilitarianOptimizer(small_problem).solve()
```

Verify:
- Model created without errors
- Variables + constraints printed
- Solver returns a status (optimal, infeasible, etc.)
- Solution sums parse correctly

---

## Next Steps

1. **Copy this boilerplate into your `src/` folder**
2. **Start with `UtilitarianOptimizer`** – most straightforward
3. **Test on a small data subset**
4. **Add fairness models one at a time**
5. **Generate a simple Pareto plot**

Good luck! Let me know if you hit any issues with PuLP/CVXPY integration.
