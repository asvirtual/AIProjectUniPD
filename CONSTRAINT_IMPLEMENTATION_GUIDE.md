# Constraint Satisfaction Optimization Engine: Implementation Guide

## Key Differences: `python-constraint` vs. `PuLP`

### python-constraint (CSP Solver)
- **Best for**: Discrete, categorical decisions with complex logical constraints
- **How it works**: Enumerate feasible solutions, filter by constraints
- **Strengths**: Intuitive preference logic, easy to add custom constraints
- **Weakness**: Not optimized for continuous optimization (no linear algebra); slower on large problems

### PuLP (Linear Programming)
- **Best for**: Continuous optimization with linear objectives/constraints
- **How it works**: Simplex method; mathematically guaranteed optimal
- **Strengths**: Fast, scalable, proven theory
- **Weakness**: Can't easily express fairness concepts like "max-min" without reformulation

---

## Architecture Overview

```
python-constraint Problem
├── Variables: allocation_AFG_Rural_Stunting, etc.
│   └── Domain: [0, 1, 2, ..., max_units] (discretized amounts)
├── Constraints: budget_ok(), proportional_ok(), max_min_ok(), etc.
│   └── Each returns True/False for a candidate solution
└── getSolutions(): returns list of all feasible assignments
    └── Engine selects solution with highest objective
```

**Key insight**: Unlike PuLP which solves "in one pass," constraint satisfaction:
1. Generates many candidate solutions
2. Tests each against all constraints
3. Returns list of feasible solutions
4. You pick the best one

---

## Implementation Details

### 1. Variable Setup: Discretization

```python
def setup_variables(self):
    max_units = int(self.problem.total_budget / self.problem.discretization_unit)
    # e.g., $100M budget / $100k unit = 1000 units max
    
    for idx, row in self.problem.data.iterrows():
        var_name = f"alloc_{row['ISO3']}_{row['Demographic_Group']}"
        # Domain: [0, 1, 2, ..., 1000] = discrete allocations in $100k chunks
        problem.addVariable(var_name, list(range(0, max_units + 1)))
```

**Trade-off**:
- Larger `discretization_unit` (e.g., $1M) → fewer variables → faster
- Smaller `discretization_unit` (e.g., $10k) → more variables → slower, finer granularity

**Example**: $100M budget with $100k unit = 1000 possible values per variable
- With 200 demographic groups = **1000^200 candidate solutions** (infeasible!)
- **Solution**: Use intelligent constraints to prune search space

---

### 2. Constraint Functions

Each constraint is a Python function returning True/False:

```python
def budget_ok(*allocations):
    """Check if total cost <= budget."""
    total_cost = sum(
        allocations[i] * discretization_unit * cost[i]
        for i in range(len(allocations))
    )
    return total_cost <= budget

# Add to problem:
problem.addConstraint(budget_ok, list_of_variable_names)
```

**Important**: Order of variables in the function signature must match order of variable names in `addConstraint()`.

---

### 3. Fairness Constraints

#### Proportional Fairness
```python
def proportional_ok(*allocations):
    """Ensure allocation >= burden share."""
    total_burden = sum of all burdens
    for i in range(len(allocations)):
        burden_share = burden[i] / total_burden
        min_alloc_units = int((budget * burden_share) / discretization_unit)
        if allocations[i] < min_alloc_units:
            return False
    return True
```

#### Max-Min Fairness
```python
def max_min_ok(*allocations):
    """Maximize the minimum allocation (all regions must meet floor)."""
    min_coverage = 0.10  # 10% coverage
    for i in range(len(allocations)):
        coverage = allocations[i] * discretization_unit / burden[i]
        if coverage < min_coverage:
            return False
    return True
```

**Note**: In CSP, you can't directly "maximize z subject to z <= allocations[i]". Instead, you iteratively increase the floor and re-solve.

---

### 4. Objective Selection

After `getSolutions()`, pick the best solution:

```python
solutions = problem.getSolutions()  # List of dicts: {var_name: value, ...}

# Select solution with max total children treated
best = max(
    solutions,
    key=lambda sol: sum(sol[var] * discretization_unit * count[var] for var in variables)
)
```

---

## Implementation Roadmap

### Priority 1: Get baseline working (Week 1)
- [x] `AllocationProblem.__init__()` ← Already done
- [ ] `UtilitarianOptimizer.setup_variables()` ← **Start here**
  - Create variables with discrete domains
  - Test on a 5-country subset
  
- [ ] `UtilitarianOptimizer.add_budget_constraint()` ← Second
  - Implement `budget_ok()` function
  - Add to problem
  
- [ ] `UtilitarianOptimizer.solve()` ← Full pipeline
  - Call `problem.getSolutions()`
  - Select best solution
  - Debug: Print solution size, timing, etc.

### Priority 2: Add fairness models (Week 2)
- [ ] `FairnessOptimizer.add_proportional_fairness_constraint()`
- [ ] `FairnessOptimizer.add_max_min_fairness_constraint()`
- [ ] `FairnessMetrics.total_lives_impacted()` + `gini_coefficient()`
- [ ] `AllocationEngine.compare_solutions()`

### Priority 3: Visualization (Week 3)
- [ ] `ParetoBoundary.generate_solutions()` – Solve for different modes/budgets
- [ ] `ParetoBoundary.plot()` – Scatter plot: efficiency vs. equity
- [ ] Export results to CSV

---

## Performance Considerations

**Problem Size Effects**:
- 50 countries × 15 demographic groups = 750 variables
- Each variable: 0-1000 units domain
- Naive solver: 1000^750 search space (impossible!)

**Solutions**:

1. **Reduce domain size**: Use larger discretization units
   ```python
   AllocationEngine(..., discretization_unit=500_000)  # $500k chunks
   ```

2. **Add tighter constraints early**:
   - Budget constraint prunes heavily
   - Demographic floors reduce feasibility
   - Start with conservative bounds

3. **Iterative solving**: Solve with relaxed fairness, then tighten
   ```python
   # Step 1: Solve baseline
   baseline = UtilitarianOptimizer(problem).solve()
   
   # Step 2: Solve with proportional fairness (more constrained)
   fair_pref = StakeholderPreferences(..., fairness_mode='proportional')
   fair = FairnessOptimizer(problem, fair_pref).solve()
   ```

4. **Monitor solver time**:
   ```python
   import time
   start = time.time()
   solutions = problem.getSolutions()
   print(f"Found {len(solutions)} solutions in {time.time() - start:.2f}s")
   ```

---

## Testing

### Small Example (Debug-Friendly)

```python
# Use only 2 countries, 2 demographic groups
small_data = engine.data[
    (engine.data['ISO3'].isin(['AFG', 'MLI'])) &
    (engine.data['Demographic_Group'].isin(['National', 'Rural']))
]

small_problem = AllocationProblem(
    data=small_data,
    total_budget=10_000_000,  # $10M (smaller)
    discretization_unit=500_000  # $500k (larger chunks = fewer variables)
)

optimizer = UtilitarianOptimizer(small_problem)
solution = optimizer.solve()

print(f"Solution: {solution}")
print(f"Budget used: {sum(solution.values()) * small_problem.discretization_unit:,.0f}")
```

### Verify Constraints

```python
# Check all solutions respect budget
from optimization_engine_constraint import FairnessOptimizer

for sol in solutions:
    total_cost = sum(
        sol[var] * problem.discretization_unit * var_cost[var]
        for var in sol
    )
    assert total_cost <= problem.total_budget, f"Budget violated: {total_cost}"

print(f"✓ All {len(solutions)} solutions respect budget")
```

---

## Advantages of This Approach

1. **Intuitive fairness logic**: Constraints map directly to fairness concepts
2. **Transparent**: Can inspect all feasible solutions, not just optimal
3. **Flexible**: Easy to add custom constraint functions
4. **Interpretable**: Each constraint has a clear business meaning

---

## Next Steps

1. **Install python-constraint**: `pip install python-constraint`
2. **Start with `UtilitarianOptimizer`**: simplest constraint set
3. **Test on small subset** (2 countries)
4. **Add fairness constraints one by one**
5. **Monitor solver time** and adjust discretization if needed
6. **Generate Pareto frontier** once all modes work

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| No solutions found | Constraints too tight | Relax demographic floors or budget |
| Solver takes hours | Too many variables | Increase `discretization_unit` |
| Variables named weird | Forgot `_prepare_data()` | Ensure all data cols are strings |
| Metric values = 0 | Didn't populate `counts` correctly | Load CSV and check Count_Stunting column |

---

## Code Template: Fill-in the `pass` Statements

```python
def setup_variables(self):
    """Create decision variables with discrete domains."""
    max_units = int(self.problem.total_budget / self.problem.discretization_unit)
    
    for idx, row in self.problem.data.iterrows():
        var_name = f"alloc_{row['ISO3']}_{row['Demographic_Group']}"
        # TODO: Store cost, count, burden in self.allocation_vars
        self.allocation_vars[var_name] = {
            'cost': row.get('Cost_stunting', 0),
            'count': row.get('Count_Stunting', 0),
            'domain': list(range(0, max_units + 1))
        }
```

Good luck! Message me if solver times are too high or constraints fail.
