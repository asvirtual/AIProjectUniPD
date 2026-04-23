"""
Fairness Metrics Tests - Direct Execution with Demo Results

Tests cover:
- Validation helpers (_allocation_df, _base_problem_df, _merged_allocation_view)
- Fairness metrics (gini_coefficient, max_min_ratio, demographic_coverage_gap)
- Efficiency metrics (total_lives_impacted)
- Error handling and edge cases

Executes tests directly using demo data instead of pytest.
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path
from optimization_engine import AllocationProblem, UtilitarianOptimizer, ConstrainedUtilitarianOptimizer, FairnessMetrics, COUNT_COLS, INTERVENTION_TYPES


# ============================================================================
# TEST UTILITIES
# ============================================================================

def assert_equal(actual, expected, msg=""):
    """Assert equality with custom message."""
    if actual != expected:
        raise AssertionError(f"{msg}\nExpected: {expected}\nActual: {actual}")

def assert_true(condition, msg=""):
    """Assert condition is true."""
    if not condition:
        raise AssertionError(f"Assertion failed: {msg}")

def assert_isinstance(obj, cls, msg=""):
    """Assert object is instance of class."""
    if not isinstance(obj, cls):
        raise AssertionError(f"{msg}\nExpected type: {cls}\nActual type: {type(obj)}")

def assert_isclose(a, b, tol=1e-10, msg=""):
    """Assert two numbers are close."""
    if not math.isclose(a, b, abs_tol=tol):
        raise AssertionError(f"{msg}\nExpected: {b} ± {tol}\nActual: {a}")

def assert_greater(a, b, msg=""):
    """Assert a > b."""
    if a <= b:
        raise AssertionError(f"{msg}\nExpected > {b}\nActual: {a}")

def assert_greater_equal(a, b, msg=""):
    """Assert a >= b."""
    if a < b:
        raise AssertionError(f"{msg}\nExpected >= {b}\nActual: {a}")

def assert_raises(exc_type, func, *args, **kwargs):
    """Assert that calling func raises exc_type."""
    try:
        func(*args, **kwargs)
        raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised")
    except exc_type:
        pass  # Expected
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__} but got {type(e).__name__}: {e}")

def print_test(name, passed):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")

def run_test(test_func):
    """Run a test and catch exceptions."""
    try:
        test_func()
        print_test(test_func.__name__, True)
        return True
    except Exception as e:
        print_test(test_func.__name__, False)
        print(f"         Error: {e}")
        return False


# ============================================================================
# TEST DATA FIXTURES (replacing pytest fixtures)
# ============================================================================

def setup_sample_allocation():
    """Create a sample allocation dict for testing."""
    return {
        "status": "Optimal",
        "total_spend": 1000,
        "total_treated": 500,
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Wealth Quintile 1 Rural", "Wealth Quintile 5 Urban"],
            "total_treated": [100, 400],
            "total_spend": [400, 600]
        })
    }

def setup_sample_problem():
    """Create a sample AllocationProblem for testing."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Wealth Quintile 1 Rural", "Wealth Quintile 5 Urban"],
        "Count_stunting": [500, 100],
        "Count_wasting": [300, 50],
        "Count_severe_wasting": [200, 30],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 2000]
    })
    return AllocationProblem(df=data, total_budget=1000)

def setup_allocation_and_problem_lives():
    """Setup for total_lives_impacted tests."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG", "PAK"],
        "Country": ["Afghanistan", "Afghanistan", "Pakistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban", "Q1 Rural"],
        "Count_stunting": [500, 100, 300],
        "Count_wasting": [300, 50, 200],
        "Count_severe_wasting": [200, 30, 100],
        "Cost_stunting": [20, 15, 18],
        "Cost_wasting": [25, 20, 22],
        "Cost_severe_wasting": [30, 25, 28],
        "Population_u5": [5000, 2000, 4000]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG", "PAK"],
            "Country": ["Afghanistan", "Afghanistan", "Pakistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban", "Q1 Rural"],
            "total_treated": [250, 100, 150],
            "total_spend": [3000, 1500, 2500]
        })
    }
    return allocation, problem

def setup_equal_coverage():
    """Create allocation with equal coverage across groups."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [1000, 1000],
        "Count_wasting": [1000, 1000],
        "Count_severe_wasting": [1000, 1000],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 5000]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [1000, 1000],
            "total_spend": [5000, 5000]
        })
    }
    return allocation, problem

def setup_unequal_coverage():
    """Create allocation with unequal coverage across groups."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [1000, 1000],
        "Count_wasting": [1000, 1000],
        "Count_severe_wasting": [1000, 1000],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 5000]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [100, 1900],  # Very unequal
            "total_spend": [1000, 9000]
        })
    }
    return allocation, problem

def setup_equal_ratio():
    """Create allocation with equal coverage (ratio = 1)."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [1000, 1000],
        "Count_wasting": [1000, 1000],
        "Count_severe_wasting": [1000, 1000],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 5000]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [1000, 1000],
            "total_spend": [5000, 5000]
        })
    }
    return allocation, problem

def setup_unequal_ratio():
    """Create allocation with unequal coverage ratio."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [1000, 1000],
        "Count_wasting": [1000, 1000],
        "Count_severe_wasting": [1000, 1000],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 5000]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [100, 1000],  # 1 vs 10 times their need
            "total_spend": [1000, 9000]
        })
    }
    return allocation, problem

def setup_country_gap():
    """Create allocation with coverage gap between groups in a country."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG", "AFG", "AFG"],
        "Country": ["Afghanistan"] * 4,
        "Demographic_group": ["Q1 Rural", "Q1 Urban", "Q5 Rural", "Q5 Urban"],
        "Count_stunting": [500, 500, 100, 100],
        "Count_wasting": [300, 300, 50, 50],
        "Count_severe_wasting": [200, 200, 30, 30],
        "Cost_stunting": [20, 15, 10, 8],
        "Cost_wasting": [25, 20, 12, 10],
        "Cost_severe_wasting": [30, 25, 15, 12],
        "Population_u5": [5000, 5000, 2000, 2000]
    })
    problem = AllocationProblem(df=data, total_budget=50000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG", "AFG", "AFG"],
            "Country": ["Afghanistan"] * 4,
            "Demographic_group": ["Q1 Rural", "Q1 Urban", "Q5 Rural", "Q5 Urban"],
            "total_treated": [300, 400, 100, 100],  # Gap between Q1 Rural (0.6) and Q1 Urban (0.8)
            "total_spend": [5000, 6000, 1000, 1000]
        })
    }
    return allocation, problem

def setup_summary_test():
    """Setup for build_summary tests."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [1000, 1000],
        "Count_wasting": [1000, 1000],
        "Count_severe_wasting": [1000, 1000],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 5000]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "status": "Optimal",
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [500, 500],
            "total_spend": [5000, 5000]
        })
    }
    return allocation, problem

def setup_zero_need_row():
    """Setup for edge case with zero need row."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [1000, 0],  # Second group has zero need
        "Count_wasting": [1000, 0],
        "Count_severe_wasting": [1000, 0],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 0]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [500, 0],
            "total_spend": [5000, 0]
        })
    }
    return allocation, problem



# ============================================================================
# VALIDATORS TESTS
# ============================================================================

def test_allocation_df_valid():
    """Test _allocation_df with valid input."""
    sample_allocation = setup_sample_allocation()
    result = FairnessMetrics._allocation_df(sample_allocation)
    assert_isinstance(result, pd.DataFrame, "_allocation_df should return DataFrame")
    assert_true(not result.empty, "_allocation_df result should not be empty")
    assert_equal(list(result.columns), ["ISO3", "Country", "Demographic_group", "total_treated", "total_spend"], "Columns mismatch")

def test_allocation_df_not_dict():
    """Test _allocation_df raises ValueError for non-dict input."""
    assert_raises(ValueError, FairnessMetrics._allocation_df, "not a dict")

def test_allocation_df_missing_allocation_df_key():
    """Test _allocation_df raises ValueError when 'allocation_df' key is missing."""
    assert_raises(ValueError, FairnessMetrics._allocation_df, {"status": "Optimal"})

def test_allocation_df_empty():
    """Test _allocation_df raises ValueError for empty DataFrame."""
    assert_raises(ValueError, FairnessMetrics._allocation_df, {"allocation_df": pd.DataFrame()})

def test_base_problem_df_valid():
    """Test _base_problem_df with valid AllocationProblem."""
    sample_problem = setup_sample_problem()
    result = FairnessMetrics._base_problem_df(sample_problem)
    assert_isinstance(result, pd.DataFrame, "_base_problem_df should return DataFrame")
    assert_true(not result.empty, "_base_problem_df result should not be empty")

def test_base_problem_df_no_filtered_df():
    """Test _base_problem_df raises ValueError when filtered_df is missing."""
    class BadProblem:
        pass
    assert_raises(ValueError, FairnessMetrics._base_problem_df, BadProblem())

def test_merged_allocation_view_valid():
    """Test _merged_allocation_view with valid inputs."""
    sample_allocation = setup_sample_allocation()
    sample_problem = setup_sample_problem()
    result = FairnessMetrics._merged_allocation_view(sample_allocation, sample_problem)
    assert_isinstance(result, pd.DataFrame, "Should return DataFrame")
    assert_true("need_total" in result.columns, "Should have need_total column")
    assert_true("coverage_ratio" in result.columns, "Should have coverage_ratio column")
    assert_equal(len(result), 2, "Should have 2 rows")
    assert_equal(result.iloc[0]["need_total"], 500 + 300 + 200, "need_total calculation incorrect")

def test_merged_allocation_view_missing_columns():
    """Test _merged_allocation_view raises ValueError for missing allocation columns."""
    sample_problem = setup_sample_problem()
    bad_allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG"],
            "Country": ["Afghanistan"],
        })
    }
    assert_raises(ValueError, FairnessMetrics._merged_allocation_view, bad_allocation, sample_problem)

# ============================================================================
# GINI COEFFICIENT TESTS
# ============================================================================

def test_gini_perfect_equality():
    """Test Gini coefficient for perfect equality (all values equal)."""
    values = [1.0, 1.0, 1.0, 1.0]
    result = FairnessMetrics._gini(values)
    assert_isclose(result, 0.0, msg="Gini for equal values should be 0")

def test_gini_perfect_inequality():
    """Test Gini coefficient for maximum inequality."""
    values = [0.0, 0.0, 0.0, 1.0]
    result = FairnessMetrics._gini(values)
    assert_greater(result, 0.5, "Gini for extreme inequality should be > 0.5")

def test_gini_empty_list():
    """Test Gini coefficient for empty list."""
    result = FairnessMetrics._gini([])
    assert_equal(result, 0.0, "Gini for empty list should be 0")

def test_gini_all_zeros():
    """Test Gini coefficient for all zeros."""
    result = FairnessMetrics._gini([0.0, 0.0, 0.0])
    assert_equal(result, 0.0, "Gini for all zeros should be 0")

def test_gini_negative_values_raises():
    """Test Gini coefficient raises ValueError for negative values."""
    assert_raises(ValueError, FairnessMetrics._gini, [1.0, -1.0, 0.5])

def test_gini_with_nan_values():
    """Test Gini coefficient ignores NaN values."""
    values = [1.0, 2.0, np.nan, 3.0]
    result = FairnessMetrics._gini(values)
    expected = FairnessMetrics._gini([1.0, 2.0, 3.0])
    assert_isclose(result, expected, msg="Gini should handle NaN values")



# ============================================================================
# TOTAL LIVES IMPACTED TESTS
# ============================================================================

def test_total_lives_impacted_sum():
    """Test total_lives_impacted correctly sums across all rows."""
    allocation, problem = setup_allocation_and_problem_lives()
    result = FairnessMetrics.total_lives_impacted(allocation, problem)
    expected = 250 + 100 + 150
    assert_equal(result, expected, "total_lives_impacted should sum all treated")

def test_total_lives_impacted_with_nan():
    """Test total_lives_impacted handles NaN values."""
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [100, np.nan],
            "total_spend": [500, 1000]
        })
    }
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [500, 100],
        "Count_wasting": [300, 50],
        "Count_severe_wasting": [200, 30],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 2000]
    })
    problem = AllocationProblem(df=data, total_budget=1000)
    result = FairnessMetrics.total_lives_impacted(allocation, problem)
    assert_equal(result, 100.0, "Should handle NaN values")

# ============================================================================
# GINI COEFFICIENT METRIC TESTS
# ============================================================================

def test_gini_equal_coverage():
    """Test Gini coefficient for equal coverage is near 0."""
    allocation, problem = setup_equal_coverage()
    result = FairnessMetrics.gini_coefficient(allocation, problem)
    assert_isclose(result, 0.0, msg="Equal coverage should have Gini near 0")

def test_gini_unequal_coverage():
    """Test Gini coefficient for unequal coverage is > 0."""
    allocation, problem = setup_unequal_coverage()
    result = FairnessMetrics.gini_coefficient(allocation, problem)
    assert_greater(result, 0.5, "Unequal coverage should have high Gini")



# ============================================================================
# MAX MIN RATIO TESTS
# ============================================================================

def test_max_min_ratio_equal():
    """Test max_min_ratio is 1 for equal coverage."""
    allocation, problem = setup_equal_ratio()
    result = FairnessMetrics.max_min_ratio(allocation, problem)
    assert_isclose(result, 1.0, msg="Equal ratio should be 1")

def test_max_min_ratio_unequal():
    """Test max_min_ratio > 1 for unequal coverage."""
    allocation, problem = setup_unequal_ratio()
    result = FairnessMetrics.max_min_ratio(allocation, problem)
    assert_greater(result, 1.0, "Unequal ratio should be > 1")



# ============================================================================
# DEMOGRAPHIC COVERAGE GAP TESTS
# ============================================================================

def test_demographic_coverage_gap_structure():
    """Test demographic_coverage_gap returns correct structure."""
    allocation, problem = setup_country_gap()
    result = FairnessMetrics.demographic_coverage_gap(allocation, problem)
    
    assert_isinstance(result, dict, "Should return dict")
    assert_true("per_country" in result, "Should have per_country key")
    assert_true("weighted_mean_gap" in result, "Should have weighted_mean_gap key")
    assert_true("unweighted_mean_gap" in result, "Should have unweighted_mean_gap key")
    assert_isinstance(result["per_country"], dict, "per_country should be dict")

def test_demographic_coverage_gap_values():
    """Test demographic_coverage_gap calculates reasonable gap values."""
    allocation, problem = setup_country_gap()
    result = FairnessMetrics.demographic_coverage_gap(allocation, problem)
    
    assert_true("AFG" in result["per_country"], "Should have AFG in per_country")
    assert_true("max_gap" in result["per_country"]["AFG"], "Should have max_gap")
    assert_true("best_group" in result["per_country"]["AFG"], "Should have best_group")
    assert_true("worst_group" in result["per_country"]["AFG"], "Should have worst_group")
    
    assert_greater_equal(result["per_country"]["AFG"]["max_gap"], 0, "Gap should be non-negative")
    assert_greater_equal(result["weighted_mean_gap"], 0, "Mean gap should be non-negative")
    assert_greater_equal(result["unweighted_mean_gap"], 0, "Unweighted mean gap should be non-negative")



# ============================================================================
# BUILD SUMMARY TESTS
# ============================================================================

def test_build_summary_structure():
    """Test build_summary returns all expected metrics."""
    allocation, problem = setup_summary_test()
    result = FairnessMetrics.build_summary(allocation, problem, label="test_solution")
    
    assert_isinstance(result, dict, "Should return dict")
    assert_true("label" in result, "Should have label key")
    assert_true("total_treated" in result, "Should have total_treated key")
    assert_true("gini_coefficient" in result, "Should have gini_coefficient key")
    assert_true("max_min_ratio" in result, "Should have max_min_ratio key")
    assert_true("demographic_coverage_gap" in result, "Should have demographic_coverage_gap key")
    assert_equal(result["label"], "test_solution", "Label should match")

def test_build_summary_values():
    """Test build_summary calculates reasonable metric values."""
    allocation, problem = setup_summary_test()
    result = FairnessMetrics.build_summary(allocation, problem)
    
    assert_equal(result["total_treated"], 1000.0, "total_treated should be 1000")
    assert_true(0 <= result["gini_coefficient"] <= 1, "gini_coefficient should be 0-1")
    assert_greater_equal(result["max_min_ratio"], 1.0, "max_min_ratio should be >= 1")

# ============================================================================
# EDGE CASES TESTS
# ============================================================================

def test_zero_need_row_excluded_from_gini():
    """Test rows with zero need are handled correctly in Gini calculation."""
    allocation, problem = setup_zero_need_row()
    
    result = FairnessMetrics.gini_coefficient(allocation, problem)
    assert_isinstance(result, float, "Should return float")

def test_max_min_ratio_with_zero_coverage():
    """Test max_min_ratio when min coverage is zero."""
    data = pd.DataFrame({
        "ISO3": ["AFG", "AFG"],
        "Country": ["Afghanistan", "Afghanistan"],
        "Demographic_group": ["Q1 Rural", "Q5 Urban"],
        "Count_stunting": [1000, 1000],
        "Count_wasting": [1000, 1000],
        "Count_severe_wasting": [1000, 1000],
        "Cost_stunting": [20, 15],
        "Cost_wasting": [25, 20],
        "Cost_severe_wasting": [30, 25],
        "Population_u5": [5000, 5000]
    })
    problem = AllocationProblem(df=data, total_budget=10000)
    allocation = {
        "allocation_df": pd.DataFrame({
            "ISO3": ["AFG", "AFG"],
            "Country": ["Afghanistan", "Afghanistan"],
            "Demographic_group": ["Q1 Rural", "Q5 Urban"],
            "total_treated": [0, 1000],  # Zero coverage for first group
            "total_spend": [0, 10000]
        })
    }
    
    result = FairnessMetrics.max_min_ratio(allocation, problem)
    assert_true(math.isinf(result), "Should return infinity for zero coverage")


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    """Execute all tests and generate demo results with fairness metrics."""
    
    # Load demo data
    data_path = Path(__file__).parent.parent / "data" / "processed" / "master_df_mece_compliant.csv"
    df = pd.read_csv(data_path)
    problem = AllocationProblem(df=df, total_budget=100_000_000)
    
    print("=" * 80)
    print("FAIRNESS METRICS TEST SUITE - Direct Execution with Demo Results")
    print("=" * 80)
    
    # ========================================================================
    # UNIT TESTS - All test functions
    # ========================================================================
    all_tests = [
        # Validators
        ("Validators", [
            test_allocation_df_valid,
            test_allocation_df_not_dict,
            test_allocation_df_missing_allocation_df_key,
            test_allocation_df_empty,
            test_base_problem_df_valid,
            test_base_problem_df_no_filtered_df,
            test_merged_allocation_view_valid,
            test_merged_allocation_view_missing_columns,
        ]),
        # Gini Calculation
        ("Gini Coefficient (Static)", [
            test_gini_perfect_equality,
            test_gini_perfect_inequality,
            test_gini_empty_list,
            test_gini_all_zeros,
            test_gini_negative_values_raises,
            test_gini_with_nan_values,
        ]),
        # Total Lives Impacted
        ("Total Lives Impacted", [
            test_total_lives_impacted_sum,
            test_total_lives_impacted_with_nan,
        ]),
        # Gini Coefficient Metric
        ("Gini Coefficient Metric", [
            test_gini_equal_coverage,
            test_gini_unequal_coverage,
        ]),
        # Max Min Ratio
        ("Max Min Ratio", [
            test_max_min_ratio_equal,
            test_max_min_ratio_unequal,
        ]),
        # Demographic Coverage Gap
        ("Demographic Coverage Gap", [
            test_demographic_coverage_gap_structure,
            test_demographic_coverage_gap_values,
        ]),
        # Build Summary
        ("Build Summary", [
            test_build_summary_structure,
            test_build_summary_values,
        ]),
        # Edge Cases
        ("Edge Cases", [
            test_zero_need_row_excluded_from_gini,
            test_max_min_ratio_with_zero_coverage,
        ]),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_group_name, tests in all_tests:
        print(f"\n{test_group_name}:")
        group_passed = 0
        group_failed = 0
        
        for test_func in tests:
            if run_test(test_func):
                group_passed += 1
                total_passed += 1
            else:
                group_failed += 1
                total_failed += 1
        
        print(f"  {group_passed}/{group_passed + group_failed} passed")
    
    # ========================================================================
    # DEMO SCENARIOS WITH FAIRNESS METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("DEMO SCENARIOS WITH FAIRNESS METRICS")
    print("=" * 80)
    
    try:
        # SCENARIO 1: Baseline
        print("\n--- SCENARIO 1: Baseline Utilitarian (No Constraints) ---")
        solution_baseline = UtilitarianOptimizer(problem).solve()
        summary_baseline = FairnessMetrics.build_summary(solution_baseline, problem, label="Baseline")
        print(f"Children treated: {summary_baseline['total_treated']:,.0f}")
        print(f"Gini coefficient: {summary_baseline['gini_coefficient']:.4f}")
        print(f"Max/Min ratio: {summary_baseline['max_min_ratio']:.4f}")
        
        # SCENARIO 2: Country constrained
        print("\n--- SCENARIO 2: Country-Constrained (10% cap) ---")
        solution_constrained = ConstrainedUtilitarianOptimizer(problem, country_cap=0.1).solve()
        summary_constrained = FairnessMetrics.build_summary(solution_constrained, problem, label="Country-Constrained")
        print(f"Children treated: {summary_constrained['total_treated']:,.0f}")
        print(f"Gini coefficient: {summary_constrained['gini_coefficient']:.4f}")
        print(f"Max/Min ratio: {summary_constrained['max_min_ratio']:.4f}")
        
        # SCENARIO 3: Demographic constrained
        print("\n--- SCENARIO 3: Country + Demographic Constrained ---")
        solution_demographic = ConstrainedUtilitarianOptimizer(
            problem,
            country_cap=0.20,
            demographic_cap=0.50,
            demographic_min_share={"Rural": 0.40, "Wealth Quintile 1": 0.20}
        ).solve()
        summary_demographic = FairnessMetrics.build_summary(solution_demographic, problem, label="Demographic-Constrained")
        print(f"Children treated: {summary_demographic['total_treated']:,.0f}")
        print(f"Gini coefficient: {summary_demographic['gini_coefficient']:.4f}")
        print(f"Max/Min ratio: {summary_demographic['max_min_ratio']:.4f}")
        
        print("\n" + "=" * 80)
        print(f"TEST SUMMARY: {total_passed} passed, {total_failed} failed")
        print("=" * 80)
        
        if total_failed == 0:
            print("\n✓ ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n✗ {total_failed} TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR running demo scenarios: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
