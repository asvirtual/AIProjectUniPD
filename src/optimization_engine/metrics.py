# ============================================================================
# Fairness metrics for evaluating allocation outcomes
# ============================================================================

import math
from typing import Any, Dict, List

import pandas as pd

from .models import (
    AllocationProblem,
    INTERVENTION_TYPES,
    COUNT_COLS,
    COUNT_SCALE_FACTOR,
    COST_SCALE_FACTOR,
    BUDGET_SCALE_FACTOR,
)


# ============================================================================
# Compute coverage, Gini coefficient, max-min ratio and other fairness metrics
# ===========================================================================

class FairnessMetrics:
    """Metrics and comparison helpers for baseline vs fairness allocations."""

    KEY_COLS = ["ISO3", "Country", "Demographic_group"]

    # Validate the optimizer output and return its allocation DataFrame.
    # Input: allocation result dict.
    # Output: a non-empty copy of allocation_df.
    # Purpose: ensure downstream metrics always work on a valid allocation table.
    @staticmethod
    def _allocation_df(allocation: Dict) -> pd.DataFrame:
        if not isinstance(allocation, dict):
            raise ValueError("allocation must be a dictionary result returned by an optimizer.")
        allocation_df = allocation.get("allocation_df")
        if allocation_df is None or not isinstance(allocation_df, pd.DataFrame):
            raise ValueError("allocation must contain an 'allocation_df' pandas DataFrame.")
        if allocation_df.empty:
            raise ValueError("allocation_df is empty.")
        return allocation_df.copy()

    # Validate the original problem data and return the filtered base DataFrame.
    # Input: AllocationProblem instance.
    # Output: a non-empty copy of problem.filtered_df.
    # Purpose: provide the reference data needed to evaluate an allocation.
    @staticmethod
    def _base_problem_df(problem: AllocationProblem) -> pd.DataFrame:
        if not hasattr(problem, "filtered_df"):
            raise ValueError("problem must expose a filtered_df attribute.")
        base_df = problem.filtered_df.copy()
        if base_df.empty:
            raise ValueError("problem.filtered_df is empty.")
        return base_df

    # Build the main evaluation view by merging base data with allocation results.
    # Input: optimizer result dict + AllocationProblem.
    # Output: merged DataFrame with need, treated, spend, and coverage ratio.
    # Purpose: create the common table used by most fairness metrics.
    @classmethod
    def _merged_allocation_view(cls, allocation: Dict, problem: AllocationProblem) -> pd.DataFrame:
        allocation_df = cls._allocation_df(allocation)
        base_df = cls._base_problem_df(problem)

        required_alloc_cols = set(cls.KEY_COLS + ["total_treated", "total_spend"])
        missing_alloc = required_alloc_cols - set(allocation_df.columns)
        if missing_alloc:
            raise ValueError(
                f"allocation_df is missing required columns: {sorted(missing_alloc)}"
            )

        working = base_df.copy()
        working["need_total"] = working.apply(
            lambda row: sum(float(pd.to_numeric(row[COUNT_COLS[t]], errors="coerce") or 0.0) for t in INTERVENTION_TYPES),
            axis=1,
        )

        merged = working.merge(
            allocation_df[cls.KEY_COLS + ["total_treated", "total_spend"]],
            on=cls.KEY_COLS,
            how="left",
            validate="one_to_one",
        )

        merged["total_treated"] = pd.to_numeric(merged["total_treated"], errors="coerce").fillna(0.0)
        merged["total_spend"] = pd.to_numeric(merged["total_spend"], errors="coerce").fillna(0.0)
        merged["coverage_ratio"] = merged.apply(
            lambda row: float(row["total_treated"]) / float(row["need_total"])
            if float(row["need_total"]) > 0
            else 0.0,
            axis=1,
        )
        return merged

    # Compute the Gini coefficient of a list of non-negative values.
    # Input: numeric list, typically coverage ratios.
    # Output: inequality score from 0 (equal) upward.
    # Purpose: summarize how unevenly benefits are distributed.
    @staticmethod
    def _gini(values: List[float]) -> float:
        clean = [float(v) for v in values if pd.notna(v)]
        if not clean:
            return 0.0
        if any(v < 0 for v in clean):
            raise ValueError("Gini coefficient is undefined for negative values.")
        if all(v == 0 for v in clean):
            return 0.0

        sorted_vals = sorted(clean)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        weighted_sum = sum((idx + 1) * val for idx, val in enumerate(sorted_vals))
        return (2 * weighted_sum) / (n * total) - (n + 1) / n

    # Measure total efficiency as the number of children treated overall.
    # Input: allocation result dict.
    # Output: total treated across all rows.
    # Purpose: quantify the aggregate impact of the solution.
    @staticmethod
    def total_lives_impacted(allocation: Dict, problem: AllocationProblem) -> float:
        """Efficiency metric: total treated children across all rows."""
        allocation_df = FairnessMetrics._allocation_df(allocation)
        if "total_treated" not in allocation_df.columns:
            raise ValueError("allocation_df must contain a 'total_treated' column.")
        return float(pd.to_numeric(allocation_df["total_treated"], errors="coerce").fillna(0.0).sum())

    # Measure inequality in achieved coverage across rows.
    # Input: allocation result dict + AllocationProblem.
    # Output: Gini coefficient on coverage ratios.
    # Purpose: check whether coverage is distributed evenly or not.
    @staticmethod
    def gini_coefficient(allocation: Dict, problem: AllocationProblem) -> float:
        """Inequality of achieved coverage across rows (0 = perfectly equal coverage)."""
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)
        return float(FairnessMetrics._gini(merged["coverage_ratio"].tolist()))

    # Compare the best-served and worst-served groups in terms of coverage.
    # Input: allocation result dict + AllocationProblem.
    # Output: max coverage divided by min coverage.
    # Purpose: capture worst-case disparity in a simple fairness indicator.
    @staticmethod
    def max_min_ratio(allocation: Dict, problem: AllocationProblem) -> float:
        """Best-off / worst-off achieved coverage ratio across rows with positive need."""
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)
        valid = merged[merged["need_total"] > 0].copy()
        if valid.empty:
            return 1.0

        min_cov = float(valid["coverage_ratio"].min())
        max_cov = float(valid["coverage_ratio"].max())

        if math.isclose(max_cov, 0.0, abs_tol=1e-12):
            return 1.0
        if math.isclose(min_cov, 0.0, abs_tol=1e-12):
            return float("inf")
        return max_cov / min_cov
    
    # Measure within-country coverage gaps between demographic groups.
    # Input: allocation result dict + AllocationProblem.
    # Output: per-country gap details and mean gap summaries.
    # Purpose: detect whether some demographic groups are systematically underserved.
    """ !!! non passato ne risultati, da valtare come chiamarlo se interessa   """
    @staticmethod
    def demographic_coverage_gap(
        allocation: Dict,
        problem: AllocationProblem,
        group_col: str = "Demographic_group",
    ) -> Dict[str, Any]:
        """
        Per ogni paese, misura il gap di copertura tra gruppi demografici.

        Returns:
            {
            "per_country": {
                "AFG": {"max_gap": 0.32, "best_group": "urban", "worst_group": "rural"},
                ...
            },
            "weighted_mean_gap": 0.18,   # media pesata sul need totale
            "unweighted_mean_gap": 0.21,
            }
        """
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)

        if group_col not in merged.columns:
            raise ValueError(
                f"Column '{group_col}' not found in data. "
                f"Available columns: {list(merged.columns)}"
            )

        per_country = {}
        weighted_gap_sum = 0.0
        total_need_sum = 0.0

        for country, country_df in merged.groupby("ISO3"):
            if country_df.shape[0] < 2:
                # Un solo gruppo demografico: gap non definito
                continue

            # Escludiamo righe senza bisogno
            valid = country_df[country_df["need_total"] > 0].copy()
            if valid.shape[0] < 2:
                continue

            idx_min = valid["coverage_ratio"].idxmin()
            idx_max = valid["coverage_ratio"].idxmax()
            gap = float(valid.at[idx_max, "coverage_ratio"] - valid.at[idx_min, "coverage_ratio"])
            country_need = float(valid["need_total"].sum())

            per_country[country] = {
                "max_gap": gap,
                "best_group": valid.at[idx_max, group_col],
                "worst_group": valid.at[idx_min, group_col],
                "n_groups": int(valid.shape[0]),
            }

            weighted_gap_sum += gap * country_need
            total_need_sum += country_need

        if not per_country:
            return {
                "per_country": {},
                "weighted_mean_gap": 0.0,
                "unweighted_mean_gap": 0.0,
            }

        unweighted_mean = float(
            sum(v["max_gap"] for v in per_country.values()) / len(per_country)
        )
        weighted_mean = float(weighted_gap_sum / total_need_sum) if total_need_sum > 0 else 0.0

        return {
            "per_country": per_country,
            "weighted_mean_gap": weighted_mean,
            "unweighted_mean_gap": unweighted_mean,
        }

    # Compare observed demographic coverage with the ideal burden-based distribution.
    # Input: allocation result dict + AllocationProblem.
    # Output: violation score between 0 and 1.
    # Purpose: measure how far treatment distribution departs from demographic need.
    """ !!! passato nei risultati, da valtare se interessa   """
    @staticmethod
    def demographic_parity_violation(
        allocation: Dict,
        problem: AllocationProblem,
        group_col: str = "Demographic_group",
    ) -> float:
        """
        Misura quanto la distribuzione di copertura per gruppo demografico
        si discosta dalla distribuzione proporzionale al bisogno.

        Metrica: total variation distance tra distribuzione di copertura
        osservata e distribuzione ideale burden-proportional, aggregata
        su tutti i paesi e pesata sul need.

        0.0 -> copertura perfettamente proporzionale al bisogno per ogni gruppo
        1.0 -> massima deviazione possibile
        """
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)

        if group_col not in merged.columns:
            raise ValueError(
                f"Column '{group_col}' not found in data. "
                f"Available columns: {list(merged.columns)}"
            )

        violations = []
        weights = []

        for country, country_df in merged.groupby("ISO3"):
            valid = country_df[country_df["need_total"] > 0].copy()
            if valid.empty:
                continue

            country_need_total = float(valid["need_total"].sum())
            country_treated_total = float(valid["total_treated"].sum())

            if math.isclose(country_need_total, 0.0, abs_tol=1e-12):
                continue
            if math.isclose(country_treated_total, 0.0, abs_tol=1e-12):
                # Nessuno trattato: tutta la deviazione è massima
                violations.append(1.0)
                weights.append(country_need_total)
                continue

            # Distribuzione ideale: proporzionale al bisogno
            burden_share = valid["need_total"] / country_need_total

            # Distribuzione osservata: proporzionale ai trattati
            treated_share = valid["total_treated"] / country_treated_total

            # Total variation distance per questo paese
            tvd = float(0.5 * (treated_share - burden_share).abs().sum())
            violations.append(tvd)
            weights.append(country_need_total)

        if not violations:
            return 0.0

        total_weight = sum(weights)
        if math.isclose(total_weight, 0.0, abs_tol=1e-12):
            return 0.0

        return float(sum(v * w for v, w in zip(violations, weights)) / total_weight)

    # Compare budget shares with need shares across the allocation.
    # Input: allocation result dict + AllocationProblem.
    # Output: total variation distance between spend and burden distributions.
    # Purpose: assess whether money follows need proportionally.
    @staticmethod
    def proportionality_violation(allocation: Dict, problem: AllocationProblem) -> float:
        """
        Distance between budget shares and burden shares.

        Returns the total variation distance:
            0   -> perfectly burden-proportional spending
            1   -> maximal deviation
        """
        merged = FairnessMetrics._merged_allocation_view(allocation, problem)
        total_need = float(merged["need_total"].sum())
        total_spend = float(merged["total_spend"].sum())

        if math.isclose(total_need, 0.0, abs_tol=1e-12) or math.isclose(total_spend, 0.0, abs_tol=1e-12):
            return 0.0

        burden_share = merged["need_total"] / total_need
        spend_share = merged["total_spend"] / total_spend
        return float(0.5 * (spend_share - burden_share).abs().sum())

    # Build a compact summary of the main efficiency and fairness metrics.
    # Input: allocation result dict + AllocationProblem.
    # Output: dictionary of key evaluation indicators for one solution.
    # Purpose: make a single allocation easy to inspect and report.
    @staticmethod
    def build_summary(allocation: Dict, problem: AllocationProblem, label: str = "solution") -> Dict[str, Any]:
        """Return a compact metric summary for one allocation result."""
        total_treated = FairnessMetrics.total_lives_impacted(allocation, problem)
        demographic_gap = FairnessMetrics.demographic_parity_violation(allocation, problem)
        return {
            "label": label,
            "status": allocation.get("status", "UNKNOWN"),
            "total_treated": total_treated,
            "total_lives_impacted": total_treated,
            "total_spend": float(allocation.get("total_spend", 0.0) or 0.0),
            "budget_utilisation_pct": float(allocation.get("budget_utilisation_pct", 0.0) or 0.0),
            "gini_coefficient": FairnessMetrics.gini_coefficient(allocation, problem),
            "max_min_ratio": FairnessMetrics.max_min_ratio(allocation, problem),
            "demographic_coverage_gap": demographic_gap,
            "demographic_parity_violation": demographic_gap,
            "proportionality_violation": FairnessMetrics.proportionality_violation(allocation, problem),
        }

    # Compare baseline and fairness-aware solutions side by side.
    # Input: two allocation result dicts + AllocationProblem.
    # Output: summaries, comparison table, and price of fairness.
    # Purpose: quantify the trade-off between efficiency and equity.
    @staticmethod
    def compare_allocations(
        baseline: Dict,
        fairness: Dict,
        problem: AllocationProblem,
        fairness_label: str = "fairness",
    ) -> Dict[str, Any]:
        """
        Compare baseline and fairness allocation outputs.

        Returns:
          - one summary dict per solution
          - a comparison DataFrame
          - price_of_fairness_pct
        """
        baseline_summary = FairnessMetrics.build_summary(baseline, problem, label="baseline")
        fairness_summary = FairnessMetrics.build_summary(fairness, problem, label=fairness_label)

        comparison_df = pd.DataFrame(
            {
                "metric": [
                    "total_lives_impacted",
                    "total_spend",
                    # "budget_utilisation_pct",
                    "gini_coefficient",
                    "max_min_ratio",
                    "proportionality_violation",
                ],
                "baseline": [
                    baseline_summary["total_lives_impacted"],
                    baseline_summary["total_spend"],
                    # baseline_summary["budget_utilisation_pct"],
                    baseline_summary["gini_coefficient"],
                    baseline_summary["max_min_ratio"],
                    baseline_summary["proportionality_violation"],
                ],
                fairness_label: [
                    fairness_summary["total_lives_impacted"],
                    fairness_summary["total_spend"],
                    # fairness_summary["budget_utilisation_pct"],
                    fairness_summary["gini_coefficient"],
                    fairness_summary["max_min_ratio"],
                    fairness_summary["proportionality_violation"],
                ],
            }
        )
        comparison_df["absolute_delta"] = comparison_df[fairness_label] - comparison_df["baseline"]
        comparison_df["relative_delta_pct"] = comparison_df.apply(
            lambda row: 100.0 * row["absolute_delta"] / row["baseline"]
            if not math.isclose(float(row["baseline"]), 0.0, abs_tol=1e-12)
            else float("nan"),
            axis=1,
        )

        baseline_lives = float(baseline_summary["total_lives_impacted"])
        fairness_lives = float(fairness_summary["total_lives_impacted"])
        price_of_fairness_pct = (
            100.0 * (baseline_lives - fairness_lives) / baseline_lives
            if not math.isclose(baseline_lives, 0.0, abs_tol=1e-12)
            else 0.0
        )

        return {
            "baseline_summary": baseline_summary,
            "fairness_summary": fairness_summary,
            "comparison_df": comparison_df,
            "price_of_fairness_pct": price_of_fairness_pct,
        }
