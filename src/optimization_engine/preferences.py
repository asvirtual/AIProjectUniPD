# ============================================================================
# Preference elicitation and validation for stakeholder preferences
# ============================================================================

import json
from typing import Any, Dict, Optional

from .models import StakeholderPreferences


class PreferenceElicitor:
    """
    Helpers for loading, validating, and normalizing stakeholder preferences.

    Expected responsibilities:
      - load preferences from JSON
      - validate metric weights
      - validate demographic/fairness constraints
      - normalize input into StakeholderPreferences-ready structures

    Constraint convention used here:
      demographic_constraints = {
          "rural_min_share": 0.40,
          "urban_max_share": 0.60,
          "female_min_share": 0.45,
          "male_max_share": 0.55,
          "poorest_quintile_min_share": 0.25,
      }

    Each key follows the pattern:
        <group_value>_<min|max>_share

    Examples:
        rural_min_share = 0.40
        female_min_share = 0.50
        q1_min_share = 0.20
    """

    ALLOWED_FAIRNESS_MODES = {"utilitarian", "max-min", "proportional"}

    def __init__(
        self,
        metric_weights: Dict[str, float],
        demographic_constraints: Optional[Dict[str, float]] = None,
        fairness_mode: str = "utilitarian",
    ):
        self.metric_weights = metric_weights
        self.demographic_constraints = demographic_constraints or {}
        self.fairness_mode = fairness_mode

    @staticmethod
    def from_json(filepath: str) -> "PreferenceElicitor":
        """
        Load preferences from JSON.

        Supported JSON shape:
        {
          "metric_weights": {
            "stunting": 1.0,
            "wasting": 1.2,
            "severe_wasting": 1.5
          },
          "demographic_constraints": {
            "rural_min_share": 0.4,
            "urban_max_share": 0.6,
            "female_min_share": 0.45
          },
          "fairness_mode": "max-min"
        }
        """
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Preference JSON must contain a top-level object.")

        return PreferenceElicitor(
            metric_weights=payload.get("metric_weights", {}),
            demographic_constraints=payload.get("demographic_constraints", {}),
            fairness_mode=payload.get("fairness_mode", "utilitarian"),
        )

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _parse_constraint_key(key: str) -> Dict[str, str]:
        """
        Parse keys like:
            rural_min_share
            urban_max_share
            female_min_share
            q1_min_share

        Returns:
            {
              "group_value": "rural",
              "bound_type": "min",
              "quantity": "share"
            }
        """
        parts = key.split("_")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid constraint key '{key}'. Expected pattern <group_value>_<min|max>_share."
            )

        quantity = parts[-1]
        bound_type = parts[-2]
        group_value = "_".join(parts[:-2])

        if not group_value:
            raise ValueError(f"Invalid constraint key '{key}': missing group value.")
        if bound_type not in {"min", "max"}:
            raise ValueError(
                f"Invalid constraint key '{key}': bound must be 'min' or 'max'."
            )
        if quantity != "share":
            raise ValueError(
                f"Invalid constraint key '{key}': currently only '_share' constraints are supported."
            )

        return {
            "group_value": group_value,
            "bound_type": bound_type,
            "quantity": quantity,
        }

    def validate(self) -> bool:
        """
        Validate weights, constraint ranges, and fairness mode.

        Rules:
          - fairness_mode must be one of the supported modes
          - metric_weights must be a non-empty dict
          - each weight must be numeric and >= 0
          - at least one metric weight must be strictly positive
          - demographic constraint values must be numeric and in [0, 1]
          - min_share and max_share for the same group cannot conflict
        """
        if self.fairness_mode not in self.ALLOWED_FAIRNESS_MODES:
            raise ValueError(
                f"Unsupported fairness_mode '{self.fairness_mode}'. "
                f"Allowed values: {sorted(self.ALLOWED_FAIRNESS_MODES)}"
            )

        if not isinstance(self.metric_weights, dict) or not self.metric_weights:
            raise ValueError("metric_weights must be a non-empty dictionary.")

        positive_weight_found = False
        for metric, weight in self.metric_weights.items():
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("Each metric name must be a non-empty string.")
            if not self._is_number(weight):
                raise ValueError(f"Metric weight for '{metric}' must be numeric.")
            if weight < 0:
                raise ValueError(f"Metric weight for '{metric}' cannot be negative.")
            if weight > 0:
                positive_weight_found = True

        if not positive_weight_found:
            raise ValueError("At least one metric weight must be strictly positive.")

        if not isinstance(self.demographic_constraints, dict):
            raise ValueError("demographic_constraints must be a dictionary.")

        grouped_bounds: Dict[str, Dict[str, float]] = {}

        for key, value in self.demographic_constraints.items():
            parsed = self._parse_constraint_key(key)

            if not self._is_number(value):
                raise ValueError(f"Constraint '{key}' must have a numeric value.")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"Constraint '{key}' must be between 0 and 1.")

            group_value = parsed["group_value"]
            bound_type = parsed["bound_type"]
            grouped_bounds.setdefault(group_value, {})[bound_type] = float(value)

        for group_value, bounds in grouped_bounds.items():
            if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
                raise ValueError(
                    f"Conflicting constraints for '{group_value}': "
                    f"min_share ({bounds['min']}) cannot exceed max_share ({bounds['max']})."
                )

        return True

    def normalized_metric_weights(self) -> Dict[str, float]:
        """Return metric weights normalized to sum to 1 when possible."""
        total = float(sum(self.metric_weights.values()))
        if total <= 0:
            raise ValueError("Cannot normalize metric weights: total weight must be > 0.")
        return {metric: float(weight) / total for metric, weight in self.metric_weights.items()}

    def normalized_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert flat constraint keys into a structured representation.

        Example input:
            {
              "rural_min_share": 0.40,
              "urban_max_share": 0.60,
              "female_min_share": 0.45
            }

        Example output:
            {
              "rural": {"type": "share", "min": 0.40},
              "urban": {"type": "share", "max": 0.60},
              "female": {"type": "share", "min": 0.45}
            }
        """
        structured: Dict[str, Dict[str, Any]] = {}

        for key, value in self.demographic_constraints.items():
            parsed = self._parse_constraint_key(key)
            group_value = parsed["group_value"]
            bound_type = parsed["bound_type"]
            quantity = parsed["quantity"]

            structured.setdefault(group_value, {"type": quantity})
            structured[group_value][bound_type] = float(value)

        return structured

    def to_preferences(self) -> StakeholderPreferences:
        """
        Validate and convert to StakeholderPreferences dataclass.

        We store:
          - normalized metric weights
          - raw flat demographic constraints
          - fairness mode

        The flat constraint format is kept for compatibility with downstream
        optimizer code, while `normalized_constraints()` is available if a
        structured form is preferred during constraint generation.
        """
        self.validate()
        return StakeholderPreferences(
            metric_weights=self.normalized_metric_weights(),
            demographic_constraints={
                key: float(value) for key, value in self.demographic_constraints.items()
            },
            fairness_mode=self.fairness_mode,
        )