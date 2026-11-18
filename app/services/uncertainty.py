"""
Empirical prediction intervals from rolling-origin backtest residuals.

For each forecast horizon ``h`` the half-width is the split-conformal quantile
of absolute recursive backtest errors at that horizon:
``q_h = |e|_(k)`` with ``k = ceil((n_h + 1) * level)``. When ``k > n_h`` there
are too few residuals for the requested level and no interval is returned
(``None``) instead of an unsupported one. Half-widths are made non-decreasing
in ``h`` (uncertainty does not shrink with the horizon).

Caveats: annual data give few residuals per horizon, residuals of a horizon
come from overlapping origins (not independent), and early origins use less
training data than production, which tends to make intervals conservative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

DEFAULT_LEVEL = 0.95
METHOD = "rolling-origin backtest residuals (split-conformal, per horizon)"


def conformal_half_width(abs_residuals: Iterable[float], level: float) -> float | None:
    """Split-conformal quantile of absolute residuals, or None if undefined."""
    values = np.sort(np.abs(np.asarray(list(abs_residuals), dtype=float)))
    n = values.size
    if n == 0:
        return None
    k = math.ceil((n + 1) * level)
    return float(values[k - 1]) if k <= n else None


def min_residuals_for_level(level: float) -> int:
    """Smallest residual count for which the conformal quantile is defined."""
    return math.ceil(level / (1 - level))


@dataclass
class ResidualCalibration:
    """Signed backtest residuals (actual - predicted) grouped by horizon."""

    level: float = DEFAULT_LEVEL
    residuals: dict[int, list[float]] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: Iterable[Any], level: float = DEFAULT_LEVEL):
        """Build from backtest records with ``horizon``, ``actual`` and ``predicted``."""
        residuals: dict[int, list[float]] = {}
        for record in records:
            residuals.setdefault(int(record.horizon), []).append(
                float(record.actual - record.predicted)
            )
        return cls(level=level, residuals=residuals)

    def half_widths(self, horizon: int) -> list[float | None]:
        """Half-widths for horizons 1..``horizon`` (None where unsupported)."""
        widths: list[float | None] = []
        running = 0.0
        for h in range(1, horizon + 1):
            width = conformal_half_width(self.residuals.get(h, []), self.level)
            if width is None or (widths and widths[-1] is None):
                widths.append(None)
                continue
            running = max(running, width)
            widths.append(running)
        return widths

    @property
    def max_calibrated_horizon(self) -> int:
        """Largest horizon with a defined interval."""
        widths = self.half_widths(max(self.residuals, default=0))
        return sum(1 for width in widths if width is not None)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly description for API metadata."""
        return {
            "method": METHOD,
            "nominal_coverage": self.level,
            "max_calibrated_horizon": self.max_calibrated_horizon,
            "residuals_per_horizon": {
                str(h): len(v) for h, v in sorted(self.residuals.items())
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize (JSON-safe)."""
        return {
            "level": self.level,
            "residuals": {str(h): v for h, v in self.residuals.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResidualCalibration":
        """Inverse of :meth:`to_dict`."""
        return cls(
            level=float(data["level"]),
            residuals={
                int(h): [float(x) for x in v] for h, v in data["residuals"].items()
            },
        )
