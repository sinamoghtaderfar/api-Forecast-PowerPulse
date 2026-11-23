"""
Empirical prediction intervals from rolling-origin backtest residuals.

Production method (:class:`ScaledResidualCalibration`, since model version 4):
volatility-scaled, horizon-pooled split-conformal intervals. Each backtest
error ``e`` made at origin ``o`` for horizon ``h`` is turned into the score
``s = |e| / (sigma_o * sqrt(h))``, where ``sigma_o`` is the standard deviation
of the last ``scale_window`` annual changes of the target observed at ``o``.
Scores of horizons ``1..max_horizon`` are pooled, ``q`` is their
split-conformal quantile, and the half-width at horizon ``h`` is
``q * sigma_T * sqrt(h)`` with ``sigma_T`` measured at the production origin.
Beyond ``max_horizon`` no interval is returned (no validation evidence).

Why: per-horizon quantiles (:class:`ResidualCalibration`, the pre-v4 method)
need >= 19 residuals per horizon and covered only 17-90 % in sequential
backtests because the residuals pre-dating the 2018+ decline were too small.
Scaling by recent volatility lets the width react to the larger changes since
2018; pooling makes intervals available at every horizon up to ``max_horizon``.
See ``docs/FORECAST_ACCURACY_REPORT.md``.

Caveats: residuals come from overlapping origins (not independent); the
sqrt(h) growth and the volatility scaling are assumptions; the intervals are
**not** guaranteed to cover 95 % - their backtested coverage is reported in
:meth:`ScaledResidualCalibration.summary`.

Legacy per-horizon rule (:class:`ResidualCalibration`): for each horizon ``h``
the half-width is ``q_h = |e|_(k)`` with ``k = ceil((n_h + 1) * level)``;
``None`` when ``k > n_h``; half-widths are made non-decreasing in ``h``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_LEVEL = 0.95
METHOD = "rolling-origin backtest residuals (split-conformal, per horizon)"
SCALED_METHOD = (
    "volatility-scaled split-conformal on rolling-origin backtest residuals "
    "(pooled over horizons, sqrt(h) growth)"
)
DEFAULT_MAX_HORIZON = 10
DEFAULT_SCALE_WINDOW = 10
# Sequential coverage is reported for forecasts made from this origin onwards.
FIRST_COVERAGE_ORIGIN = 2000
COVERAGE_GROUPS = {"1-3": (1, 3), "4-5": (4, 5), "6-10": (6, 10)}


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


def volatility_scale(y: Sequence[float], window: int = DEFAULT_SCALE_WINDOW) -> float:
    """
    Sample standard deviation of the last ``window`` annual changes of ``y``.

    Only uses the values passed in (all observed at the forecast origin).
    Returns a tiny positive number for a series without variation so that
    scores stay finite.
    """
    changes = np.diff(np.asarray(y, dtype=float))[-window:]
    if changes.size < 2:
        return 1e-9
    return max(float(np.std(changes, ddof=1)), 1e-9)


@dataclass
class ScaledResidualCalibration:
    """
    Volatility-scaled, horizon-pooled split-conformal calibration (see module doc).

    ``origins``/``horizons``/``years``/``residuals``/``scales`` are parallel
    lists of the backtest errors (actual - predicted) and of ``sigma`` at each
    error's origin; ``current_scale`` is ``sigma`` at the production origin.
    """

    level: float = DEFAULT_LEVEL
    max_horizon: int = DEFAULT_MAX_HORIZON
    scale_window: int = DEFAULT_SCALE_WINDOW
    current_scale: float = 1.0
    origins: list[int] = field(default_factory=list)
    horizons: list[int] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)

    @classmethod
    def from_backtest(
        cls,
        records: Iterable[Any],
        target_by_year: dict[int, float],
        level: float = DEFAULT_LEVEL,
        max_horizon: int = DEFAULT_MAX_HORIZON,
        scale_window: int = DEFAULT_SCALE_WINDOW,
    ) -> "ScaledResidualCalibration":
        """
        Build from backtest records (``origin``, ``horizon``, ``year``,
        ``actual``, ``predicted``) and the observed target per year. ``sigma``
        of an origin only uses targets of years <= that origin.
        """
        years_sorted = sorted(target_by_year)
        values = [float(target_by_year[y]) for y in years_sorted]

        def scale_at(origin: int) -> float:
            upto = [v for y, v in zip(years_sorted, values) if y <= origin]
            return volatility_scale(upto, scale_window)

        cache: dict[int, float] = {}
        cal = cls(
            level=float(level),
            max_horizon=int(max_horizon),
            scale_window=int(scale_window),
            current_scale=volatility_scale(values, scale_window),
        )
        for record in records:
            origin = int(record.origin)
            if origin not in cache:
                cache[origin] = scale_at(origin)
            cal.origins.append(origin)
            cal.horizons.append(int(record.horizon))
            cal.years.append(int(record.year))
            cal.residuals.append(float(record.actual - record.predicted))
            cal.scales.append(float(cache[origin]))
        return cal

    def _scores(self, max_year: int | None = None) -> np.ndarray:
        h = np.asarray(self.horizons, dtype=float)
        mask = h <= self.max_horizon
        if max_year is not None:
            mask &= np.asarray(self.years) <= max_year
        res = np.abs(np.asarray(self.residuals, dtype=float))
        scale = np.asarray(self.scales, dtype=float)
        return res[mask] / (scale[mask] * np.sqrt(h[mask]))

    @property
    def quantile(self) -> float | None:
        """Conformal quantile of the pooled scores (None if too few)."""
        if not self.residuals:
            return None
        return conformal_half_width(self._scores(), self.level)

    def half_widths(self, horizon: int) -> list[float | None]:
        """Half-widths for horizons 1..``horizon`` (None beyond ``max_horizon``)."""
        q = self.quantile
        return [
            (
                None
                if q is None or h > self.max_horizon
                else float(q * self.current_scale * math.sqrt(h))
            )
            for h in range(1, horizon + 1)
        ]

    @property
    def max_calibrated_horizon(self) -> int:
        """Largest horizon with an interval."""
        return self.max_horizon if self.quantile is not None else 0

    def residuals_by_horizon(self) -> dict[int, list[float]]:
        """Signed residuals grouped by horizon."""
        out: dict[int, list[float]] = {}
        for h, r in zip(self.horizons, self.residuals):
            out.setdefault(h, []).append(r)
        return out

    def quantile_at(self, origin: int) -> float | None:
        """Score quantile using only errors whose target year is <= ``origin``."""
        return conformal_half_width(self._scores(max_year=origin), self.level)

    def sequential_coverage(
        self,
        first_origin: int = FIRST_COVERAGE_ORIGIN,
        groups: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Backtested coverage without look-ahead: the interval of each forecast
        (origin >= ``first_origin``, h <= ``max_horizon``) only uses scores of
        errors whose target year is <= its origin (:meth:`quantile_at`).
        Grouped by horizon range (default: 1-3, 4-5, 6-10 and all).
        """
        if groups is None:
            groups = {**COVERAGE_GROUPS, "all": (1, self.max_horizon)}
        hits: dict[str, list[tuple[bool, float]]] = {}
        quantiles: dict[int, float | None] = {}
        for origin, h, res, scale in zip(
            self.origins, self.horizons, self.residuals, self.scales
        ):
            if origin < first_origin or h > self.max_horizon:
                continue
            if origin not in quantiles:
                quantiles[origin] = self.quantile_at(origin)
            q = quantiles[origin]
            if q is None:
                continue
            width = q * scale * math.sqrt(h)
            for label, (lo, hi) in groups.items():
                if lo <= h <= hi:
                    hits.setdefault(label, []).append((abs(res) <= width, 2 * width))
        return {
            label: {
                "n": len(values),
                "coverage": float(np.mean([c for c, _ in values])),
                "mean_width": float(np.mean([w for _, w in values])),
            }
            for label, values in hits.items()
        }

    def backtest_mae(
        self, first_origin: int = FIRST_COVERAGE_ORIGIN
    ) -> dict[int, dict[str, float]]:
        """MAE and bias (forecast - actual) per horizon <= max_horizon."""
        out: dict[int, list[float]] = {}
        for origin, h, res in zip(self.origins, self.horizons, self.residuals):
            if origin >= first_origin and h <= self.max_horizon:
                out.setdefault(h, []).append(res)
        return {
            h: {
                "n": len(v),
                "mae": float(np.mean(np.abs(v))),
                "bias": float(-np.mean(v)),
            }
            for h, v in sorted(out.items())
        }

    def summary(self) -> dict[str, Any]:
        """JSON-friendly description for API metadata (superset of the legacy one)."""
        q = self.quantile
        return {
            "method": SCALED_METHOD,
            "nominal_coverage": self.level,
            "max_calibrated_horizon": self.max_calibrated_horizon,
            "residuals_per_horizon": {
                str(h): len(v) for h, v in sorted(self.residuals_by_horizon().items())
            },
            "scale_twh": self.current_scale,
            "scale_window_years": self.scale_window,
            "pooled_horizons": self.max_horizon,
            "score_quantile": q,
            "backtest_coverage": self.sequential_coverage(),
            "validated_coverage": False,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize (JSON-safe)."""
        return {
            "kind": "scaled",
            "level": self.level,
            "max_horizon": self.max_horizon,
            "scale_window": self.scale_window,
            "current_scale": self.current_scale,
            "origins": list(self.origins),
            "horizons": list(self.horizons),
            "years": list(self.years),
            "residuals": list(self.residuals),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScaledResidualCalibration":
        """Inverse of :meth:`to_dict`; raises on the legacy format."""
        if data.get("kind") != "scaled":
            raise ValueError("Not a scaled calibration")
        return cls(
            level=float(data["level"]),
            max_horizon=int(data["max_horizon"]),
            scale_window=int(data["scale_window"]),
            current_scale=float(data["current_scale"]),
            origins=[int(x) for x in data["origins"]],
            horizons=[int(x) for x in data["horizons"]],
            years=[int(x) for x in data["years"]],
            residuals=[float(x) for x in data["residuals"]],
            scales=[float(x) for x in data["scales"]],
        )
