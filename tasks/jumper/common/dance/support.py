"""Validate references with changing near-ground support instead of four fixed feet.

A schedule labels near-ground feet, not measured contact forces. All tolerances
belong to the task. This check validates reference consistency, not dynamic
stability or hardware executability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class SupportSchedule:
    """LM/RM/LR/RR stance labels sampled at ``dt`` with task-owned tolerances."""

    stance: np.ndarray
    dt: float
    height_tolerance: float
    max_site_drop: float
    min_support_feet: int

    def __post_init__(self):
        mask = np.asarray(self.stance)
        if mask.ndim != 2 or mask.shape[1] != 4 or len(mask) < 2:
            raise ValueError("Support schedule must have shape [frames, 4]")
        if mask.dtype != np.bool_:
            raise ValueError("Support schedule must use boolean stance labels")
        if not 0 < self.dt < 1 or not 1 <= self.min_support_feet <= 4:
            raise ValueError("Invalid support schedule rate or minimum support count")
        if not np.isfinite([self.height_tolerance, self.max_site_drop]).all():
            raise ValueError("Support tolerances must be finite")
        if self.height_tolerance <= 0 or self.max_site_drop <= 0:
            raise ValueError("Support tolerances must be positive")
        mask = mask.copy()
        mask.flags.writeable = False
        object.__setattr__(self, "stance", mask)

    @property
    def cache_key(self) -> str:
        """Changing labels, rate or tolerances must invalidate cached conversion."""
        h = hashlib.sha256(self.stance.tobytes())
        h.update(
            json.dumps(
                [self.dt, self.height_tolerance, self.max_site_drop, self.min_support_feet]
            ).encode()
        )
        return "scheduled-support-v1:" + h.hexdigest()

    def validate(self, site_z: np.ndarray, home_z: float, target_dt: float) -> dict:
        """Require named stance feet near the floor and no excessive downward site offset."""
        z = np.asarray(site_z)
        if z.ndim != 2 or z.shape[1] != 4 or not np.isfinite(z).all():
            raise ValueError("Reference support-site heights must be finite [frames, 4]")
        if len(z) < 2 or not np.isfinite(home_z) or not 0 < target_dt < 1:
            raise ValueError("Invalid reference support timeline or floor height")
        # Resample the schedule at nearest source ticks, matching the reference
        # duration. Do not stretch a short/wrong schedule to fit a different clip.
        if abs((len(z) - 1) * target_dt - (len(self.stance) - 1) * self.dt) > self.dt + 1e-8:
            raise ValueError("Support schedule duration differs from reference")
        idx = np.rint(np.arange(len(z)) * target_dt / self.dt).astype(int)
        if idx[-1] >= len(self.stance):
            raise ValueError("Support schedule does not cover the reference")
        mask = self.stance[idx]
        counts = mask.sum(axis=1)
        if np.any(counts < self.min_support_feet):
            raise ValueError("Reference schedule has too few near-ground support feet")
        error = np.abs(z - home_z)
        if np.any(error[mask] > self.height_tolerance):
            raise ValueError("Scheduled stance foot is not near the reference floor")
        if np.any(z < home_z - self.max_site_drop):
            raise ValueError("Reference foot site is below the home-height tolerance")
        return {
            "mode": "scheduled near-ground support, not contact-force validation",
            "cache_key": self.cache_key,
            "min_support_feet": int(counts.min()),
            "max_stance_height_error_m": float(error[mask].max()),
            "lowest_site_relative_to_home_m": float((z - home_z).min()),
        }
