from __future__ import annotations

import numpy as np

class DataCollector:
    """Collect per-run outcomes

    We store burned/affected outcomes as fractions in [0, 1] so they are comparable across grid
    sizes
    """

    def __init__(self):
        self.burned_fraction: list[float] = []
        self.affected_fraction: list[float] = []

    def add_run(self, burned: np.ndarray, affected: np.ndarray) -> None:
        """Add a single run's burned/affected boolean masks as fractional outcomes"""
        # Convert boolean masks into global fractions for this run
        n_cells = float(burned.size)
        self.burned_fraction.append(float(burned.sum()) / n_cells)
        self.affected_fraction.append(float(affected.sum()) / n_cells)

    def convert_to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return collected fractions as NumPy arrays (burned_fraction, affected_fraction)"""
        return np.asarray(self.burned_fraction, dtype=float), np.asarray(self.affected_fraction, dtype=float)

    @staticmethod
    def calculate_ci95_mean(samples: np.ndarray) -> tuple[float, float, float]:
        """Compute a 95% CI for the mean using a normal approximation

        Returns (mean, lower, upper)
        """
        sample_array = np.asarray(samples, dtype=float)
        if sample_array.ndim != 1:
            raise ValueError("samples must be a 1D array")
        sample_count = int(sample_array.size)
        if sample_count <= 1:
            raise ValueError("need at least 2 samples")

        sample_mean = float(sample_array.mean())
        standard_error = float(sample_array.std(ddof=1)) / float(np.sqrt(sample_count))
        half_width = 1.96 * standard_error
        return sample_mean, sample_mean - half_width, sample_mean + half_width