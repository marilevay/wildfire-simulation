from __future__ import annotations

import numpy as np

class Forest:
    """A gridded landscape with fuel density in [0, 1].

    Density is interpreted as available fuel / canopy cover at each cell.
    """

    def __init__(self, density: np.ndarray):
        """Create a forest from a 2D density grid (values are clipped to [0,1])."""
        density = np.asarray(density, dtype=float)
        if density.ndim != 2:
            raise ValueError("density must be a 2D array")

        self.density = np.clip(density, 0.0, 1.0)
        self.n_trees = self.density.size

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as (n_rows, n_cols)."""
        return self.density.shape

    def consider_density(self, density: np.ndarray) -> Forest:
        """Return a new Forest instance with a different density grid."""
        return Forest(density)

    def apply_thinning(self, factor: float) -> Forest:
        """Mitigation strategy: uniformly reduce density by a multiplicative factor."""
        if factor < 0.0 or factor > 1.0:
            raise ValueError("factor must be in [0, 1]")
        thinned_density = np.clip(self.density.astype(float) * float(factor), 0.0, 1.0)
        return self.consider_density(thinned_density)


