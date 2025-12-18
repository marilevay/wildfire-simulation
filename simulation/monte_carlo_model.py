from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .data_collector import DataCollector
from .forest import Forest


@dataclass(frozen=True)
class MonteCarloSummary:
    """Arrays of outcomes from repeated Monte Carlo wildfire simulations.

    Each element corresponds to one stochastic simulation run.
    """

    burned_fraction: np.ndarray
    affected_fraction: np.ndarray

@dataclass(frozen=True)
class FireModelParams:
    """Parameters controlling stochastic fire spread.

    The spread probability is computed from destination density (optionally nonlinearly) and then
    biased by wind alignment.
    """

    wind: Tuple[float, float] = (0.0, 0.0)
    wind_strength: float = 0.0
    density_exponent: float = 1.0
    base_spread: float = 1.0
 

class MonteCarlo:
    """Run repeated stochastic fire simulations on a Forest.

    High-level flow:
    - sample an ignition location (weighted by density)
    - grow a burned cluster via neighbor-to-neighbor stochastic spread
    - derive an affected mask (burned + neighbors)
    - repeat to estimate distributions and risk maps
    """

    def __init__(self, forest: Forest, params: FireModelParams, n_runs: int, rng: Optional[np.random.Generator] = None):
        """Create a Monte Carlo runner.

        If `rng` is not provided, a new NumPy default RNG is created.
        """
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")

        self.forest = forest
        self.params = params
        self.n_runs = int(n_runs)
        self.random_generator = rng if rng is not None else np.random.default_rng()

    def run(self) -> DataCollector:
        """Run `n_runs` simulations and return per-run burned/affected fractions."""
        collector = DataCollector()
        total_cells = float(self.forest.density.size)

        for _ in range(int(self.n_runs)):
            # One stochastic fire realization.
            burned_mask = self._simulate_fire(density=self.forest.density, ignition=None)
            affected_mask = self._adjacent_mask(burned_mask)
            collector.burned_fraction.append(float(burned_mask.sum()) / total_cells)
            collector.affected_fraction.append(float(affected_mask.sum()) / total_cells)

        return collector

    def risk_map(self, n_runs: Optional[int] = None) -> np.ndarray:
        """Estimate per-cell probability of being affected (burned or adjacent)."""
        runs = self.n_runs if n_runs is None else int(n_runs)
        if runs <= 0:
            raise ValueError("n_runs must be positive")

        affected_counts = np.zeros_like(self.forest.density, dtype=float)
        for _ in range(int(runs)):
            # Count how often each cell is affected across runs.
            burned_mask = self._simulate_fire(density=self.forest.density, ignition=None)
            affected_mask = self._adjacent_mask(burned_mask)
            affected_counts += affected_mask.astype(float)

        return affected_counts / float(runs)

    def _neighbors(self, i: int, j: int, nrows: int, ncols: int):
        """Yield neighbor coordinates using Moore (8-neighbor) connectivity."""
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbor_i, neighbor_j = i + di, j + dj
                if 0 <= neighbor_i < nrows and 0 <= neighbor_j < ncols:
                    yield neighbor_i, neighbor_j

    def _spread_probability(self, density_to: float, delta: Tuple[int, int]) -> float:
        """Probability of spreading from a burning cell to a neighbor cell."""
        if density_to <= 0.0:
            return 0.0

        # Base spread increases with destination fuel density.
        density_component = float(np.clip(density_to, 0.0, 1.0)) ** float(self.params.density_exponent)
        spread_probability = float(self.params.base_spread) * density_component

        wind_x, wind_y = self.params.wind
        wind_norm = float(np.hypot(wind_x, wind_y))
        if self.params.wind_strength > 0.0 and wind_norm > 0.0:
            # Wind bias is proportional to cosine alignment with the wind direction.
            direction_x, direction_y = float(delta[1]), float(delta[0])
            direction_norm = float(np.hypot(direction_x, direction_y))
            if direction_norm > 0.0:
                alignment = (direction_x * wind_x + direction_y * wind_y) / (direction_norm * wind_norm)
                spread_probability *= 1.0 + float(self.params.wind_strength) * float(alignment)

        return float(np.clip(spread_probability, 0.0, 1.0))

    def _simulate_fire(self, density: np.ndarray, ignition: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Simulate one fire event and return a boolean mask of burned cells."""
        if density.ndim != 2:
            raise ValueError("density must be a 2D array")

        nrows, ncols = density.shape

        if ignition is None:
            # Sample ignition proportional to density so non-fuel cells are unlikely/never chosen.
            ignition_weights = np.clip(density.astype(float), 0.0, 1.0)
            weight_sum = float(ignition_weights.sum())
            if weight_sum <= 0.0:
                return np.zeros_like(density, dtype=bool)
            flat_index = self.random_generator.choice(nrows * ncols, p=(ignition_weights.ravel() / weight_sum))
            ignition = (int(flat_index // ncols), int(flat_index % ncols))

        ignition_row, ignition_col = ignition
        if not (0 <= ignition_row < nrows and 0 <= ignition_col < ncols):
            raise ValueError("ignition out of bounds")

        # Burned mask is the only state we track (frontier expansion process).
        burned_mask = np.zeros((nrows, ncols), dtype=bool)
        if density[ignition_row, ignition_col] <= 0.0:
            return burned_mask

        burned_mask[ignition_row, ignition_col] = True
        frontier = [(ignition_row, ignition_col)]

        while frontier:
            cell_i, cell_j = frontier.pop()
            for neighbor_i, neighbor_j in self._neighbors(cell_i, cell_j, nrows, ncols):
                if burned_mask[neighbor_i, neighbor_j]:
                    continue

                delta_i, delta_j = neighbor_i - cell_i, neighbor_j - cell_j
                spread_probability = self._spread_probability(float(density[neighbor_i, neighbor_j]), (delta_i, delta_j))
                if self.random_generator.random() < spread_probability:
                    burned_mask[neighbor_i, neighbor_j] = True
                    frontier.append((neighbor_i, neighbor_j))

        return burned_mask

    def _adjacent_mask(self, burned: np.ndarray) -> np.ndarray:
        """Return a boolean mask of cells that are burned or adjacent to burned."""
        nrows, ncols = burned.shape
        adjacent_mask = burned.copy()
        for i in range(nrows):
            for j in range(ncols):
                if not burned[i, j]:
                    continue
                for neighbor_i, neighbor_j in self._neighbors(i, j, nrows, ncols):
                    adjacent_mask[neighbor_i, neighbor_j] = True
        return adjacent_mask
