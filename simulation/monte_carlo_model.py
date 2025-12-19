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

        If rng is not provided, a new NumPy default RNG is created.
        """
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")

        self.forest = forest
        self.params = params
        self.n_runs = int(n_runs)
        self.random_generator = rng if rng is not None else np.random.default_rng()

    def run(self) -> DataCollector:
        """Run n_runs simulations and return per-run burned/affected fractions."""
        collector = DataCollector()
        total_cells = float(self.forest.density.size)
        
        # Precompute static spread probabilities and wind factors to speed up the loop
        spread_map, wind_factors = self._precompute_spread_dynamics(self.forest.density)
        
        # Precompute ignition weights
        density = self.forest.density
        ignition_weights = np.clip(density.astype(float), 0.0, 1.0)
        weight_sum = float(ignition_weights.sum())
        ignition_probs = ignition_weights.ravel() / weight_sum if weight_sum > 0 else None

        for _ in range(int(self.n_runs)):
            # One stochastic fire realization
            burned_mask = self._simulate_fire_optimized(
                density=density, 
                spread_map=spread_map, 
                wind_factors=wind_factors,
                ignition_probs=ignition_probs
            )
            affected_mask = self._adjacent_mask(burned_mask)
            collector.burned_fraction.append(float(burned_mask.sum()) / total_cells)
            collector.affected_fraction.append(float(affected_mask.sum()) / total_cells)

        return collector

    def risk_map(self, n_runs: Optional[int] = None) -> np.ndarray:
        """Estimate per-cell probability of being affected (burned or adjacent)"""
        runs = self.n_runs if n_runs is None else int(n_runs)
        if runs <= 0:
            raise ValueError("n_runs must be positive")

        affected_counts = np.zeros_like(self.forest.density, dtype=float)
        
        # Precomputations
        density = self.forest.density
        spread_map, wind_factors = self._precompute_spread_dynamics(density)
        ignition_weights = np.clip(density.astype(float), 0.0, 1.0)
        weight_sum = float(ignition_weights.sum())
        ignition_probs = ignition_weights.ravel() / weight_sum if weight_sum > 0 else None

        for _ in range(int(runs)):
            # Count how often each cell is affected across runs.
            burned_mask = self._simulate_fire_optimized(
                density=density,
                spread_map=spread_map,
                wind_factors=wind_factors,
                ignition_probs=ignition_probs
            )
            affected_mask = self._adjacent_mask(burned_mask)
            affected_counts += affected_mask.astype(float)

        return affected_counts / float(runs)

    def _neighbors(self, i: int, j: int, nrows: int, ncols: int):
        """Yield neighbor coordinates using Moore (8-neighbor) connectivity"""
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbor_i, neighbor_j = i + di, j + dj
                if 0 <= neighbor_i < nrows and 0 <= neighbor_j < ncols:
                    yield neighbor_i, neighbor_j

    def _precompute_spread_dynamics(self, density: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Precompute base spread maps and wind factors."""
        # Base spread increases with destination fuel density
        density_clamped = np.clip(density, 0.0, 1.0)
        spread_map = float(self.params.base_spread) * (density_clamped ** float(self.params.density_exponent))
        
        # Precompute wind factors for all 8 directions
        wind_factors = {}
        wind_x, wind_y = self.params.wind
        wind_norm = float(np.hypot(wind_x, wind_y))
        
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                
                factor = 1.0
                if self.params.wind_strength > 0.0 and wind_norm > 0.0:
                    direction_x, direction_y = float(dj), float(di)
                    direction_norm = float(np.hypot(direction_x, direction_y))
                    if direction_norm > 0.0:
                        alignment = (direction_x * wind_x + direction_y * wind_y) / (direction_norm * wind_norm)
                        factor = 1.0 + float(self.params.wind_strength) * float(alignment)
                
                wind_factors[(di, dj)] = factor
                
        return spread_map, wind_factors

    def _spread_probability(self, density_to: float, delta: Tuple[int, int]) -> float:
        """Probability of spreading from a burning cell to a neighbor cell (legacy method)"""
        # Kept for backward compatibility if needed, but core logic is now precomputed
        if density_to <= 0.0:
            return 0.0

        density_component = float(np.clip(density_to, 0.0, 1.0)) ** float(self.params.density_exponent)
        spread_probability = float(self.params.base_spread) * density_component

        wind_x, wind_y = self.params.wind
        wind_norm = float(np.hypot(wind_x, wind_y))
        if self.params.wind_strength > 0.0 and wind_norm > 0.0:
            direction_x, direction_y = float(delta[1]), float(delta[0])
            direction_norm = float(np.hypot(direction_x, direction_y))
            if direction_norm > 0.0:
                alignment = (direction_x * wind_x + direction_y * wind_y) / (direction_norm * wind_norm)
                spread_probability *= 1.0 + float(self.params.wind_strength) * float(alignment)

        return float(np.clip(spread_probability, 0.0, 1.0))

    def _simulate_fire_optimized(self, density: np.ndarray, spread_map: np.ndarray, wind_factors: dict, ignition_probs: Optional[np.ndarray]) -> np.ndarray:
        """Simulate one fire event using precomputed probability maps"""
        nrows, ncols = density.shape
        burned_mask = np.zeros((nrows, ncols), dtype=bool)
        
        if ignition_probs is None:
             return burned_mask

        # Sample ignition
        flat_index = self.random_generator.choice(nrows * ncols, p=ignition_probs)
        ignition_row, ignition_col = int(flat_index // ncols), int(flat_index % ncols)

        if density[ignition_row, ignition_col] <= 0.0:
            return burned_mask

        burned_mask[ignition_row, ignition_col] = True
        frontier = [(ignition_row, ignition_col)]

        # Cache standard random method to avoid attribute lookup in loop
        rand = self.random_generator.random

        while frontier:
            cell_i, cell_j = frontier.pop()
            
            # Inline neighbor check (avoids generator overhead)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    
                    neighbor_i, neighbor_j = cell_i + di, cell_j + dj
                    
                    if not (0 <= neighbor_i < nrows and 0 <= neighbor_j < ncols):
                        continue
                        
                    if burned_mask[neighbor_i, neighbor_j]:
                        continue
                    
                    # Look up precomputed spread probability
                    # prob = base_spread_of_dest * wind_factor_of_direction
                    prob = spread_map[neighbor_i, neighbor_j] * wind_factors[(di, dj)]
                    
                    if prob > 0.0 and rand() < prob:
                        burned_mask[neighbor_i, neighbor_j] = True
                        frontier.append((neighbor_i, neighbor_j))

        return burned_mask

    def _simulate_fire(self, density: np.ndarray, ignition: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Simulate one fire event (legacy wrapper for compatibility)."""
        spread_map, wind_factors = self._precompute_spread_dynamics(density)
        
        ignition_probs = None
        if ignition is None:
            ignition_weights = np.clip(density.astype(float), 0.0, 1.0)
            weight_sum = float(ignition_weights.sum())
            if weight_sum > 0:
                ignition_probs = ignition_weights.ravel() / weight_sum
        
        # If ignition is provided explicitly, handle it (not optimized path but supported)
        if ignition is not None:
            nrows, ncols = density.shape
            burned_mask = np.zeros((nrows, ncols), dtype=bool)
            ir, ic = ignition
            if density[ir, ic] > 0.0:
                burned_mask[ir, ic] = True
                frontier = [(ir, ic)]
                rand = self.random_generator.random
                while frontier:
                    cell_i, cell_j = frontier.pop()
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0: continue
                            ni, nj = cell_i + di, cell_j + dj
                            if 0 <= ni < nrows and 0 <= nj < ncols and not burned_mask[ni, nj]:
                                prob = spread_map[ni, nj] * wind_factors[(di, dj)]
                                if prob > 0 and rand() < prob:
                                    burned_mask[ni, nj] = True
                                    frontier.append((ni, nj))
            return burned_mask

        return self._simulate_fire_optimized(density, spread_map, wind_factors, ignition_probs)

    def _adjacent_mask(self, burned: np.ndarray) -> np.ndarray:
        """Return a boolean mask of cells that are burned or adjacent to burned.
        
        Optimized with NumPy vectorization (bitwise OR shifts)
        """
        rows, cols = burned.shape
        # Start with the burned cells themselves
        adjacent = burned.copy()
        
        # Shift burned mask in all 8 directions to find neighbors
        # We use slicing to shift: adjacent[dst_slice] |= burned[src_slice]
        
        # Down (i+1)
        adjacent[1:, :] |= burned[:-1, :]
        # Up (i-1)
        adjacent[:-1, :] |= burned[1:, :]
        # Right (j+1)
        adjacent[:, 1:] |= burned[:, :-1]
        # Left (j-1)
        adjacent[:, :-1] |= burned[:, 1:]
        
        # Diagonals
        # Down-Right
        adjacent[1:, 1:] |= burned[:-1, :-1]
        # Down-Left
        adjacent[1:, :-1] |= burned[:-1, 1:]
        # Up-Right
        adjacent[:-1, 1:] |= burned[1:, :-1]
        # Up-Left
        adjacent[:-1, :-1] |= burned[1:, 1:]
        
        return adjacent
