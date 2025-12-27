# Wildfire Risk Simulation

A stochastic wildfire simulation based on percolation theory and Monte Carlo methods, applied to real-world forest density data from the Hansen Global Forest Change dataset.

This project models wildfire spread as a discrete-time cellular automaton on a spatial grid. It demonstrates how fuel connectivity drives large-scale fire risk and evaluates the effectiveness of mechanical thinning as a mitigation strategy.

## Key Findings

- **Supercritical State**: The analysis of Northern Canada (60N, 120W) reveals that the landscape is in a "supercritical" percolation state, where high fuel connectivity allows small ignitions to grow into system-spanning fires.
- **Phase Transition**: A uniform thinning strategy (reducing density to 60%) triggers a phase transition, fragmenting the fuel network and effectively eliminating catastrophic fire risk.
- **Robustness**: These results hold across a 20-year time series (2000–2019), suggesting that connectivity-based mitigation is a robust policy tool.

## Features

- **Stochastic Spread Model**: Simulates fire propagation based on local fuel density, wind conditions, and random chance using a Moore neighborhood (8-neighbor connectivity).
- **Real-World Data**: Integrates 20 years of forest cover data from the [Hansen Global Forest Change](https://earthenginepartners.appspot.com/science-2013-global-forest) dataset.
- **Monte Carlo Analysis**: Estimates burn probabilities and risk distributions through thousands of repeated simulation runs.
- **Mitigation Evaluation**: Compares baseline scenarios against thinning strategies to quantify risk reduction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marilevay/wildfire-simulation.git
   cd wildfire-simulation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project requires Python 3.10+.*

## Usage

### Running Simulations
The primary entry point for running experiments and generating visualizations is the Jupyter Notebook:

```bash
jupyter notebook wildfire-visualization.ipynb
```

This notebook will:
1. Download the necessary Hansen GFC GeoTIFF tiles (cached locally in `data/`).
2. Preprocess the data into simulation grids.
3. Run Monte Carlo simulations for baseline and thinning scenarios.
4. Generate risk maps, histograms, and time-series plots.

### Package Structure
The core logic is organized in the `simulation/` package:

- `simulation/forest.py`: Manages the landscape grid and fuel density state.
- `simulation/monte_carlo_model.py`: Implements the stochastic fire spread algorithm and optimized percolation logic.
- `simulation/data_collector.py`: Handles aggregation of results and statistical metrics (confidence intervals).

## Project Structure

```
.
├── data/                       # Cached simulation data
├── simulation/                 # Core Python simulation package
│   ├── __init__.py
│   ├── data_collector.py
│   ├── forest.py
│   └── monte_carlo_model.py
├── cs166-final-project.pdf     # Final PDF report
├── wildfire-visualization.ipynb # Main analysis notebook
├── requirements.txt            # Python dependencies
└── README.md
```

## Theoretical Background

The model assumes that fire spread is a percolation process. For a cell with density $d$, the probability of transmitting fire to a neighbor depends on $d^\alpha$ and wind alignment.

- **Criticality**: Large fires emerge only when the landscape density exceeds a critical percolation threshold $p_c$.
- **Uncertainty**: Because spread is probabilistic, risk is estimated as the mean burned fraction over $N$ Monte Carlo runs (typically $N=100-200$ for rapid analysis, $N>2000$ for high precision).

## Credits

**Author**: Marina Levay  
**Course**: CS166 (Minerva University)  
**Data**: Hansen/UMD/Google/USGS/NASA Global Forest Change dataset.

## License

See the `LICENSE` file for further details.

