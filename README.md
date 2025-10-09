# Distribution-Shift: Regime-Switching Procurement Optimization

A regime-switching distributionally robust optimization (DRO) framework for procurement planning under demand uncertainty with Hidden Markov Model (HMM) state estimation.

## 🔍 Overview

This repository implements a procurement optimization model that:
- **Identifies demand regimes** using Hidden Markov Models (HMM)
- **Learns transition dynamics** between demand states
- **Optimizes procurement decisions** using distributionally robust optimization
- **Compares regime-switching vs. pooled approaches** for various sample sizes

## 📊 Key Features

### 1. **Markov Chain Transition Matrix Learning**
The system automatically learns transition probabilities between demand regimes:

```python
# Example learned transition matrix (N=40 samples)
P = [[0.600, 0.400],     # From Low Demand  → [Low, High]
     [0.421, 0.579]]     # From High Demand → [Low, High]
```

**Key Insights:**
- **Regime Persistence**: 60% chance to stay in low demand, 58% chance to stay in high demand
- **Switching Probability**: ~40-42% chance of regime changes each period
- **Long-run Equilibrium**: 51.3% low demand, 48.7% high demand

### 2. **Distributionally Robust Optimization**
- McCormick relaxation for bilinear terms
- Wasserstein ambiguity sets with cross-validated epsilon tuning
- CVXPY implementation with multiple solver backends

### 3. **Comprehensive Analysis Tools**
- Transition matrix visualization and analysis
- Out-of-sample performance evaluation
- Regime forecasting and business insights

## 🚀 Quick Start

### Prerequisites
```bash
# Activate your conda environment
conda activate Sai2501

# Required packages
pip install numpy pandas matplotlib seaborn cvxpy hmmlearn openpyxl
```

### Basic Usage
```bash
# Run the main optimization with regime-switching analysis
python rs_base.py

# Analyze learned transition matrices
python transition_matrix_analysis.py

# Get business insights and forecasting applications
python transition_matrix_applications.py

# Create visualizations
python plot_base.py
```

## 📁 Repository Structure

```
├── rs_base.py                          # Main optimization script
├── oos_analys_RS.py                    # Out-of-sample analysis
├── transition_matrix_analysis.py       # Transition matrix deep dive
├── transition_matrix_applications.py   # Forecasting applications
├── transition_matrix_summary.py        # Business insights summary
├── plot_base.py                        # Visualization tools
├── visualization.py                    # Additional plotting
├── input_parameters.xlsx               # Model parameters
├── results/
│   ├── results_RS_vs_noRS.xlsx        # Comparison results
│   ├── results_comparison.xlsx        # Performance deltas
│   ├── transition_analysis_*.csv      # Learned matrices
│   └── transition_matrices_analysis.png # Visualizations
└── plots/                             # Generated plots
```

## 🔬 Methodology

### 1. **Hidden Markov Model Training**
```python
def train_hmm_with_sorted_states(data, n_components=2, random_state=42):
    """
    Train Gaussian HMM and sort states by mean for stable indexing.
    Returns: model, remapped_states, sorted_transmat, sorted_means, state_counts
    """
```

**Features:**
- Gaussian emissions with configurable covariance
- State sorting by mean for interpretable results
- Cross-validation for robust parameter estimation

### 2. **Demand Generation**
Supports multiple demand distributions:
- **Gaussian**: Correlated demand with AR(1) structure
- **Gamma**: Heavy-tailed demand patterns
- **Log-normal**: Skewed demand distributions
- **Weibull**: Extreme value distributions
- **Mixture**: Combined distribution regimes

### 3. **Optimization Framework**
```python
def optimize_McC_rs(self, K, epsilon, r, Nk, solver_preference=None):
    """
    McCormick DRO formulation with regime-switching.
    
    Args:
        K: Number of regimes
        epsilon: Wasserstein radius per regime
        r: Regime probability weights
        Nk: Sample counts per regime
    """
```

## 📈 Results Analysis

### Transition Matrix Properties

| Sample Size | Low→Low | Low→High | High→Low | High→High | Stationary [Low, High] |
|-------------|---------|----------|----------|-----------|----------------------|
| N=10        | 0.500   | 0.500    | 0.400    | 0.600     | [0.444, 0.556]      |
| N=20        | 0.500   | 0.500    | 0.308    | 0.692     | [0.381, 0.619]      |
| N=40        | 0.600   | 0.400    | 0.421    | 0.579     | [0.513, 0.487]      |

### Performance Comparison (RS vs No-RS)

| N  | Weight | RS Cost   | No-RS Cost | Improvement |
|----|--------|-----------|------------|-------------|
| 40 | 0.3    | 385,059   | 385,236    | 0.05%       |
| 40 | 0.5    | 381,088   | 381,283    | 0.05%       |
| 40 | 0.7    | 379,583   | 379,814    | 0.06%       |

## 🎯 Business Applications

### 1. **Short-term Planning (1-2 periods)**
- **Regime persistence**: ~58-60% probability
- **Strategy**: Plan for current regime, maintain flexibility
- **Risk**: Keep safety stock for potential switches

### 2. **Medium-term Planning (3-5 periods)**
- **Convergence**: Probabilities approach equilibrium
- **Strategy**: Balance high/low demand scenarios
- **Implementation**: Adaptive procurement strategies

### 3. **Long-term Planning (10+ periods)**
- **Equilibrium**: Use stationary distribution (51% Low, 49% High)
- **Strategy**: Design for balanced demand patterns
- **Focus**: Robustness over regime-specific optimization

### 4. **Forecasting Example**
Starting from high demand regime:
```
Period 1: 44.1% Low, 55.9% High
Period 2: 49.8% Low, 50.2% High
Period 3: 51.4% Low, 48.6% High
Period 5: 51.3% Low, 48.7% High (equilibrium)
```

## 🔧 Configuration

### Model Parameters (`input_parameters.xlsx`)
- **Costs**: Holding cost (h=5), Backlog cost (b=20)
- **Initial Inventory**: I₀=1800, B₀=0
- **Suppliers**: Order costs, lead times, capacities, quality levels
- **Prices**: Time-varying supplier pricing

### Experiment Settings
```python
# Sample sizes for analysis
input_sample_no = [10, 20, 40]

# Cross-validation parameters
k_fold = 5
epsilon_grid = [0, 30, 60]

# Demand mixture weights for OoS evaluation
oos_weights = [0.3, 0.5, 0.7]
```

## 📊 Visualization Outputs

1. **Transition Matrix Heatmaps**: Visual representation of learned probabilities
2. **Stationary Distribution Plots**: Long-run regime equilibrium
3. **Performance Comparison Charts**: RS vs No-RS cost analysis
4. **Regime Evolution Plots**: Cross-validation stability analysis

## 🔍 Mathematical Properties

### Transition Matrix Validation
- ✅ **Row Stochastic**: Each row sums to 1.0
- ✅ **Irreducible**: All states reachable from all other states
- ✅ **Aperiodic**: No cyclical patterns
- ✅ **Fast Mixing**: Quick convergence (mixing time < 1 period)

### Stationary Distribution
Computed as the principal eigenvector of P^T:
```python
eigenvals, eigenvecs = np.linalg.eig(P.T)
stationary = eigenvecs[:, 0].real / sum(eigenvecs[:, 0].real)
```

## 🛠️ Development

### Adding New Demand Distributions
```python
def get_custom_demand(time_horizon, params, M, seed=None):
    """Template for adding new demand distributions."""
    # Implementation here
    return pd.DataFrame(samples)
```

### Extending Optimization Models
The framework supports adding new DRO formulations by extending the `Models` class in `rs_base.py`.

### Custom Analysis Scripts
Use the transition matrix API for custom forecasting:
```python
from rs_base import train_hmm_with_sorted_states
model, _, transmat, _, _ = train_hmm_with_sorted_states(data, n_components=2)
```

## 📚 References

1. **Distributionally Robust Optimization**: Wasserstein ambiguity sets
2. **Hidden Markov Models**: Regime identification and transition learning
3. **McCormick Relaxation**: Bilinear term linearization
4. **Procurement Optimization**: Multi-supplier, multi-period planning

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Contact

- **Repository**: [Distribution-Shift](https://github.com/pauldebdeep9/Distribution-Shift)
- **Author**: Debdeep Paul
- **Environment**: Sai2501

---

*Generated on October 10, 2025 - Comprehensive analysis of regime-switching procurement optimization with learned Markov chain dynamics.*