#!/usr/bin/env python3
"""
Transition Matrix Analysis for Markov Chain Model
Visualizes and analyzes the learned transition matrices from the regime-switching model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import json

def load_transition_matrices():
    """Load all transition matrices from saved files."""
    transition_files = glob("transition_analysis_N*_K2_*.csv")
    summary_files = glob("transition_analysis_summary_N*_K2_*.csv")
    
    matrices = {}
    summaries = {}
    
    for tfile in transition_files:
        # Extract N from filename
        parts = tfile.split('_')
        n_part = [p for p in parts if p.startswith('N')][0]
        N = int(n_part[1:])
        
        # Load transition matrix
        df = pd.read_csv(tfile, index_col=0)
        matrices[N] = df.values
        
        # Load corresponding summary
        sfile = tfile.replace('transition_analysis_', 'transition_analysis_summary_')
        if sfile in summary_files:
            summary_df = pd.read_csv(sfile)
            summaries[N] = summary_df.iloc[0].to_dict()
    
    return matrices, summaries

def analyze_transition_properties(transition_matrix):
    """Analyze properties of a transition matrix."""
    P = np.array(transition_matrix)
    
    # Basic properties
    properties = {
        'is_doubly_stochastic': np.allclose(P.sum(axis=0), 1.0),
        'is_row_stochastic': np.allclose(P.sum(axis=1), 1.0),
    }
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(P.T)
    
    # Find stationary distribution (real part of eigenvector with eigenvalue 1)
    stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
    stationary = eigenvecs[:, stationary_idx].real
    stationary = stationary / np.sum(stationary)
    properties['stationary_distribution'] = stationary
    
    # Second largest eigenvalue (mixing time related)
    sorted_eigenvals = np.sort(np.abs(eigenvals))[::-1]
    properties['second_eigenvalue'] = sorted_eigenvals[1] if len(sorted_eigenvals) > 1 else 0
    
    # Mixing time approximation
    if properties['second_eigenvalue'] > 0 and properties['second_eigenvalue'] < 1:
        properties['mixing_time_approx'] = -1 / np.log(properties['second_eigenvalue'])
    else:
        properties['mixing_time_approx'] = np.inf
    
    return properties

def visualize_transition_matrices(matrices, summaries):
    """Create visualizations for transition matrices."""
    n_matrices = len(matrices)
    fig, axes = plt.subplots(2, n_matrices, figsize=(5*n_matrices, 10))
    
    if n_matrices == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (N, matrix) in enumerate(sorted(matrices.items())):
        # Heatmap of transition matrix
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=[f'State {j}' for j in range(matrix.shape[1])],
                   yticklabels=[f'State {j}' for j in range(matrix.shape[0])],
                   ax=axes[0, i], cbar_kws={'label': 'Transition Probability'})
        axes[0, i].set_title(f'Transition Matrix (N={N})')
        axes[0, i].set_xlabel('To State')
        axes[0, i].set_ylabel('From State')
        
        # Bar plot of stationary distribution
        props = analyze_transition_properties(matrix)
        stationary = props['stationary_distribution']
        
        bars = axes[1, i].bar(range(len(stationary)), stationary, 
                             color=['lightblue', 'lightcoral'])
        axes[1, i].set_title(f'Stationary Distribution (N={N})')
        axes[1, i].set_xlabel('State')
        axes[1, i].set_ylabel('Probability')
        axes[1, i].set_xticks(range(len(stationary)))
        axes[1, i].set_xticklabels([f'State {j}' for j in range(len(stationary))])
        
        # Add values on bars
        for bar, val in zip(bars, stationary):
            axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('transition_matrices_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(matrices, summaries):
    """Print detailed analysis of all transition matrices."""
    print("=" * 80)
    print("DETAILED TRANSITION MATRIX ANALYSIS")
    print("=" * 80)
    
    for N in sorted(matrices.keys()):
        matrix = matrices[N]
        summary = summaries.get(N, {})
        props = analyze_transition_properties(matrix)
        
        print(f"\n📊 ANALYSIS FOR N = {N}")
        print("-" * 50)
        
        print("Transition Matrix:")
        print(f"  {matrix}")
        
        print(f"\nMatrix Properties:")
        print(f"  • Row stochastic: {props['is_row_stochastic']}")
        print(f"  • Doubly stochastic: {props['is_doubly_stochastic']}")
        print(f"  • Second eigenvalue: {props['second_eigenvalue']:.4f}")
        print(f"  • Mixing time (approx): {props['mixing_time_approx']:.2f}" if np.isfinite(props['mixing_time_approx']) else "  • Mixing time (approx): ∞")
        
        print(f"\nStationary Distribution:")
        for i, prob in enumerate(props['stationary_distribution']):
            print(f"  • State {i}: {prob:.4f}")
        
        if summary:
            state_means = eval(summary.get('State_Means', '[]'))
            state_counts = eval(summary.get('State_Counts', '[]'))
            print(f"\nState Information:")
            for i, (mean, count) in enumerate(zip(state_means, state_counts)):
                print(f"  • State {i}: mean={mean:.4f}, count={count} samples")
        
        # Transition interpretations
        print(f"\nTransition Interpretations:")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    print(f"  • P(State {i} → State {j}) = {matrix[i,j]:.3f}")
                else:
                    print(f"  • P(State {i} → State {i}) = {matrix[i,j]:.3f} (self-transition)")

def main():
    """Main function to run the analysis."""
    print("Loading transition matrices...")
    matrices, summaries = load_transition_matrices()
    
    if not matrices:
        print("No transition matrix files found!")
        print("Make sure you've run rs_base.py with K=2 to generate the matrices.")
        return
    
    print(f"Found {len(matrices)} transition matrices for sample sizes: {sorted(matrices.keys())}")
    
    # Print detailed analysis
    print_detailed_analysis(matrices, summaries)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_transition_matrices(matrices, summaries)
    
    print("\n✅ Analysis complete! Check 'transition_matrices_analysis.png' for visualizations.")

if __name__ == "__main__":
    main()