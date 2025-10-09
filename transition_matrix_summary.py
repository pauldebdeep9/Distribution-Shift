#!/usr/bin/env python3
"""
Summary of Learned Transition Matrices from Markov Chain Model

This script provides a comprehensive summary of the transition matrices learned
by the Hidden Markov Model (HMM) in the regime-switching procurement optimization.
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 70)
    print("LEARNED TRANSITION MATRICES - SUMMARY REPORT")
    print("=" * 70)
    
    print("\n🔍 WHAT ARE THESE TRANSITION MATRICES?")
    print("-" * 50)
    print("The transition matrices represent the learned probabilistic relationships")
    print("between demand regimes in your procurement optimization model.")
    print("• State 0: Low demand regime (Gaussian distribution)")
    print("• State 1: High demand regime (Gamma distribution)")
    print("• P(i,j) = Probability of transitioning from state i to state j")
    
    # Summary of results from the run
    results = {
        "N=10": {
            "matrix": np.array([[0.5, 0.5], [0.4, 0.6]]),
            "state_counts": [4, 6],
            "stationary": [0.4444, 0.5556]
        },
        "N=20": {
            "matrix": np.array([[0.5, 0.5], [0.307692, 0.692308]]),
            "state_counts": [7, 13],
            "stationary": [0.3810, 0.6190]
        },
        "N=40": {
            "matrix": np.array([[0.6, 0.4], [0.421053, 0.578947]]),
            "state_counts": [20, 20],
            "stationary": [0.5128, 0.4872]
        }
    }
    
    print("\n📈 RESULTS BY SAMPLE SIZE")
    print("-" * 50)
    
    for sample_size, data in results.items():
        matrix = data["matrix"]
        counts = data["state_counts"]
        stationary = data["stationary"]
        
        print(f"\n{sample_size} (State 0: {counts[0]} samples, State 1: {counts[1]} samples)")
        print(f"Transition Matrix:")
        print(f"  From Low→Low:  {matrix[0,0]:.3f}   From Low→High:  {matrix[0,1]:.3f}")
        print(f"  From High→Low: {matrix[1,0]:.3f}   From High→High: {matrix[1,1]:.3f}")
        print(f"Long-run probabilities: Low={stationary[0]:.3f}, High={stationary[1]:.3f}")
    
    print("\n📊 KEY INSIGHTS")
    print("-" * 50)
    
    print("1. REGIME PERSISTENCE:")
    for sample_size, data in results.items():
        matrix = data["matrix"]
        low_persist = matrix[0,0]
        high_persist = matrix[1,1]
        print(f"   • {sample_size}: Low regime stays {low_persist:.1%}, High regime stays {high_persist:.1%}")
    
    print("\n2. REGIME SWITCHING PATTERNS:")
    for sample_size, data in results.items():
        matrix = data["matrix"]
        low_to_high = matrix[0,1]
        high_to_low = matrix[1,0]
        print(f"   • {sample_size}: Low→High {low_to_high:.1%}, High→Low {high_to_low:.1%}")
    
    print("\n3. LONG-RUN EQUILIBRIUM:")
    for sample_size, data in results.items():
        stationary = data["stationary"]
        print(f"   • {sample_size}: {stationary[0]:.1%} in Low regime, {stationary[1]:.1%} in High regime")
    
    print("\n4. SAMPLE SIZE EFFECTS:")
    print("   • With more samples (N=10→40), transition patterns stabilize")
    print("   • N=40 shows more balanced regime distribution (50-50 vs 40-60)")
    print("   • Larger samples provide more reliable transition estimates")
    
    print("\n🔧 MATHEMATICAL PROPERTIES")
    print("-" * 50)
    print("All matrices are:")
    print("• Row stochastic: Each row sums to 1.0 (valid probability distributions)")
    print("• Irreducible: All states can reach all other states")
    print("• Aperiodic: No cyclical patterns in state transitions")
    print("• Fast mixing: Low second eigenvalues → quick convergence to equilibrium")
    
    print("\n💼 BUSINESS IMPLICATIONS")
    print("-" * 50)
    print("1. DEMAND FORECASTING:")
    print("   • If currently in low demand, ~50-60% chance to stay low next period")
    print("   • If currently in high demand, ~58-69% chance to stay high next period")
    
    print("\n2. INVENTORY PLANNING:")
    print("   • Regime persistence suggests preparing for current regime to continue")
    print("   • But significant switching probabilities require flexibility")
    
    print("\n3. PROCUREMENT STRATEGY:")
    print("   • Balance between regime-specific optimization and robustness")
    print("   • Higher sample sizes (N=40) provide more reliable regime identification")
    
    print("\n4. RISK MANAGEMENT:")
    print("   • ~40-50% probability of regime switches requires contingency planning")
    print("   • Long-run equilibrium shows roughly balanced demand patterns")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()