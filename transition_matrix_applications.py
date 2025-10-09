#!/usr/bin/env python3
"""
Transition Matrix Applications - Prediction and Forecasting

This script demonstrates how to use the learned transition matrices
for practical demand regime forecasting and planning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_latest_transition_matrix():
    """Load the most reliable transition matrix (N=40)."""
    # Using the N=40 results as they have the most samples
    P = np.array([[0.6, 0.4], 
                  [0.421053, 0.578947]])
    return P

def predict_regime_sequence(P, initial_state, n_steps):
    """
    Predict the most likely sequence of states given an initial state.
    
    Args:
        P: Transition matrix
        initial_state: Starting state (0 or 1)
        n_steps: Number of future periods to predict
    
    Returns:
        predicted_sequence: Most likely state sequence
        probabilities: Probability distributions at each step
    """
    current_dist = np.zeros(P.shape[0])
    current_dist[initial_state] = 1.0
    
    sequence = [initial_state]
    probabilities = [current_dist.copy()]
    
    for _ in range(n_steps):
        # Compute next period distribution
        current_dist = current_dist @ P
        probabilities.append(current_dist.copy())
        
        # Most likely state
        next_state = np.argmax(current_dist)
        sequence.append(next_state)
    
    return sequence, probabilities

def compute_n_step_transitions(P, n_steps):
    """Compute n-step transition probabilities."""
    P_n = np.linalg.matrix_power(P, n_steps)
    return P_n

def simulate_regime_paths(P, initial_state, n_steps, n_simulations=1000):
    """Simulate multiple possible regime paths."""
    paths = []
    
    for _ in range(n_simulations):
        path = [initial_state]
        current_state = initial_state
        
        for _ in range(n_steps):
            # Sample next state according to transition probabilities
            probs = P[current_state, :]
            next_state = np.random.choice(len(probs), p=probs)
            path.append(next_state)
            current_state = next_state
        
        paths.append(path)
    
    return np.array(paths)

def analyze_forecasting_accuracy():
    """Analyze how forecasting accuracy changes with horizon."""
    P = load_latest_transition_matrix()
    
    print("🔮 REGIME FORECASTING ANALYSIS")
    print("=" * 50)
    
    # Stationary distribution (long-run probabilities)
    eigenvals, eigenvecs = np.linalg.eig(P.T)
    stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
    stationary = eigenvecs[:, stationary_idx].real
    stationary = stationary / np.sum(stationary)
    
    print(f"Long-run regime probabilities:")
    print(f"  • Low Demand (State 0): {stationary[0]:.3f}")
    print(f"  • High Demand (State 1): {stationary[1]:.3f}")
    
    print(f"\n📊 FORECASTING HORIZONS")
    print("-" * 30)
    
    # Show how predictions evolve over time
    for initial_state in [0, 1]:
        state_name = "Low Demand" if initial_state == 0 else "High Demand"
        print(f"\nStarting from {state_name} (State {initial_state}):")
        
        for horizon in [1, 2, 3, 5, 10]:
            P_n = compute_n_step_transitions(P, horizon)
            prob_same = P_n[initial_state, initial_state]
            prob_other = P_n[initial_state, 1-initial_state]
            
            print(f"  {horizon} periods ahead: {prob_same:.3f} same regime, {prob_other:.3f} different regime")

def create_forecasting_example():
    """Create a practical forecasting example."""
    P = load_latest_transition_matrix()
    
    print(f"\n🎯 PRACTICAL FORECASTING EXAMPLE")
    print("=" * 50)
    
    # Scenario: Currently in high demand regime
    current_state = 1  # High demand
    forecast_horizon = 5
    
    print(f"Scenario: Currently experiencing HIGH demand regime")
    print(f"Question: What will demand regime look like over next {forecast_horizon} periods?")
    
    # Most likely sequence
    sequence, probabilities = predict_regime_sequence(P, current_state, forecast_horizon)
    
    print(f"\nMost Likely Regime Sequence:")
    regime_names = ["Low", "High"]
    for i, state in enumerate(sequence):
        if i == 0:
            print(f"  Period 0 (current): {regime_names[state]} demand")
        else:
            print(f"  Period {i}: {regime_names[state]} demand")
    
    print(f"\nProbability Distributions by Period:")
    for i, probs in enumerate(probabilities):
        if i == 0:
            print(f"  Period 0 (current): Low={probs[0]:.3f}, High={probs[1]:.3f}")
        else:
            print(f"  Period {i}: Low={probs[0]:.3f}, High={probs[1]:.3f}")
    
    # Simulation analysis
    print(f"\n📈 SIMULATION ANALYSIS ({1000} scenarios)")
    print("-" * 40)
    
    np.random.seed(42)  # For reproducible results
    paths = simulate_regime_paths(P, current_state, forecast_horizon, 1000)
    
    # Calculate regime percentages by period
    for period in range(1, forecast_horizon + 1):
        states_at_period = paths[:, period]
        low_pct = np.mean(states_at_period == 0) * 100
        high_pct = np.mean(states_at_period == 1) * 100
        
        print(f"  Period {period}: {low_pct:.1f}% Low demand, {high_pct:.1f}% High demand")

def business_recommendations():
    """Provide business recommendations based on transition matrix."""
    print(f"\n💼 BUSINESS RECOMMENDATIONS")
    print("=" * 50)
    
    print("1. SHORT-TERM PLANNING (1-2 periods):")
    print("   • Regime persistence is moderate (~58-60%)")
    print("   • Plan primarily for current regime but maintain flexibility")
    print("   • Keep safety stock for potential regime switches")
    
    print("\n2. MEDIUM-TERM PLANNING (3-5 periods):")
    print("   • Regime probabilities converge toward equilibrium")
    print("   • Plan for balanced mix of high/low demand scenarios")
    print("   • Implement adaptive procurement strategies")
    
    print("\n3. LONG-TERM PLANNING (10+ periods):")
    print("   • Use stationary distribution (51% Low, 49% High)")
    print("   • Design capacity for balanced demand patterns")
    print("   • Focus on robustness rather than regime-specific optimization")
    
    print("\n4. RISK MANAGEMENT:")
    print("   • ~40-42% chance of regime switches each period")
    print("   • Implement early warning systems for regime changes")
    print("   • Maintain operational flexibility for both regimes")

def main():
    """Run the complete forecasting analysis."""
    print("TRANSITION MATRIX APPLICATIONS")
    print("Using Sai2501 Environment")
    print("=" * 70)
    
    # Load and display the transition matrix
    P = load_latest_transition_matrix()
    print(f"\nLearned Transition Matrix (N=40 samples):")
    print(f"           To Low    To High")
    print(f"From Low   {P[0,0]:.3f}     {P[0,1]:.3f}")
    print(f"From High  {P[1,0]:.3f}     {P[1,1]:.3f}")
    
    # Run analyses
    analyze_forecasting_accuracy()
    create_forecasting_example()
    business_recommendations()
    
    print(f"\n" + "=" * 70)
    print("Analysis complete! Use these insights for demand planning.")

if __name__ == "__main__":
    main()