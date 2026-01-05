#!/usr/bin/env python3
"""
FEATURE BINDING TEST (Integration)
===================================
Testing whether Pacman unifies distributed features (Color + Position)
into coherent object representations.

Theory: Treisman, Singer - The Binding Problem.

Protocol:
1. Stimulus: Present Ghost with Color (neurons 50-60) + Position (neurons 100-110).
2. Measure: Correlation between Color and Position neuron activity.

Prediction:
- Pacman: Binding Index > 0.6 (features unified)
- Ghost: Binding Index ≈ 0 (features independent)
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble

def run_binding_trial(system, color_indices, position_indices, color_stim, position_stim):
    """Run a single binding trial."""
    # Reset
    system._physics.v[:] = -65.0
    system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
    
    # Inject both features simultaneously
    external = np.zeros(system._physics.cfg.n_neurons)
    external[color_indices] = color_stim
    external[position_indices] = position_stim
    
    # Run and collect traces
    color_trace = []
    position_trace = []
    
    for _ in range(100):
        system._physics.step(external)
        color_v = system._physics.v[color_indices]
        position_v = system._physics.v[position_indices]
        color_trace.append(np.mean(color_v))
        position_trace.append(np.mean(position_v))
    
    # Compute correlation between color and position activity
    if np.std(color_trace) > 0 and np.std(position_trace) > 0:
        binding, _ = stats.pearsonr(color_trace, position_trace)
    else:
        binding = 0.0
    
    return binding, color_trace, position_trace

def run_ghost_binding(ghost_brain, color_input, position_input):
    """
    Ghost has no temporal dynamics, but we can test if color and position
    inputs influence output independently or together.
    """
    # Present both features
    combined_input = np.array([color_input, position_input, 0.0, 0.0])
    output1, h1 = ghost_brain.forward(combined_input)
    
    # Present color only
    color_only = np.array([color_input, 0.0, 0.0, 0.0])
    output2, h2 = ghost_brain.forward(color_only)
    
    # Present position only
    position_only = np.array([0.0, position_input, 0.0, 0.0])
    output3, h3 = ghost_brain.forward(position_only)
    
    # Binding = how much combined differs from sum of parts (superadditivity)
    sum_of_parts = (np.mean(h2) + np.mean(h3)) / 2
    combined = np.mean(h1)
    
    # Binding index: if combined > sum of parts, there's integration
    binding = (combined - sum_of_parts) / max(combined, 0.001)
    
    return binding

def run_feature_binding_test():
    print("="*60)
    print("FEATURE BINDING TEST: UNIFIED PERCEPTION")
    print("============================================================")
    
    # 1. SETUP
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True
    
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    
    # Feature-specific neurons
    color_indices = list(range(50, 70))      # "Color" neurons
    position_indices = list(range(100, 120)) # "Position" neurons
    
    n_trials = 20
    
    pacman_binding = []
    ghost_binding = []
    
    print(f"\n[TEST] Running {n_trials} trials...")
    
    # Run Pacman Trials
    print("      Pacman: ", end="", flush=True)
    for _ in range(n_trials):
        # Vary stimulus intensity slightly for variability
        color_stim = 400 + np.random.normal(0, 50)
        position_stim = 400 + np.random.normal(0, 50)
        
        b, ct, pt = run_binding_trial(pacman, color_indices, position_indices, color_stim, position_stim)
        pacman_binding.append(b)
        print(".", end="", flush=True)
    print(" Done.")
    
    # Run Ghost Trials
    print("      Blinky: ", end="", flush=True)
    for _ in range(n_trials):
        color_input = 3.0 + np.random.normal(0, 0.5)
        position_input = 3.0 + np.random.normal(0, 0.5)
        
        b = run_ghost_binding(blinky, color_input, position_input)
        ghost_binding.append(b)
        print(".", end="", flush=True)
    print(" Done.")
    
    # 2. ANALYSIS
    p_mean, p_std = np.mean(pacman_binding), np.std(pacman_binding)
    g_mean, g_std = np.mean(ghost_binding), np.std(ghost_binding)
    
    print("\n" + "="*60)
    print("RESULTS & STATISTICAL ANALYSIS")
    print("="*60)
    
    print(f"PACMAN Binding Index: {p_mean:.3f} ± {p_std:.3f}")
    print(f"GHOST Binding Index:  {g_mean:.3f} ± {g_std:.3f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(pacman_binding, ghost_binding, equal_var=False)
    print(f"T-Statistic: {t_stat:.2f}")
    print(f"P-Value:     {p_val:.4e}")
    
    # One-sample test against threshold
    t_thresh, p_thresh = stats.ttest_1samp(pacman_binding, 0.6)
    
    print("-" * 40)
    
    # Verdict
    if p_mean > 0.6:
        print("PACMAN: FEATURE BINDING CONFIRMED.")
        print("        Color and Position are unified (Correlation > 0.6).")
    elif p_mean > 0.3:
        print("PACMAN: PARTIAL BINDING.")
        print("        Some correlation but below threshold.")
    else:
        print("PACMAN: BINDING FAILED.")
        print("        Features remain independent.")
    
    if abs(g_mean) < 0.3:
        print("GHOST:  INDEPENDENT FEATURES (Expected).")
    else:
        print("GHOST:  UNEXPECTED BINDING!")

if __name__ == "__main__":
    run_feature_binding_test()
