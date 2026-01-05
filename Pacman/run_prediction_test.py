#!/usr/bin/env python3
"""
PROJECT PACMAN: "SURPRISE" TEST (PREDICTIVE PROCESSING) - STATISTICAL UPGRADE
=============================================================================
Testing Free Energy Principle with Multi-Trial Rigor (N=20).

Protocol:
1. Measure Naive Response to B (20 trials).
2. Train on A (Heavy exposure).
3. Measure Deviant Response to B (20 trials).
4. Calculate MMN Significance (T-test).
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl

def get_activation(system, stimulus_indices):
    """Run one trial and get mean activation."""
    # Reset (keep weights, reset state)
    system._physics.v[:] = -65.0
    system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
    
    stim_plan = system.neurons.create_stim_plan()
    for i in stimulus_indices: 
        stim_plan.add_biphasic_pulse(i, 1000, 500, charge_balance=True)
    system.neurons.stim(stim_plan)
    
    trace = system.neurons.record(duration_sec=0.1)
    return np.mean(trace.samples)

def run_prediction_test():
    print("="*60)
    print("PREDICTION TEST: STATISTICAL RIGOR (N=20)")
    print("============================================================")
    
    # 1. SETUP
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True
    pacman._physics.cfg.learning_rate = 0.2
    
    n_trials = 20
    
    # Stimuli definitions
    indices_A = list(range(0, 10))
    indices_B = list(range(20, 30))
    
    # --- PHASE 1: MEASURE NAIVE RESPONSE TO B ---
    print("\n[PHASE 1] Measuring Naive Response to 'B' (Control)...")
    naive_scores = []
    print("      Sampling: ", end="", flush=True)
    for _ in range(n_trials):
        score = get_activation(pacman, indices_B)
        naive_scores.append(score)
        print(".", end="", flush=True)
    print(" Done.")
    
    # --- PHASE 2: TRAIN ON A ---
    print("\n[PHASE 2] Training on 'A' (Establishing Prior)...")
    print("      Training: ", end="", flush=True)
    for _ in range(40): # Increased training duration
        stim_plan = pacman.neurons.create_stim_plan()
        for i in indices_A:
            stim_plan.add_biphasic_pulse(i, 1000, 500, charge_balance=True)
        pacman.neurons.stim(stim_plan)
        pacman._physics.step() # Process
        if _ % 2 == 0: print(".", end="", flush=True)
    print(" Done.")
    
    # --- PHASE 3: MEASURE DEVIANT RESPONSE TO B ---
    print("\n[PHASE 3] Measuring Deviant Response to 'B' (Violation)...")
    deviant_scores = []
    print("      Sampling: ", end="", flush=True)
    for _ in range(n_trials):
        # We need to maintain the "Context of A" somewhat?
        # Actually FEP says synaptic weights changed, so B alone should trigger error.
        score = get_activation(pacman, indices_B)
        deviant_scores.append(score)
        print(".", end="", flush=True)
    print(" Done.")
    
    # 3. ANALYSIS
    naive_mean, naive_std = np.mean(naive_scores), np.std(naive_scores)
    dev_mean, dev_std = np.mean(deviant_scores), np.std(deviant_scores)
    
    print("\n" + "="*60)
    print("RESULTS & STATISTICAL ANALYSIS")
    print("="*60)
    
    print(f"Naive Response:   {naive_mean:.4f} ± {naive_std:.4f}")
    print(f"Deviant Response: {dev_mean:.4f} ± {dev_std:.4f}")
    
    mmn = dev_mean - naive_mean
    print(f"MMN (Difference): {mmn:.4f}")
    
    # T-Test
    t_stat, p_val = stats.ttest_ind(deviant_scores, naive_scores, equal_var=False)
    print(f"T-Statistic: {t_stat:.2f}")
    print(f"P-Value:     {p_val:.4e}")
    
    print("-" * 40)
    
    if p_val < 0.05 and mmn > 0:
        print("VERDICT: PACMAN IS SURPRISED (Significant).")
        print("         Prediction Error confirmed (p < 0.05).")
    elif p_val < 0.05 and mmn < 0:
        print("VERDICT: PACMAN IS BORED (Significant Negative MMN).")
        print("         Adaptation/Fatigue dominance?")
    else:
        print("VERDICT: NO SIGNIFICANT SURPRISE.")
        print("         Difference likely due to noise.")

if __name__ == "__main__":
    run_prediction_test()
