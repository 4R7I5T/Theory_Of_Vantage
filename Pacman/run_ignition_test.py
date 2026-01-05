#!/usr/bin/env python3
"""
PROJECT PACMAN: "GLOBAL IGNITION" TEST (GWT) - STATISTICAL UPGRADE
==================================================================
Testing Global Workspace Theory with Multi-Trial Rigor (N=20).

Protocol:
1. Stimulus: Inject "Pellet Detected" signal into sensory cortex.
2. Measure: Broadcast Ratio across 20 trials.
3. Prediction:
   - Pacman: Mean Broadcast > 50% (Robust Ignition).
   - Ghosts: Mean Broadcast < 20% (Robust Locality).
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble

def calculate_broadcast_ratio(trace_matrix, threshold=-30.0):
    """Calculate % of unique neurons that activated (spiked)."""
    spikes = (trace_matrix > threshold).astype(int)
    active_mask = np.any(spikes, axis=1) # Across time (axis=1) for each electrode/neuron
    return np.sum(active_mask) / len(active_mask)

def calculate_ghost_broadcast(ghost_brain, input_vector):
    _, h = ghost_brain.forward(input_vector)
    n_active = np.sum(h > 0.01)
    return n_active / len(h)

def run_trial(system, system_type="pacman"):
    if system_type == "pacman":
        # Reset
        system._physics.v[:] = -65.0
        system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
        
        # Stimulate Sensory Cortex (0-20)
        stim_plan = system.neurons.create_stim_plan()
        for elec in range(20):
            stim_plan.add_biphasic_pulse(elec, 1000, 500, charge_balance=True)
        system.neurons.stim(stim_plan)
        
        # Record
        trace = system.neurons.record(duration_sec=0.2)
        return calculate_broadcast_ratio(trace.samples)
        
    elif system_type == "ghost":
        # Feedforward is deterministic unless inputs change
        # But we can add input noise to verify robustness
        noisy_input = np.ones(4) * 5.0 + np.random.normal(0, 0.5, 4)
        return calculate_ghost_broadcast(system, noisy_input)

def run_ignition_test():
    print("="*60)
    print("GLOBAL IGNITION TEST: STATISTICAL RIGOR (N=20)")
    print("============================================================")
    
    # 1. SETUP
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True 
    
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    
    n_trials = 20
    pacman_scores = []
    ghost_scores = []
    
    print(f"\n[TEST] Running {n_trials} trials per subject...")
    
    # Run Pacman Trials
    print("      Pacman: ", end="", flush=True)
    for i in range(n_trials):
        score = run_trial(pacman, "pacman")
        pacman_scores.append(score)
        print(".", end="", flush=True)
    print(f" Done.")
    
    # Run Ghost Trials
    print("      Blinky: ", end="", flush=True)
    for i in range(n_trials):
        score = run_trial(blinky, "ghost")
        ghost_scores.append(score)
        print(".", end="", flush=True)
    print(f" Done.")
    
    # 2. ANALYSIS
    p_mean, p_std = np.mean(pacman_scores), np.std(pacman_scores)
    g_mean, g_std = np.mean(ghost_scores), np.std(ghost_scores)
    
    print("\n" + "-"*40)
    print(f"PACMAN broadcast: {p_mean:.1%} ± {p_std:.1%}")
    print(f"GHOST broadcast:  {g_mean:.1%} ± {g_std:.1%}")
    print("-" * 40)
    
    # 3. VERDICT
    # T-test equivalent (simple separation check)
    separation = p_mean - g_mean
    combined_std = np.sqrt(p_std**2 + g_std**2)
    z_score = separation / combined_std if combined_std > 0 else 999
    
    print(f"\nZ-Score of Separation: {z_score:.2f}")
    
    if p_mean > 0.5 and z_score > 3.0:
        print("VERDICT: GWT SUPPORTED (Statistically Significant).")
        print("         Integrated architecture reliably ignites.")
    else:
        print("VERDICT: INCONCLUSIVE.")
        if p_std > 0.2:
            print("         Warning: High variance in Pacman (Unstable Ignition).")

if __name__ == "__main__":
    run_ignition_test()
