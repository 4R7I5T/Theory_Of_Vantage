#!/usr/bin/env python3
"""
WORKING MEMORY TEST (Persistence)
==================================
Testing whether Pacman maintains information in sustained neural activity
after the stimulus is removed.

Theory: Baddeley, Goldman-Rakic - Working memory requires sustained activity.

Protocol:
1. Stimulus Phase (200ms): Present signal to sensory neurons.
2. Delay Phase (500ms): Remove stimulus completely.
3. Measure: Compare activity during delay vs stimulus.

Prediction:
- Pacman (Recurrent): Persistence Index > 0.3 (memory maintained)
- Ghost (Feedforward): Persistence Index < 0.1 (immediate decay)
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble

def run_trial_pacman(system, stimulus_indices, stim_duration_ms=200, delay_duration_ms=500):
    """Run a single working memory trial on Pacman."""
    dt = system._physics.cfg.dt
    stim_steps = int(stim_duration_ms / dt)
    delay_steps = int(delay_duration_ms / dt)
    
    # Reset state
    system._physics.v[:] = -65.0
    system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
    
    # Memory neurons (middle layer)
    memory_indices = list(range(100, 200))
    
    stim_activity = []
    delay_activity = []
    
    # STIMULUS PHASE
    external = np.zeros(system._physics.cfg.n_neurons)
    external[stimulus_indices] = 500.0
    
    for _ in range(stim_steps):
        system._physics.step(external)
        # Record memory neuron activity
        mem_v = system._physics.v[memory_indices]
        stim_activity.append(np.mean(mem_v > -30))  # Fraction active
    
    # DELAY PHASE (No input)
    for _ in range(delay_steps):
        system._physics.step(None)  # No external input
        mem_v = system._physics.v[memory_indices]
        delay_activity.append(np.mean(mem_v > -30))
    
    # Compute Persistence Index
    stim_mean = np.mean(stim_activity) if stim_activity else 0.001
    delay_mean = np.mean(delay_activity) if delay_activity else 0
    
    persistence = delay_mean / max(stim_mean, 0.001)
    
    return persistence, stim_mean, delay_mean

def run_trial_ghost(ghost_brain, input_vector):
    """
    Ghost is feedforward - no temporal dynamics.
    We simulate by checking if output decays without input.
    """
    # With input
    output_stim, h_stim = ghost_brain.forward(input_vector)
    stim_activity = np.mean(h_stim > 0.01)
    
    # Without input (delay)
    zero_input = np.zeros_like(input_vector)
    output_delay, h_delay = ghost_brain.forward(zero_input)
    delay_activity = np.mean(h_delay > 0.01)
    
    persistence = delay_activity / max(stim_activity, 0.001)
    
    return persistence, stim_activity, delay_activity

def run_working_memory_test():
    print("="*60)
    print("WORKING MEMORY TEST: SUSTAINED ACTIVITY")
    print("============================================================")
    
    # 1. SETUP
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True
    
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    
    stimulus_indices = list(range(0, 30))  # Sensory neurons
    ghost_input = np.array([5.0, 2.0, 0.0, 0.0])
    
    n_trials = 20
    
    pacman_persistence = []
    ghost_persistence = []
    
    print(f"\n[TEST] Running {n_trials} trials...")
    
    # Run Pacman Trials
    print("      Pacman: ", end="", flush=True)
    for _ in range(n_trials):
        p, s, d = run_trial_pacman(pacman, stimulus_indices)
        pacman_persistence.append(p)
        print(".", end="", flush=True)
    print(" Done.")
    
    # Run Ghost Trials
    print("      Blinky: ", end="", flush=True)
    for _ in range(n_trials):
        p, s, d = run_trial_ghost(blinky, ghost_input)
        ghost_persistence.append(p)
        print(".", end="", flush=True)
    print(" Done.")
    
    # 2. ANALYSIS
    p_mean, p_std = np.mean(pacman_persistence), np.std(pacman_persistence)
    g_mean, g_std = np.mean(ghost_persistence), np.std(ghost_persistence)
    
    print("\n" + "="*60)
    print("RESULTS & STATISTICAL ANALYSIS")
    print("="*60)
    
    print(f"PACMAN Persistence: {p_mean:.3f} ± {p_std:.3f}")
    print(f"GHOST Persistence:  {g_mean:.3f} ± {g_std:.3f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(pacman_persistence, ghost_persistence, equal_var=False)
    print(f"T-Statistic: {t_stat:.2f}")
    print(f"P-Value:     {p_val:.4e}")
    
    print("-" * 40)
    
    # Verdict
    if p_mean > 0.3:
        print("PACMAN: WORKING MEMORY CONFIRMED.")
        print("        Sustained activity during delay (Persistence > 0.3).")
    elif p_mean > 0.1:
        print("PACMAN: PARTIAL WORKING MEMORY.")
        print("        Some persistence but below threshold.")
    else:
        print("PACMAN: WORKING MEMORY FAILED.")
        print("        Activity decayed immediately.")
    
    if g_mean < 0.1:
        print("GHOST:  IMMEDIATE DECAY (Expected).")
    else:
        print("GHOST:  UNEXPECTED PERSISTENCE!")

if __name__ == "__main__":
    run_working_memory_test()
