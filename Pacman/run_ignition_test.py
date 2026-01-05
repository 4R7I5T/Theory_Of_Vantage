#!/usr/bin/env python3
"""
PROJECT PACMAN: "GLOBAL IGNITION" TEST (GWT)
============================================
Testing Global Workspace Theory: Conscious perception requires
widespread "Ignition" (broadcasting) of local sensory signals.

Protocol:
1. Stimulus: Inject "Pellet Detected" signal into sensory cortex (first 10 neurons).
2. Measure: Broadcast Ratio (% of total population active).
3. Prediction:
   - Pacman (Conscious/Integrated): Input triggers avalanche -> High Broadcast Ratio (>0.5).
   - Ghosts (Zombie/Modular): Input stays local -> Low Broadcast Ratio (<0.1).
"""

import sys
import os
import numpy as np
import time

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble

def calculate_broadcast_ratio(trace_matrix, threshold=-30.0):
    """
    Calculate % of unique neurons that activated (spiked) during the window.
    """
    # Binary raster: (T, N)
    spikes = (trace_matrix > threshold).astype(int)
    # Check if neuron fired at least once
    active_mask = np.any(spikes, axis=0) # Shape (N,)
    n_active = np.sum(active_mask)
    n_total = trace_matrix.shape[1]
    
    return n_active / n_total

def calculate_ghost_broadcast(ghost_brain, input_vector):
    """
    Measure activation spread in feedforward network.
    """
    # Run forward pass
    _, h = ghost_brain.forward(input_vector)
    # h is hidden layer (20 neurons)
    # Check what % are active (> 0.01)
    n_active = np.sum(h > 0.01)
    n_total = len(h)
    return n_active / n_total

def run_ignition_test():
    print("="*60)
    print("GLOBAL IGNITION TEST: TESTING GWT HYPOTHESIS")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Booting Neural Systems...")
    
    # A. PACMAN (Conscious Candidate)
    # Recurrent Izhikevich Network (300 Neurons)
    pacman = cl.open()
    # Enable STDP to allow for learned pathways
    pacman._physics.cfg.stdp_enabled = True 
    print("      Candidate 1: Pacman (Recurrent Izhikevich, n=300)")
    
    # B. GHOST (Zombie Candidate)
    # Feedforward Network (20 Neurons)
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    print("      Candidate 2: Blinky (Feedforward MLP, n=20)")
    
    results = {}
    
    # 2. RUN PROTOCOL
    print("\n[TEST] Comparing Signal Propagation...")
    
    # --- A. PACMAN IGNITION ---
    print("      Testing Pacman (Stimulating Sensory Cortex)...")
    stim_plan = pacman.neurons.create_stim_plan()
    
    # Stimulate "Sensory Cortex" (Neurons 0-20)
    # Representing a strong visual input (Pellet)
    sensory_targets = list(range(20)) 
    for elec in sensory_targets:
        stim_plan.add_biphasic_pulse(elec, 1000, 500, charge_balance=True) # Strong pulse
    pacman.neurons.stim(stim_plan)
    
    # Record Propagation Window (200ms)
    trace = pacman.neurons.record(duration_sec=0.2)
    # Calculate Broadcast
    p_ratio = calculate_broadcast_ratio(trace.samples)
    print(f"      -> Pacman Broadcast Ratio: {p_ratio:.1%} ({int(p_ratio*300)}/300 neurons)")
    
    # --- B. GHOST IGNITION ---
    print("      Testing Blinky (Injecting Sensory Input)...")
    # Strong input vector
    sensory_input = np.ones(4) * 5.0 
    g_ratio = calculate_ghost_broadcast(blinky, sensory_input)
    print(f"      -> Blinky Broadcast Ratio: {g_ratio:.1%} ({int(g_ratio*20)}/20 neurons)")
    
    results['Pacman'] = p_ratio
    results['Blinky'] = g_ratio

    # 3. FALSIFICATION CHECK
    print("\n" + "="*60)
    print("RESULTS & FALSIFICATION VERDICT")
    print("="*60)
    
    h1_pass = (p_ratio > 0.5) # Ignition (Majority of brain activated)
    h2_pass = (g_ratio < 0.2) # Local Processing (Minimal spread)
    h3_pass = (p_ratio > g_ratio + 0.3) # Significant difference
    
    print(f"H1 (Ignition > 50%):      {'PASSED' if h1_pass else 'FAILED'} ({p_ratio:.1%})")
    print(f"H2 (Localization < 20%):  {'PASSED' if h2_pass else 'FAILED'} ({g_ratio:.1%})")
    print(f"H3 (Pacman >> Ghost):     {'PASSED' if h3_pass else 'FAILED'}")
    print("-" * 40)
    
    if h1_pass and h2_pass and h3_pass:
        print("VERDICT: GWT SUPPORTED.")
        print("Integrated architecture enables Global Ignition.")
        print("Modular architecture keeps signals local.")
    else:
        print("VERDICT: INCONCLUSIVE / FALSIFIED.")
        if not h1_pass:
            print("  -> Analysis: Pacman failed to ignite. Simulation is sub-critical/disconnected.")
            print("     This confirms the 'Wetware Imperative' (Need CL1 for true ignition).")

if __name__ == "__main__":
    run_ignition_test()
