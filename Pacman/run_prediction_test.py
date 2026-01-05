#!/usr/bin/env python3
"""
PROJECT PACMAN: "SURPRISE" TEST (PREDICTIVE PROCESSING)
=======================================================
Testing Free Energy Principle (Friston).
Consciousness minimizes Prediction Error (Surprise).

Protocol (Oddball Paradigm):
1. Train: Present standard stimulus "A" repeatedly (Establish Prior).
2. Test: Present deviant stimulus "B" (Violate Prior).
3. Measure: Mismatch Negativity (MMN) = Response(B|A) - Response(B|Naive).
   (Does the context of A make B surprising?)

Prediction:
   - Pacman (STDP): Adapts to A. B causes higher error/activity. High MMN.
   - Ghost (Fixed): No adaptation. B Is processed same as Naive. Zero MMN.
"""

import sys
import os
import numpy as np
import time

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble

def get_mean_activation(trace_matrix):
    # Mean firing rate or voltage
    return np.mean(trace_matrix)

def get_ghost_activation(ghost, inputs):
    # Mean hidden activity
    acts = []
    for x in inputs:
        _, h = ghost.forward(x)
        acts.append(np.mean(h))
    return np.mean(acts)

def run_prediction_test():
    print("="*60)
    print("PREDICTION TEST: TESTING FREE ENERGY PRINCIPLE")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Booting and Training...")
    
    # A. PACMAN
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True # Essential for learning priors
    pacman._physics.cfg.learning_rate = 0.2 # Boost for fast learning
    
    # B. GHOST
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    
    # Stimuli
    stim_A = np.zeros(300)
    stim_A[0:10] = 1000.0 # Pattern A: Input to 0-10
    
    stim_B = np.zeros(300)
    stim_B[20:30] = 1000.0 # Pattern B: Input to 20-30
    
    input_A_ghost = np.array([5.0, 0.0, 0.0, 0.0])
    input_B_ghost = np.array([0.0, 5.0, 0.0, 0.0])
    
    # 2. RUN PROTOCOL
    
    # --- STEP 1: MEASURE NAIVE RESPONSE TO B (Control) ---
    print("\n[PHASE 1] Measuring Naive Response to 'B'...")
    
    # Pacman Naive B
    stim_plan = pacman.neurons.create_stim_plan()
    for i in range(20, 30): 
        stim_plan.add_biphasic_pulse(i, 1000, 500, charge_balance=True)
    pacman.neurons.stim(stim_plan)
    trace_naive = pacman.neurons.record(duration_sec=0.1)
    act_pacman_naive = get_mean_activation(trace_naive.samples)
    
    # Ghost Naive B
    act_ghost_naive = get_ghost_activation(blinky, [input_B_ghost]*10)
    
    # --- STEP 2: TRAIN ON A (Establish Prior) ---
    print("[PHASE 2] Training on 'A' (Repetitive Stimulation)...")
    
    # Pacman Train A (Repeated exposure)
    # We run 20 cycles of A
    for _ in range(20):
        stim_plan = pacman.neurons.create_stim_plan()
        for i in range(10): # Pattern A
            stim_plan.add_biphasic_pulse(i, 1000, 500, charge_balance=True)
        pacman.neurons.stim(stim_plan)
        pacman.neurons.record(duration_sec=0.05) # Process
        
    # Ghost "Train" A (Output doesn't change weights)
    get_ghost_activation(blinky, [input_A_ghost]*100)
    
    # --- STEP 3: MEASURE DEVIANT RESPONSE TO B (Surprise) ---
    print("[PHASE 3] Measuring Deviant Response to 'B' (Violation)...")
    
    # Pacman Deviant B
    stim_plan = pacman.neurons.create_stim_plan()
    for i in range(20, 30): 
        stim_plan.add_biphasic_pulse(i, 1000, 500, charge_balance=True)
    pacman.neurons.stim(stim_plan)
    trace_deviant = pacman.neurons.record(duration_sec=0.1)
    act_pacman_deviant = get_mean_activation(trace_deviant.samples)
    
    # Ghost Deviant B
    act_ghost_deviant = get_ghost_activation(blinky, [input_B_ghost]*10)
    
    # 3. CALCULATE MMN (Mismatch Negativity)
    mmn_pacman = act_pacman_deviant - act_pacman_naive
    mmn_ghost = act_ghost_deviant - act_ghost_naive
    
    print("\n" + "="*60)
    print("RESULTS & PREDICTION VERDICT")
    print("="*60)
    
    print(f"PACMAN MMN: {mmn_pacman:.4f} (Deviant {act_pacman_deviant:.2f} vs Naive {act_pacman_naive:.2f})")
    print(f"GHOST MMN:  {mmn_ghost:.4f}")
    
    print("-" * 40)
    if abs(mmn_pacman) > 1.0: # Significant difference
        print("PACMAN: SURPRISED. (Prediction Error detected).")
        print("        Verdict: SUPPORTING Predictive Processing.")
    else:
        print("PACMAN: INDIFFERENT. (No MMN).")
        print("        Verdict: FAILED PP Test.")
        
    if abs(mmn_ghost) < 0.001:
        print("GHOST:  ROBOTIC. (Identical response).")
    else:
        print("GHOST:  ...Confused? (Weights shouldn't change).")

if __name__ == "__main__":
    run_prediction_test()
