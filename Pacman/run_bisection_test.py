#!/usr/bin/env python3
"""
PROJECT PACMAN: "BISECTION" TEST (IIT)
======================================
Testing Integrated Information Theory (Irreducibility).
A conscious system is Irreducible ($\Phi > 0$).
Bisecting it should cause qualitative collapse, not just quantitative reduction.

Protocol:
1. "Virtual Lobotomy": Sever connections between Left (0-149) and Right (150-299) hemispheres.
2. Stimulus: Inject signal into Left Sensory Cortex (0-20).
3. Measure: Broadcast Ratio (Global Ignition).
4. Prediction:
   - Pacman (Recurrent): Catastrophic Collapse (<10% or just local).
     (If criticality depends on global loop, even local ignition fails).
   - Ghost (Feedforward): Linear Reduction (~50%).
     (Left side functions perfectly well independently).
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
    spikes = (trace_matrix > threshold).astype(int)
    active_mask = np.any(spikes, axis=0) 
    return np.sum(active_mask) / len(active_mask)

def calculate_ghost_broadcast(ghost_brain, input_vector):
    _, h = ghost_brain.forward(input_vector)
    return np.sum(h > 0.01) / len(h)

def run_bisection_test():
    print("="*60)
    print("BISECTION TEST: TESTING ITT IRREDUCIBILITY")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Booting and Bisecting...")
    
    # A. PACMAN (Conscious)
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True
    
    # --- VIRTUAL LOBOTOMY ---
    print("      [SURGERY] Performing Corpus Callosotomy on Pacman...")
    W = pacman._physics.S
    # Cut Left->Right (0-150 -> 150-300)
    W[150:300, 0:150] = 0.0
    # Cut Right->Left (150-300 -> 0-150)
    W[0:150, 150:300] = 0.0
    print("      [SURGERY] Pacman hemispheres disconnected.")
    
    # B. GHOST (Zombie)
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    
    # --- VIRTUAL LOBOTOMY ---
    print("      [SURGERY] Performing Corpus Callosotomy on Blinky...")
    # Setup: 4 inputs, 20 hidden.
    # Left: In 0-1, Hidden 0-9. Right: In 2-3, Hidden 10-19.
    # Cut Cross-over weights
    # W1: (Input, Hidden)
    blinky.W1[0:2, 10:20] = 0.0 # Left In -> Right Hidden
    blinky.W1[2:4, 0:10] = 0.0  # Right In -> Left Hidden
    
    print("      [SURGERY] Blinky hemispheres disconnected.")
    
    # 2. RUN IGNITION TEST (Left Side Only)
    print("\n[TEST] Stimulating LEFT Hemisphere Only...")
    
    # --- A. PACMAN ---
    stim_plan = pacman.neurons.create_stim_plan()
    sensory_targets = list(range(20)) # Within Left (0-149)
    for elec in sensory_targets:
        stim_plan.add_biphasic_pulse(elec, 1000, 500, charge_balance=True)
    pacman.neurons.stim(stim_plan)
    
    trace = pacman.neurons.record(duration_sec=0.2)
    p_ratio = calculate_broadcast_ratio(trace.samples)
    print(f"      -> Pacman Broadcast Ratio: {p_ratio:.1%} ({int(p_ratio*300)}/300)")
    
    # --- B. GHOST ---
    # Left Input Only (5, 5, 0, 0)
    sensory_input = np.array([5.0, 5.0, 0.0, 0.0])
    g_ratio = calculate_ghost_broadcast(blinky, sensory_input)
    print(f"      -> Blinky Broadcast Ratio: {g_ratio:.1%} ({int(g_ratio*20)}/20)")
    
    # 3. VERDICT
    print("\n" + "="*60)
    print("RESULTS & IIT VERDICT")
    print("="*60)
    
    # Intact Pacman was 100%. Intact Ghost (Full Input) was 45%.
    # Here Ghost has Half Input.
    
    print(f"PACMAN (Post-Op): {p_ratio:.1%}")
    print(f"GHOST (Post-Op):  {g_ratio:.1%}")
    
    # Hypothesis:
    # Pacman should collapse from 100% to ~50% (Linear) or <50% (Catastrophic).
    # Ghost should be ~25% (Half of 50%).
    
    print("-" * 40)
    print("Interpretation:")
    
    if p_ratio >= 0.49:
        print("PACMAN: REDUCIBLE. (Linear reduction to ~50%)")
        print("        The left hemisphere functions independently.")
    else:
        print("PACMAN: IRREDUCIBLE. (Catastrophic collapse)")
        
    if g_ratio >= 0.2:
        print("GHOST:  REDUCIBLE. (Linear reduction)")
    else:
        print("GHOST:  IRREDUCIBLE.")
        
    print("-" * 40)
    print("Note: Ideally, a Conscious system fails 'Qualitatively' (Collapse),")
    print("while a Zombie fails 'Quantitatively' (Proportional).")

if __name__ == "__main__":
    run_bisection_test()
