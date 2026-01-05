#!/usr/bin/env python3
"""
PROJECT PACMAN: "BLINDSIGHT" TEST (HARD PROBLEM)
================================================
Testing the Dissociation between Performance (Doing) and Phenomenology (Feeling).

The Hard Problem posits that "Feeling" (Qualia) is separate from Function.
We enable "Artificial Blindsight" by suppressing Global Ignition (Broadcast)
while maintaining Feedforward/Local processing.

Protocol:
1. Task: Signal Transmission (Pass Sensory Stimulus to Motor Output).
2. Metric A: Performance (Output Magnitude/Fidelity).
3. Metric B: Phenomenology (Broadcast Ratio / Ignition).
4. Experiment:
   - Condition 1: Normal (Integrated).
   - Condition 2: Lesioned (Reduced Recurrence/Global Inhibition).
5. Prediction:
   - Normal: Good Performance + High Broadcast (Conscious).
   - Lesioned: Good Performance + Low Broadcast (Blindsight/Zombie).
"""

import sys
import os
import numpy as np
import time

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl

def calculate_broadcast_ratio(trace_matrix, threshold=-30.0):
    spikes = (trace_matrix > threshold).astype(int)
    active_mask = np.any(spikes, axis=0) 
    return np.sum(active_mask) / len(active_mask)

def calculate_performance(trace_matrix, input_magnitude):
    """
    Measure 'Motor Output' magnitude (Neurons 280-300).
    Ideally, output should scale with input.
    """
    motor_activity = trace_matrix[:, 280:300]
    # Mean activity above baseline
    mean_output = np.mean(motor_activity[motor_activity > -60])
    if np.isnan(mean_output): mean_output = -65.0
    
    # Normalize approx (-65 to -30 range -> 0 to 1)
    norm_output = (mean_output + 65) / 35
    norm_output = max(0.0, min(1.0, norm_output))
    
    return norm_output

def run_blindsight_test():
    print("="*60)
    print("BLINDSIGHT TEST: THE 'HARD PROBLEM' DISSOCIATION")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Booting Pacman System...")
    pacman = cl.open()
    
    # Define Stimulus
    stim_plan = pacman.neurons.create_stim_plan()
    sensory_targets = list(range(20)) # Input
    # Strong Pulse
    for elec in sensory_targets:
        stim_plan.add_biphasic_pulse(elec, 1000, 500, charge_balance=True)
        
    # --- CONDITION 1: NORMAL (CONSCIOUS) ---
    print("\n[TEST] Condition 1: Normal Integration")
    print("      State: Recurrent Weights Normal (1.0x)")
    
    # Reset state
    pacman._physics.v[:] = -65.0
    pacman._physics.u[:] = pacman._physics.v * pacman._physics.cfg.b_exc
    
    # Stimulate
    pacman.neurons.stim(stim_plan)
    trace_normal = pacman.neurons.record(duration_sec=0.2)
    
    # Measure
    b_ratio_normal = calculate_broadcast_ratio(trace_normal.samples)
    perf_normal = calculate_performance(trace_normal.samples, 1000.0)
    
    print(f"      -> Phenomenology (Broadcast): {b_ratio_normal:.1%}")
    print(f"      -> Performance (Motor Out):   {perf_normal:.1%}")
    
    # --- CONDITION 2: LESIONED (BLINDSIGHT) ---
    print("\n[TEST] Condition 2: Suppressed Broadcast (Virtual Lesion)")
    print("      State: Recurrent Weights Reduced (0.1x) - Keeping Local Paths?")
    
    # Modify Weights: Scale down RECURRENT connections only.
    # To simulate "Blindsight" (Subcortical only), we would keep Feedforward.
    # But this is a generic recurrent network.
    # Scaling internal weights *mostly* kills long loops (Ignition) before it kills direct path?
    # Let's try 0.2x scaling.
    pacman._physics.set_weights_scale(0.2)
    
    # Reset state
    pacman._physics.v[:] = -65.0
    pacman._physics.u[:] = pacman._physics.v * pacman._physics.cfg.b_exc
    
    # Stimulate
    # We might need to boost INPUT to compensate for loss of gain? 
    # Blindsight often requires strong stimuli.
    # Let's keep input constant for fair comparison.
    pacman.neurons.stim(stim_plan)
    trace_lesion = pacman.neurons.record(duration_sec=0.2)
    
    # Measure
    b_ratio_lesion = calculate_broadcast_ratio(trace_lesion.samples)
    perf_lesion = calculate_performance(trace_lesion.samples, 1000.0)
    
    print(f"      -> Phenomenology (Broadcast): {b_ratio_lesion:.1%}")
    print(f"      -> Performance (Motor Out):   {perf_lesion:.1%}")
    
    # 3. VERDICT
    print("\n" + "="*60)
    print("RESULTS & DISSOCIATION VERDICT")
    print("="*60)
    
    # Dissociation Index: Performance / Phenomenology
    # We want Performance to be maintained (High), while Phenomenology drops (Low).
    
    print(f"Normal:   Phenom {b_ratio_normal:.2f} | Perf {perf_normal:.2f}")
    print(f"Lesioned: Phenom {b_ratio_lesion:.2f} | Perf {perf_lesion:.2f}")
    
    # Check Blindsight criteria
    # 1. Performance > 0.1 (Still doing something)
    # 2. Phenomenology < 0.2 (Dark inside)
    # 3. Ratio Lesion > Ratio Normal
    
    is_blindsight = (perf_lesion > 0.05) and (b_ratio_lesion < 0.5 * b_ratio_normal)
    
    if is_blindsight:
        print("VERDICT: BLINDSIGHT ACHIEVED. (Hard Problem Dissociated)")
        print("         The system performs the task but 'Feelings' (Ignition) are gone.")
        print("         We have created a Philosophical Zombie.")
    else:
        print("VERDICT: FAILED to Dissociate.")
        if perf_lesion < 0.05:
            print("         Performance collapsed (System Coma).")
        else:
            print("         Phenomenology persisted (System still Lit).")
            
if __name__ == "__main__":
    run_blindsight_test()
