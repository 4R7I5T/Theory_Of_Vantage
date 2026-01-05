#!/usr/bin/env python3
"""
PROJECT PACMAN: "BLINDSIGHT" TEST (HARD PROBLEM) - v2
======================================================
Fixed: Sustained stimulus during recording.

Testing the Dissociation between Performance (Doing) and Phenomenology (Feeling).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl

def calculate_broadcast_ratio(trace_matrix, threshold=-30.0):
    """How many electrodes (out of 59) showed activity above threshold?"""
    spikes = (trace_matrix > threshold).astype(int)
    active_mask = np.any(spikes, axis=1)  # across time for each electrode
    return np.sum(active_mask) / len(active_mask)

def calculate_performance(trace_matrix):
    """Check if Motor Cortex (electrodes 50-58) responded."""
    # Use last electrodes as "motor output"
    motor_electrodes = slice(50, 59)
    motor_activity = trace_matrix[motor_electrodes, :]
    mean_output = np.mean(motor_activity)
    peak_output = np.max(motor_activity)
    
    # Normalize: -65 (rest) -> 0, -30 (spiking) -> 1
    norm_output = (mean_output + 65) / 35
    return max(0.0, min(1.0, norm_output))

def run_with_sustained_stimulus(physics, input_neurons, duration_sec=0.2, amplitude=500.0):
    """
    Run physics with sustained current injection to input neurons.
    Returns voltage traces for all neurons.
    """
    dt_ms = physics.cfg.dt
    n_steps = int(duration_sec * 1000 / dt_ms)
    n_neurons = physics.cfg.n_neurons
    
    traces = np.zeros((n_neurons, n_steps))
    
    for t in range(n_steps):
        # Inject current to input neurons on every step
        external = np.zeros(n_neurons)
        external[input_neurons] = amplitude
        
        physics.step(external)
        traces[:, t] = physics.v.copy()
    
    return traces

def traces_to_electrode(traces, n_electrodes=59, neurons_per_elec=5):
    """Convert neuron traces to electrode-level LFP."""
    electrode_traces = np.zeros((n_electrodes, traces.shape[1]))
    for e in range(n_electrodes):
        start = e * neurons_per_elec
        end = min(start + neurons_per_elec, traces.shape[0])
        if start < traces.shape[0]:
            electrode_traces[e, :] = np.mean(traces[start:end, :], axis=0)
    return electrode_traces

def run_blindsight_test():
    print("="*60)
    print("BLINDSIGHT TEST v2: THE 'HARD PROBLEM' DISSOCIATION")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Booting Pacman System...")
    pacman = cl.open()
    physics = pacman._physics
    
    # Input neurons (Sensory Cortex): 0-29
    input_neurons = list(range(30))
    # Output neurons (Motor Cortex): 270-299
    output_neurons = list(range(270, 300))
    
    # --- CONDITION 1: NORMAL (FULLY CONNECTED) ---
    print("\n[TEST] Condition 1: Normal Integration")
    print("      All weights intact (Full Recurrence)")
    
    # Reset
    physics.v[:] = -65.0
    physics.u[:] = physics.v * physics.cfg.b_exc
    
    # Run with sustained stimulus
    traces_normal = run_with_sustained_stimulus(physics, input_neurons, 
                                                 duration_sec=0.1, amplitude=300.0)
    elec_normal = traces_to_electrode(traces_normal)
    
    b_ratio_normal = calculate_broadcast_ratio(elec_normal)
    perf_normal = calculate_performance(elec_normal)
    
    # Also directly check output neurons
    output_activity = np.mean(traces_normal[output_neurons, :])
    output_peak = np.max(traces_normal[output_neurons, :])
    
    print(f"      -> Broadcast (all electrodes): {b_ratio_normal:.1%}")
    print(f"      -> Motor Cortex Mean V: {output_activity:.1f}mV (peak: {output_peak:.1f}mV)")
    print(f"      -> Performance Score: {perf_normal:.1%}")
    
    # --- CONDITION 2: SELECTIVE LESION (SHORT-RANGE ONLY) ---
    print("\n[TEST] Condition 2: Selective Broadcast Suppression")
    print("      Long-range connections (>100 apart) eliminated")
    print("      Short-range (<100 apart) preserved")
    
    # Create new system for lesion
    pacman_lesion = cl.open()
    physics_lesion = pacman_lesion._physics
    
    # Selective Lesion: Kill long-range, keep short-range
    n = physics_lesion.cfg.n_neurons
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 100:  # Long-range
                physics_lesion.S[i, j] = 0.0  # Complete elimination
    
    # Reset
    physics_lesion.v[:] = -65.0
    physics_lesion.u[:] = physics_lesion.v * physics_lesion.cfg.b_exc
    
    # Run with sustained stimulus
    traces_lesion = run_with_sustained_stimulus(physics_lesion, input_neurons,
                                                 duration_sec=0.1, amplitude=300.0)
    elec_lesion = traces_to_electrode(traces_lesion)
    
    b_ratio_lesion = calculate_broadcast_ratio(elec_lesion)
    perf_lesion = calculate_performance(elec_lesion)
    
    output_activity_lesion = np.mean(traces_lesion[output_neurons, :])
    output_peak_lesion = np.max(traces_lesion[output_neurons, :])
    
    print(f"      -> Broadcast (all electrodes): {b_ratio_lesion:.1%}")
    print(f"      -> Motor Cortex Mean V: {output_activity_lesion:.1f}mV (peak: {output_peak_lesion:.1f}mV)")
    print(f"      -> Performance Score: {perf_lesion:.1%}")
    
    # 3. VERDICT
    print("\n" + "="*60)
    print("RESULTS & DISSOCIATION VERDICT")
    print("="*60)
    
    print(f"Normal:   Broadcast {b_ratio_normal:.0%} | Performance {perf_normal:.0%}")
    print(f"Lesioned: Broadcast {b_ratio_lesion:.0%} | Performance {perf_lesion:.0%}")
    print("-" * 40)
    
    # Blindsight = Performance Maintained, Broadcast Reduced
    broadcast_drop = b_ratio_normal - b_ratio_lesion
    perf_drop = perf_normal - perf_lesion
    
    # Dissociation: Broadcast drops more than Performance
    dissociation_index = broadcast_drop - perf_drop
    
    print(f"Broadcast Drop: {broadcast_drop:.1%}")
    print(f"Performance Drop: {perf_drop:.1%}")
    print(f"Dissociation Index: {dissociation_index:.2f}")
    print("-" * 40)
    
    if dissociation_index > 0.2:
        print("VERDICT: BLINDSIGHT ACHIEVED!")
        print("         Broadcast (Phenomenology) reduced more than Performance.")
        print("         This DISSOCIATES Function from Feeling.")
    elif broadcast_drop > 0.1 and perf_drop < 0.1:
        print("VERDICT: PARTIAL DISSOCIATION")
        print("         Broadcast reduced while Performance maintained.")
    else:
        print("VERDICT: NO DISSOCIATION")
        if broadcast_drop < 0.05:
            print("         Broadcast persisted despite lesion.")
        if perf_drop > broadcast_drop:
            print("         Performance and Broadcast collapsed together (Structuralism).")

if __name__ == "__main__":
    run_blindsight_test()
