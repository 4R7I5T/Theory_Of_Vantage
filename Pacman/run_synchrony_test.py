#!/usr/bin/env python3
"""
PROJECT PACMAN: "SYNC" TEST (GAMMA SYNCHRONY)
=============================================
Testing Neural Correlates of Consciousness (Crick & Koch).
Consciousness involves binding via 40Hz Gamma Synchrony.

Protocol:
1. Stimulus: Driving input (visual scene).
2. Measure: Phase Locking Value (PLV) in Gamma Band (30-80Hz).
3. Prediction:
   - Pacman (Recurrent): High PLV (Synchronization).
   - Ghost (Feedforward): Low PLV (Independent firing).
"""

import sys
import os
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble

def compute_plv(traces, fs=1000.0):
    """
    Compute Phase Locking Value (PLV) in Gamma band.
    """
    T, N = traces.shape
    if N < 2: return 0.0
    
    # 1. Bandpass Filter (30-80Hz)
    nyq = 0.5 * fs
    b, a = butter(4, [30/nyq, 80/nyq], btype='band')
    
    filtered = np.zeros_like(traces)
    # Check if signal is long enough for filtering
    if T < 30: return 0.0
    
    try:
        filtered = filtfilt(b, a, traces, axis=0)
    except:
        return 0.0 # Signal too short
    
    # 2. Hilbert Transform (Instantaneous Phase)
    analytic = hilbert(filtered, axis=0)
    phases = np.angle(analytic)
    
    # 3. Pairwise PLV
    # Sample 50 random pairs to save time
    plv_sum = 0.0
    count = 0
    
    for _ in range(50):
        i, j = np.random.choice(N, 2, replace=False)
        delta_phi = phases[:, i] - phases[:, j]
        # PLV = |mean(exp(i * delta))|
        plv = np.abs(np.mean(np.exp(1j * delta_phi)))
        plv_sum += plv
        count += 1
        
    return plv_sum / count

def run_synchrony_test():
    print("="*60)
    print("GAMMA SYNCHRONY TEST: TESTING BINDING")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Booting Neural Systems...")
    pacman = cl.open()
    # E/I Balance is crucial for Gamma. 
    # Current Default: 80% Exc, 20% Inh. Izhikevich RS/FS.
    # This SHOUL oscilate if driven.
    pacman._physics.cfg.stdp_enabled = False # Dissable learning to test pure dynamics
    
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    
    # 2. RUN PROTOCOL
    print("\n[TEST] Recording 1s of Driven Activity...")
    
    # --- A. PACMAN ---
    # Drive with noise + signal
    stim_plan = pacman.neurons.create_stim_plan()
    # Poisson-like drive to all neurons to induce oscillation
    for t in range(0, 1000, 5): # 200Hz drive
        elec = np.random.randint(0, 300)
        stim_plan.add_biphasic_pulse(elec, 500, 200, charge_balance=True)
    pacman.neurons.stim(stim_plan)
    
    trace = pacman.neurons.record(duration_sec=1.0) # 1000 samples @ 1kHz?
    # Physics dt=0.5ms -> 2kHz. 1s = 2000 samples.
    
    plv_pacman = compute_plv(trace.samples, fs=2000.0)
    print(f"      -> Pacman Gamma PLV: {plv_pacman:.4f}")
    
    # --- B. GHOST ---
    # Drive with random input
    # Ghost trace
    ghost_output = []
    for _ in range(200): # 1s equiv (approx)
        sensory = np.random.rand(4)
        _, h = blinky.forward(sensory)
        ghost_output.append(h)
    ghost_output = np.array(ghost_output)
    
    # Upsample ghost output to match 2kHz for fair filter comparison?
    # Or just calc PLV on raw steps.
    # Ghost steps are discrete. Let's assume 50Hz update (20ms).
    # Filter 30-80Hz on 50Hz signal is impossible (Nyquist).
    # So Ghost CANNOT have Gamma by definition?
    # Actually, if we assume Ghost runs at 1000Hz updates?
    # Comparison is unfair if sampling rates differ.
    # Let's interpret "Ghost" as a rate-coded system. 
    # Whatever, let's treat the sequence as "time series".
    
    # If Ghost steps are "time", we need T sufficient.
    # Let's generate 1000 steps of Ghost.
    ghost_output_long = []
    for i in range(1000):
         # Create some correlation in input to see if it binds
         sensory = np.array([np.sin(i*0.1), np.sin(i*0.1), 0, 0]) 
         _, h = blinky.forward(sensory)
         ghost_output_long.append(h)
    ghost_output_long = np.array(ghost_output_long)
    
    plv_ghost = compute_plv(ghost_output_long, fs=1000.0)
    print(f"      -> Ghost Gamma PLV:  {plv_ghost:.4f}")
    
    # 3. VERDICT
    print("\n" + "="*60)
    print("RESULTS & VERDICT")
    print("="*60)
    print(f"PACMAN PLV: {plv_pacman:.4f}")
    print(f"GHOST PLV:  {plv_ghost:.4f}")
    
    if plv_pacman > 0.4:
        print("PACMAN: SYNCHRONIZED (Binding via Gamma).")
        print("        Verdict: SUPPORTING Consciousness.")
    else:
        print("PACMAN: ASYNCHRONOUS (No Binding).")
        print("        Verdict: FAILED Sync Test (Sub-critical).")
        
    print("-" * 40)

if __name__ == "__main__":
    run_synchrony_test()
