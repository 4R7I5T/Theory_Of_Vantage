#!/usr/bin/env python3
"""
PROJECT PACMAN: "ZAP & ZIP" CONSCIOUSNESS TEST
==============================================
Falsifying the Zombie Hypothesis using Perturbational Complexity Index (PCI).

Protocol:
1. Baseline: Record spontaneous firing (LZC_spontaneous)
2. Zap: Deliver strong pulse to random subset of neurons
3. Zip: Record evoked firing (LZC_evoked)
4. PCI = LZC_evoked - LZC_spontaneous

Prediction:
- Pacman (Integrated): High PCI (Rich, sustained response)
- Ghosts (Feedforward): Low PCI (Simple, transient response)
"""

import sys
import os
import numpy as np
import time

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PA')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble
from pa_toolkit.complexity import lempel_ziv_complexity

def binarize_spikes(trace_matrix, threshold=-30.0):
    """Convert voltage traces to binary spike raster."""
    # Simple threshold crossing
    # 1 if v > threshold (spike), 0 otherwise mismatch?
    # Actually Izhikevich spike is onset. 
    # Let's say v > -20 is a spike peak.
    return (trace_matrix > threshold).astype(int)

def binarize_activations(activation_matrix, threshold=0.1):
    """Convert continuous activations to binary raster."""
    return (activation_matrix > threshold).astype(int)

def downsample_raster(raster, bin_size=10):
    """
    Downsample binary raster by OR-ing within bins.
    Reduces data size for LZC computation.
    """
    T, N = raster.shape
    n_bins = T // bin_size
    if n_bins < 1: return raster
    
    binned = np.zeros((n_bins, N), dtype=int)
    for i in range(n_bins):
        chunk = raster[i*bin_size : (i+1)*bin_size]
        binned[i] = np.any(chunk, axis=0).astype(int)
    return binned

def run_zap_and_zip():
    print("="*60)
    print("ZAP & ZIP PROTOCOL: TESTING FOR CONSCIOUSNESS")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Booting Neural Systems...")
    
    # A. PACMAN (Conscious Candidate)
    # Recurrent Izhikevich Network (300 Neurons)
    pacman = cl.open()
    # Enable strong STDP to ensure integration
    pacman._physics.cfg.stdp_enabled = True 
    print("      Candidate 1: Pacman (Recurrent Izhikevich, n=300)")
    
    # B. GHOST (Zombie Candidate)
    # Feedforward Network (20 Neurons)
    ghosts = GhostBrainEnsemble()
    # We'll just test one ghost brain (Blinky)
    blinky = ghosts.brains['blinky']
    print("      Candidate 2: Blinky (Feedforward MLP, n=20)")
    
    results = {}
    
    # 2. RUN PROTOCOL
    for candidate_name in ["Pacman", "Blinky"]:
        print(f"\n[TEST] Testing {candidate_name}...")
        
        # --- PHASE 1: BASELINE (SPONTANEOUS) ---
        print("      Phase 1: Recording Spontaneous Activity (500ms)...")
        
        if candidate_name == "Pacman":
            # Record 500ms of "thinking"
            # We need to stimulate slightly to keep it alive (thalamic input)
            stim_plan = pacman.neurons.create_stim_plan()
            # Random noise input
            for t in range(0, 100, 10): # Every 10ms
                 elec = np.random.randint(0, 300)
                 stim_plan.add_biphasic_pulse(elec, 500, 200, charge_balance=True)
            pacman.neurons.stim(stim_plan)
            
            trace = pacman.neurons.record(duration_sec=0.1)
            # Threshold voltage to get binary spikes
            # Izhikevich spikes peak at +30mV. Threshold at 0.
            raster_baseline = (trace.samples > 0).astype(int) 
            
        else:
            # Ghost: Run forward passes with random input
            raster_baseline = []
            for _ in range(20): # 100ms equivalent (5ms steps)
                # Random sensory input
                sensory = np.random.rand(4) 
                _, h = blinky.forward(sensory) 
                raster_baseline.append(h)
            raster_baseline = binarize_activations(np.array(raster_baseline))
            
        # Optimization: Downsample to avoid O(N^2) string issues
        raster_baseline_binned = downsample_raster(raster_baseline, bin_size=5)
        lzc_baseline = lempel_ziv_complexity(raster_baseline_binned)
        print(f"      -> Baseline LZC: {lzc_baseline:.4f}")
        
        
        # --- PHASE 2: ZAP (PERTURBATION) ---
        print("      Phase 2: ZAPPING System (Perturbation)...")
        
        if candidate_name == "Pacman":
            # Strong Pulse to 10 random neurons
            stim_plan = pacman.neurons.create_stim_plan()
            targets = np.random.choice(300, 10, replace=False)
            for elec in targets:
                stim_plan.add_biphasic_pulse(int(elec), 2000, 500, charge_balance=True) # Strong pulse
            pacman.neurons.stim(stim_plan)
            
            # Record Response (Evoked)
            trace = pacman.neurons.record(duration_sec=0.1)
            raster_evoked = (trace.samples > 0).astype(int)
            
        else:
            # Ghost: "Zap" = Massive input activation
            raster_evoked = []
            # First step: ZAP inputs (all 1.0)
            _, h = blinky.forward(np.ones(4)*5.0) 
            raster_evoked.append(h)
            # Subsequent steps: Decay/Silence
            for _ in range(19):
                _, h = blinky.forward(np.zeros(4))
                raster_evoked.append(h)
            raster_evoked = binarize_activations(np.array(raster_evoked))
            
        # --- PHASE 3: ZIP (COMPLEXITY) ---
        raster_evoked_binned = downsample_raster(raster_evoked, bin_size=5)
        lzc_evoked = lempel_ziv_complexity(raster_evoked_binned)
        print(f"      -> Evoked LZC:   {lzc_evoked:.4f}")
        
        # --- PHASE 4: PCI CALCULATION ---
        # PCI proxy = Change in complexity relative to baseline
        # Actually simplified PCI often just looks at LZC_evoked normalized
        # But let's look at the Delta
        pci = lzc_evoked - lzc_baseline
        results[candidate_name] = {
            'baseline': lzc_baseline,
            'evoked': lzc_evoked,
            'pci': pci,
            'raster_evoked': raster_evoked
        }
        print(f"      => PCI Score:    {pci:.4f}")

    # 3. FALSIFICATION CHECK
    print("\n" + "="*60)
    print("RESULTS & FALSIFICATION VERDICT")
    print("="*60)
    
    p_pci = results['Pacman']['pci']
    g_pci = results['Blinky']['pci']
    
    print(f"PACMAN PCI: {p_pci:.4f}")
    print(f"GHOST PCI:  {g_pci:.4f}")
    
    h1_pass = results['Pacman']['evoked'] > 0.3 # High complexity response
    h2_pass = results['Blinky']['evoked'] < 0.1 # Low complexity response
    h3_pass = (p_pci > g_pci + 0.1) # Significant difference
    
    print("-" * 40)
    print(f"H1 (Pacman Complexity > 0.3): {'PASSED' if h1_pass else 'FAILED'}")
    print(f"H2 (Ghost Complexity < 0.1):  {'PASSED' if h2_pass else 'FAILED'}")
    print(f"H3 (Pacman > Ghost):          {'PASSED' if h3_pass else 'FAILED'}")
    print("-" * 40)
    
    if h1_pass and h2_pass and h3_pass:
        print("VERDICT: ZOMBIE HYPOTHESIS REFUTED.")
        print("Consciousness requires INTEGRATION (PCI).")
        print("Pacman is Conscious. Ghosts are Zombies.")
    else:
        print("VERDICT: INCONCLUSIVE / FALSIFIED.")
        
    # Visuals
    # Save rasters to npy for plotting if needed
    np.save('zap_zip_results.npy', results)

if __name__ == "__main__":
    run_zap_and_zip()
