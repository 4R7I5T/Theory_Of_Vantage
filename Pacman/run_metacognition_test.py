#!/usr/bin/env python3
"""
METACOGNITION TEST (Confidence-Accuracy Correlation)
=====================================================
Testing whether Pacman "knows what it knows" - if internal confidence
signals correlate with actual accuracy.

Theory: Flavell, Koriat - Metacognitive awareness.

Protocol:
1. Clear Trials: Present unambiguous stimuli.
2. Ambiguous Trials: Present degraded/noisy stimuli.
3. Measure: Activity magnitude (confidence proxy) vs accuracy.

Prediction:
- Pacman: CAC > 0.5 (confidence tracks accuracy)
- Ghost: CAC ≈ 0 (no self-monitoring)
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrainEnsemble

def get_response(system, stimulus, stimulus_indices):
    """Get motor response and confidence for a given stimulus."""
    # Reset
    system._physics.v[:] = -65.0
    system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
    
    # Inject stimulus
    external = np.zeros(system._physics.cfg.n_neurons)
    external[stimulus_indices] = stimulus
    
    # Run for 100 steps
    for _ in range(100):
        system._physics.step(external)
    
    # Motor output (decision)
    motor_indices = list(range(250, 300))
    motor_activity = system._physics.v[motor_indices]
    decision = np.mean(motor_activity > -30)
    
    # Confidence = overall activity magnitude (more active = more confident)
    all_activity = system._physics.v
    confidence = np.mean(all_activity > -40)  # Broader threshold for confidence
    
    return decision, confidence

def run_metacognition_test():
    print("="*60)
    print("METACOGNITION TEST: CONFIDENCE-ACCURACY CORRELATION")
    print("============================================================")
    
    # 1. SETUP
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True
    
    ghosts = GhostBrainEnsemble()
    blinky = ghosts.brains['blinky']
    
    stimulus_indices = list(range(0, 30))
    n_trials_per_condition = 15
    
    # Results storage
    pacman_data = {'clarity': [], 'confidence': [], 'accuracy': []}
    ghost_data = {'clarity': [], 'confidence': [], 'accuracy': []}
    
    print(f"\n[TEST] Running {n_trials_per_condition * 2} trials per subject...")
    
    # CLEAR TRIALS (High stimulus strength)
    print("      Clear Trials: ", end="", flush=True)
    for _ in range(n_trials_per_condition):
        # Pacman
        stim_strength = 500.0  # Strong, clear signal
        decision, confidence = get_response(pacman, stim_strength, stimulus_indices)
        accuracy = 1.0 if decision > 0.3 else 0.0  # Should respond to clear signal
        pacman_data['clarity'].append(1.0)
        pacman_data['confidence'].append(confidence)
        pacman_data['accuracy'].append(accuracy)
        
        # Ghost
        ghost_input = np.array([5.0, 0.0, 0.0, 0.0])
        output, h = blinky.forward(ghost_input)
        ghost_confidence = np.mean(np.abs(output))
        ghost_accuracy = 1.0  # Deterministic
        ghost_data['clarity'].append(1.0)
        ghost_data['confidence'].append(ghost_confidence)
        ghost_data['accuracy'].append(ghost_accuracy)
        
        print(".", end="", flush=True)
    print(" Done.")
    
    # AMBIGUOUS TRIALS (Low stimulus strength + noise)
    print("      Ambiguous Trials: ", end="", flush=True)
    for _ in range(n_trials_per_condition):
        # Pacman
        stim_strength = 50.0 + np.random.normal(0, 30)  # Weak, noisy signal
        stim_strength = max(0, stim_strength)
        decision, confidence = get_response(pacman, stim_strength, stimulus_indices)
        accuracy = 1.0 if (decision > 0.3 and stim_strength > 50) else (0.0 if decision > 0.3 else 0.5)
        pacman_data['clarity'].append(0.0)
        pacman_data['confidence'].append(confidence)
        pacman_data['accuracy'].append(accuracy)
        
        # Ghost
        ghost_input = np.array([0.5 + np.random.normal(0, 0.3), 0.0, 0.0, 0.0])
        ghost_input = np.clip(ghost_input, 0, 5)
        output, h = blinky.forward(ghost_input)
        ghost_confidence = np.mean(np.abs(output))
        ghost_accuracy = 0.5  # Random performance on ambiguous
        ghost_data['clarity'].append(0.0)
        ghost_data['confidence'].append(ghost_confidence)
        ghost_data['accuracy'].append(ghost_accuracy)
        
        print(".", end="", flush=True)
    print(" Done.")
    
    # 2. ANALYSIS
    print("\n" + "="*60)
    print("RESULTS & STATISTICAL ANALYSIS")
    print("="*60)
    
    # Confidence-Accuracy Correlation
    pacman_cac, pacman_p = stats.pearsonr(pacman_data['confidence'], pacman_data['accuracy'])
    ghost_cac, ghost_p = stats.pearsonr(ghost_data['confidence'], ghost_data['accuracy'])
    
    print(f"PACMAN CAC: {pacman_cac:.3f} (p={pacman_p:.4f})")
    print(f"GHOST CAC:  {ghost_cac:.3f} (p={ghost_p:.4f})")
    
    # Confidence difference between clear and ambiguous
    pacman_conf_clear = np.mean([c for c, cl in zip(pacman_data['confidence'], pacman_data['clarity']) if cl == 1.0])
    pacman_conf_ambig = np.mean([c for c, cl in zip(pacman_data['confidence'], pacman_data['clarity']) if cl == 0.0])
    conf_diff = pacman_conf_clear - pacman_conf_ambig
    
    print(f"\nPACMAN Confidence (Clear):     {pacman_conf_clear:.3f}")
    print(f"PACMAN Confidence (Ambiguous): {pacman_conf_ambig:.3f}")
    print(f"Confidence Difference:         {conf_diff:.3f}")
    
    print("-" * 40)
    
    # Verdict
    if pacman_cac > 0.5 and pacman_p < 0.05:
        print("PACMAN: METACOGNITION CONFIRMED.")
        print("        Confidence correlates with accuracy (CAC > 0.5).")
    elif pacman_cac > 0.2:
        print("PACMAN: PARTIAL METACOGNITION.")
        print("        Some correlation but below threshold.")
    else:
        print("PACMAN: METACOGNITION FAILED.")
        print("        No confidence-accuracy relationship.")
    
    if abs(ghost_cac) < 0.2:
        print("GHOST:  NO SELF-MONITORING (Expected).")
    else:
        print("GHOST:  UNEXPECTED METACOGNITION!")

if __name__ == "__main__":
    run_metacognition_test()
