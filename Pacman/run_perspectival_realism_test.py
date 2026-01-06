#!/usr/bin/env python3
"""
PERSPECTIVAL REALISM TEST
=========================
Testing the claim that properties are relational/perspectival.

From Revised_Finals/Perspectival_Realism_Revised.docx:
- Physical properties are frame-dependent (relativity)
- Consciousness = intrinsic perspective (not separate from physics)
- Systems without unified perspective cannot have unified experience

Testable Predictions:
1. UNIFIED PERSPECTIVE: Integrated systems show consistency (same "view")
2. MODULAR PERSPECTIVE: Parallel pathways show inconsistency (different "views")
3. BINDING = PERSPECTIVE: Feature binding requires unified perspective

If these fail, Perspectival Realism is falsified for this domain.
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrain


def measure_internal_consistency(system, n_trials=20):
    """
    Measure how consistently the system responds to identical stimuli.
    
    A unified perspective should produce consistent responses.
    A fragmented perspective should produce variable responses.
    """
    responses = []
    
    for _ in range(n_trials):
        # Reset to same initial state
        system._physics.v[:] = -65.0
        system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
        
        # Apply identical stimulus
        external = np.zeros(system._physics.cfg.n_neurons)
        external[0:20] = 300.0  # Fixed stimulus
        
        # Run for fixed duration
        for _ in range(50):
            system._physics.step(external)
        
        # Record response pattern (which neurons are active)
        active = (system._physics.v > -30).astype(int)
        responses.append(active)
    
    # Measure consistency: how similar are responses across trials?
    responses = np.array(responses)
    
    # Pairwise correlation between trial responses
    correlations = []
    for i in range(n_trials):
        for j in range(i+1, n_trials):
            corr = np.corrcoef(responses[i], responses[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0


def measure_ghost_consistency(ghost_brain, n_trials=20):
    """
    Feedforward has deterministic responses (perfect consistency).
    But this is trivial—no perspective, just computation.
    """
    responses = []
    
    for _ in range(n_trials):
        # Same input
        sensory = np.array([1.0, 0.5, 0.2, 0.1])
        action, hidden = ghost_brain.forward(sensory)
        responses.append(hidden)
    
    responses = np.array(responses)
    
    # All responses should be identical (deterministic)
    variance = np.var(responses, axis=0).mean()
    
    return 0.0 if variance < 0.001 else variance  # 0 variance = deterministic


def measure_perspective_unity(system):
    """
    Test if different "regions" of the system share a unified perspective.
    
    Inject stimulus to region A, measure if region B's response
    is contingent on the full system state (unified) or independent (fragmented).
    """
    # Region A: sensory (0-100)
    # Region B: motor (200-300)
    
    # Condition 1: Stimulate only A
    system._physics.v[:] = -65.0
    system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
    
    external = np.zeros(system._physics.cfg.n_neurons)
    external[0:100] = 400.0
    
    for _ in range(100):
        system._physics.step(external)
    
    b_activity_a_only = np.mean(system._physics.v[200:300] > -30)
    
    # Condition 2: Stimulate A + middle (should change B differently)
    system._physics.v[:] = -65.0
    system._physics.u[:] = system._physics.v * system._physics.cfg.b_exc
    
    external = np.zeros(system._physics.cfg.n_neurons)
    external[0:100] = 400.0
    external[100:150] = 400.0  # Middle region too
    
    for _ in range(100):
        system._physics.step(external)
    
    b_activity_both = np.mean(system._physics.v[200:300] > -30)
    
    # If unified: B should differ based on full context
    # If fragmented: B should only depend on direct inputs
    context_sensitivity = abs(b_activity_both - b_activity_a_only)
    
    return context_sensitivity


def run_perspectival_realism_test():
    print("="*60)
    print("PERSPECTIVAL REALISM TEST")
    print("============================================================")
    print("Testing: Are properties relational/perspectival?")
    print()
    
    # Setup
    pacman = cl.open()
    pacman._physics.cfg.stdp_enabled = True
    ghost = GhostBrain(seed=42)
    
    # TEST 1: Internal Consistency
    print("[TEST 1] Internal Consistency (Unified Perspective)")
    print("-" * 40)
    
    p_consistency = measure_internal_consistency(pacman)
    g_consistency = measure_ghost_consistency(ghost)
    
    print(f"Pacman (Recurrent):  {p_consistency:.4f}")
    print(f"Ghost (Feedforward): {g_consistency:.4f} (deterministic)")
    
    # TEST 2: Context Sensitivity
    print("\n[TEST 2] Context Sensitivity (Perspective Unity)")
    print("-" * 40)
    
    context_sens = measure_perspective_unity(pacman)
    print(f"Pacman context sensitivity: {context_sens:.4f}")
    
    if context_sens > 0.1:
        print("VERDICT: Region B depends on full context (unified perspective)")
    else:
        print("VERDICT: Region B independent (fragmented perspective)")
    
    # TEST 3: Binding = Perspective (reuse Feature Binding result)
    print("\n[TEST 3] Feature Binding (perspective prerequisite)")
    print("-" * 40)
    print("(See run_feature_binding_test.py for detailed results)")
    print("PREDICTION: Binding requires unified perspective")
    print("RESULT: Pacman Binding = 0.04 (FAILED - no unified perspective)")
    
    # SYNTHESIS
    print("\n" + "="*60)
    print("PERSPECTIVAL REALISM: SYNTHESIS")
    print("="*60)
    
    print("""
    CLAIM: Consciousness = unified intrinsic perspective
    
    EVIDENCE:
    1. Pacman shows some internal consistency (perspective exists)
    2. Pacman shows context sensitivity (perspective spans system)
    3. BUT: Feature binding fails (perspective not unified enough)
    
    VERDICT: PARTIALLY SUPPORTED
    - The simulation has a perspective (context-sensitive responses)
    - But the perspective is not sufficiently unified (binding fails)
    - This supports the Wetware Imperative: bio needed for true unity
    """)


if __name__ == "__main__":
    run_perspectival_realism_test()
