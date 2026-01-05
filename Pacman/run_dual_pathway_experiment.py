#!/usr/bin/env python3
"""
DUAL-PATHWAY ARCHITECTURE EXPERIMENT
=====================================
Proving that Blindsight requires architectural modularity, not just parameter tuning.

We build TWO brain architectures:
1. UNIFIED: Single recurrent Izhikevich network (current Pacman brain)
2. DUAL-PATHWAY: Subcortical Reflex + Cortical Conscious pathways

We then lesion the "conscious" pathway and compare:
- UNIFIED: Should show correlated collapse (both behavior AND ignition fail)
- DUAL-PATHWAY: Should show DISSOCIATION (behavior persists, ignition fails = BLINDSIGHT)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.physics import NeuralPhysics, PhysicsConfig

# =============================================================================
# ARCHITECTURE 1: UNIFIED BRAIN (Current)
# =============================================================================

class UnifiedBrain:
    """Single recurrent network. Lesion affects everything."""
    
    def __init__(self):
        self.physics = NeuralPhysics()
        self.n_neurons = self.physics.cfg.n_neurons  # 300
        
        # Regions
        self.sensory = list(range(0, 50))      # Input
        self.motor = list(range(250, 300))     # Output
        # Everything in between is "integration"
        
    def process(self, sensory_input, record=False):
        """
        Process sensory input and return motor output.
        """
        # Reset for clean trial
        self.physics.v[:] = -65.0
        self.physics.u[:] = self.physics.v * self.physics.cfg.b_exc
        
        # Inject sensory current
        external = np.zeros(self.n_neurons)
        external[self.sensory] = sensory_input * 100  # Scale input
        
        # Run for 50 steps
        traces = []
        for _ in range(50):
            self.physics.step(external)
            if record:
                traces.append(self.physics.v.copy())
        
        # Read motor output
        motor_output = np.mean(self.physics.v[self.motor])
        motor_active = np.sum(self.physics.v[self.motor] > -30) / len(self.motor)
        
        # Compute broadcast (for phenomenology)
        broadcast = np.sum(self.physics.v > -30) / self.n_neurons
        
        return motor_active, broadcast, traces
    
    def lesion_global(self, severity=0.9):
        """Lesion: Reduce ALL long-range weights."""
        n = self.n_neurons
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 50:
                    self.physics.S[i, j] *= (1 - severity)


# =============================================================================
# ARCHITECTURE 2: DUAL-PATHWAY BRAIN (New)
# =============================================================================

class DualPathwayBrain:
    """
    Two parallel pathways:
    1. SUBCORTICAL (Reflex): Direct sensory -> motor, feedforward, fast
    2. CORTICAL (Conscious): Sensory -> Global Workspace -> Motor, recurrent, slow
    
    The key insight: These are SEPARATE weight matrices.
    Lesioning cortical does NOT affect subcortical.
    """
    
    def __init__(self):
        # Subcortical path: Simple feedforward matrix
        self.n_sensory = 50
        self.n_motor = 50
        self.W_subcortical = np.random.randn(self.n_motor, self.n_sensory) * 0.5
        
        # Cortical path: Full recurrent network
        self.cortical = NeuralPhysics()
        self.n_cortical = self.cortical.cfg.n_neurons  # 300
        
        # Cortical regions
        self.cortical_sensory = list(range(0, 50))
        self.cortical_motor = list(range(250, 300))
        
        # Mix ratio: How much cortical influences final motor output
        self.cortical_gain = 0.5
        self.subcortical_gain = 0.5
        
    def process(self, sensory_input, record=False):
        """
        Process through BOTH pathways, combine outputs.
        """
        # === SUBCORTICAL PATH (Fast Reflex) ===
        subcortical_output = np.tanh(self.W_subcortical @ (sensory_input * np.ones(self.n_sensory)))
        subcortical_motor = np.mean(subcortical_output > 0)  # Fraction active
        
        # === CORTICAL PATH (Conscious Integration) ===
        # Reset cortical state
        self.cortical.v[:] = -65.0
        self.cortical.u[:] = self.cortical.v * self.cortical.cfg.b_exc
        
        # Inject to cortical sensory
        external = np.zeros(self.n_cortical)
        external[self.cortical_sensory] = sensory_input * 100
        
        # Run cortical dynamics
        traces = []
        for _ in range(50):
            self.cortical.step(external)
            if record:
                traces.append(self.cortical.v.copy())
        
        # Cortical motor output
        cortical_motor = np.sum(self.cortical.v[self.cortical_motor] > -30) / len(self.cortical_motor)
        
        # Cortical broadcast (phenomenology)
        broadcast = np.sum(self.cortical.v > -30) / self.n_cortical
        
        # === COMBINE ===
        # Final motor = weighted sum of both pathways
        final_motor = (self.subcortical_gain * subcortical_motor + 
                       self.cortical_gain * cortical_motor)
        
        return final_motor, broadcast, traces
    
    def lesion_cortical(self, severity=0.9):
        """
        Lesion ONLY the cortical pathway (consciousness).
        Subcortical reflex remains INTACT.
        """
        n = self.n_cortical
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 50:
                    self.cortical.S[i, j] *= (1 - severity)
        
        # Also reduce cortical gain
        self.cortical_gain *= (1 - severity)


# =============================================================================
# PACMAN GAME SIMULATION
# =============================================================================

def simulate_pacman_trial(brain, lesioned=False):
    """
    Simulate a Pacman trial.
    Sensory input = distance to ghost (higher = closer = danger)
    Motor output = movement (higher = escape behavior)
    """
    
    # Simulate ghost approaching
    ghost_distances = np.linspace(0.1, 1.0, 10)  # Far to close
    
    motor_responses = []
    broadcasts = []
    
    for dist in ghost_distances:
        motor, broadcast, _ = brain.process(dist)
        motor_responses.append(motor)
        broadcasts.append(broadcast)
    
    return {
        'motor_mean': np.mean(motor_responses),
        'motor_peak': np.max(motor_responses),
        'broadcast_mean': np.mean(broadcasts),
        'behavior_score': np.mean(motor_responses),  # Did it respond to threat?
        'consciousness_score': np.mean(broadcasts)   # Was it "aware"?
    }


def run_experiment():
    print("="*70)
    print("DUAL-PATHWAY ARCHITECTURE EXPERIMENT")
    print("Proving that Blindsight requires Modularity, not Tuning")
    print("="*70)
    
    results = {}
    
    # === TEST 1: UNIFIED BRAIN ===
    print("\n" + "="*50)
    print("ARCHITECTURE 1: UNIFIED BRAIN")
    print("="*50)
    
    unified = UnifiedBrain()
    
    print("\n[Normal]")
    res_unified_normal = simulate_pacman_trial(unified, lesioned=False)
    print(f"  Behavior (Motor):     {res_unified_normal['behavior_score']:.1%}")
    print(f"  Consciousness (Ign):  {res_unified_normal['consciousness_score']:.1%}")
    
    print("\n[Lesioned] (Long-range weights reduced 90%)")
    unified.lesion_global(severity=0.9)
    res_unified_lesion = simulate_pacman_trial(unified, lesioned=True)
    print(f"  Behavior (Motor):     {res_unified_lesion['behavior_score']:.1%}")
    print(f"  Consciousness (Ign):  {res_unified_lesion['consciousness_score']:.1%}")
    
    unified_dissociation = (res_unified_normal['consciousness_score'] - res_unified_lesion['consciousness_score']) - \
                           (res_unified_normal['behavior_score'] - res_unified_lesion['behavior_score'])
    print(f"\n  Dissociation Index: {unified_dissociation:.2f}")
    
    results['unified'] = {
        'normal': res_unified_normal,
        'lesion': res_unified_lesion,
        'dissociation': unified_dissociation
    }
    
    # === TEST 2: DUAL-PATHWAY BRAIN ===
    print("\n" + "="*50)
    print("ARCHITECTURE 2: DUAL-PATHWAY BRAIN")
    print("="*50)
    
    dual = DualPathwayBrain()
    
    print("\n[Normal] (Subcortical + Cortical)")
    res_dual_normal = simulate_pacman_trial(dual, lesioned=False)
    print(f"  Behavior (Motor):     {res_dual_normal['behavior_score']:.1%}")
    print(f"  Consciousness (Ign):  {res_dual_normal['consciousness_score']:.1%}")
    
    print("\n[Cortical Lesion] (Subcortical INTACT)")
    dual.lesion_cortical(severity=0.9)
    res_dual_lesion = simulate_pacman_trial(dual, lesioned=True)
    print(f"  Behavior (Motor):     {res_dual_lesion['behavior_score']:.1%}")
    print(f"  Consciousness (Ign):  {res_dual_lesion['consciousness_score']:.1%}")
    
    dual_dissociation = (res_dual_normal['consciousness_score'] - res_dual_lesion['consciousness_score']) - \
                        (res_dual_normal['behavior_score'] - res_dual_lesion['behavior_score'])
    print(f"\n  Dissociation Index: {dual_dissociation:.2f}")
    
    results['dual'] = {
        'normal': res_dual_normal,
        'lesion': res_dual_lesion,
        'dissociation': dual_dissociation
    }
    
    # === FINAL VERDICT ===
    print("\n" + "="*70)
    print("VERDICT: ARCHITECTURAL REQUIREMENT FOR BLINDSIGHT")
    print("="*70)
    
    print(f"\nUNIFIED Brain Dissociation:     {unified_dissociation:+.2f}")
    print(f"DUAL-PATHWAY Brain Dissociation: {dual_dissociation:+.2f}")
    print("-" * 50)
    
    if dual_dissociation > unified_dissociation + 0.1:
        print("RESULT: DUAL-PATHWAY achieves GREATER dissociation!")
        print("        Blindsight IS possible with modular architecture.")
        print("        Blindsight IS NOT possible with unified architecture.")
        print("\n        ==> ZOMBIES REQUIRE MODULARITY <==")
    elif dual_dissociation > 0.1:
        print("RESULT: DUAL-PATHWAY shows dissociation (Consciousness drops > Behavior)")
        print("        This is BLINDSIGHT: Action without Awareness.")
    else:
        print("RESULT: Neither architecture achieved significant dissociation.")
    
    # Show the behavioral comparison
    print("\n" + "-"*50)
    print("POST-LESION BEHAVIOR COMPARISON:")
    print(f"  UNIFIED:      Behavior = {res_unified_lesion['behavior_score']:.1%}")
    print(f"  DUAL-PATHWAY: Behavior = {res_dual_lesion['behavior_score']:.1%}")
    
    if res_dual_lesion['behavior_score'] > res_unified_lesion['behavior_score'] + 0.1:
        print("\n  The DUAL-PATHWAY brain SURVIVED the lesion behaviorally!")
        print("  The UNIFIED brain collapsed.")
        print("  ==> PROOF: Modularity enables Zombies <==")

if __name__ == "__main__":
    run_experiment()
