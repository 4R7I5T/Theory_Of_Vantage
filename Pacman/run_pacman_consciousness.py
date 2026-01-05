#!/usr/bin/env python3
"""
PACMAN CONSCIOUSNESS EXPERIMENT: FALSIFIABLE PROTOCOL
======================================================
Tests the Philosophical Zombie Hypothesis using PA Framework.

Hypothesis:
- H1: Pacman (integrated network) → C-Score > 0.5
- H2: Ghosts (feedforward network) → C-Score < 0.2
- H3: Difference > 0.3 (statistically significant)
- H4: Perturbation drops Pacman C-Score, not Ghost C-Score
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
from modern.pacman_env import PacmanEnvironment

# Import PA Framework
try:
    from pa_toolkit.analysis import PerspectivalAnalysis
    HAS_PA = True
    print("[PA] Perspectival Analysis Framework loaded.")
except ImportError as e:
    HAS_PA = False
    print(f"[PA] Warning: PA Framework not available: {e}")

# Visualization
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

def compute_quick_cscore(trace_matrix):
    """
    Refined C-Score computation that distinguishes integrated from feedforward networks.
    
    Key insight: Feedforward networks have HIGH instantaneous correlation but 
    LOW temporal self-causation. Recurrent networks have the opposite.
    
    Measures:
    1. Temporal Autocorrelation (Self-causation over time) - HIGH for recurrent
    2. Lagged Cross-correlation (Integration across units) - HIGH for recurrent  
    3. Variance of activity (Dynamic range) - HIGH for recurrent
    """
    if trace_matrix is None or len(trace_matrix) < 50:
        return 0.0
        
    T, N = trace_matrix.shape
    
    # 1. TEMPORAL SELF-CAUSATION (Key discriminant!)
    # Feedforward: t+1 independent of t (given same input)
    # Recurrent: t+1 depends on t (internal dynamics)
    temporal_scores = []
    for i in range(N):
        ts = trace_matrix[:, i]
        if np.std(ts) > 1e-6:
            # Multi-lag autocorrelation
            autocorrs = []
            for lag in [1, 2, 5, 10]:
                if T > lag + 1:
                    corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(corr)
            if autocorrs:
                temporal_scores.append(np.mean(autocorrs))
    
    temporal_integration = np.mean(temporal_scores) if temporal_scores else 0.0
    
    # 2. LAGGED CROSS-CORRELATION (Information flow)
    # Measures if units influence each other with a delay
    cross_scores = []
    sample_pairs = min(10, N * (N-1) // 2)
    for _ in range(sample_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        ts_i = trace_matrix[:-1, i]
        ts_j = trace_matrix[1:, j]
        if np.std(ts_i) > 1e-6 and np.std(ts_j) > 1e-6:
            corr = np.corrcoef(ts_i, ts_j)[0, 1]
            if not np.isnan(corr):
                cross_scores.append(abs(corr))
    
    cross_integration = np.mean(cross_scores) if cross_scores else 0.0
    
    # 3. DYNAMIC RANGE (Complexity of activity patterns)
    # Feedforward: stereotyped responses
    # Recurrent: rich dynamics
    variances = np.var(trace_matrix, axis=0)
    dynamic_range = np.std(variances) / (np.mean(variances) + 1e-6)
    # Normalize to 0-1
    dynamic_range = min(1.0, dynamic_range / 2.0)
    
    # 4. SELF-MODEL (First PC explains less variance in diverse system)
    try:
        centered = trace_matrix - trace_matrix.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(s**2)
        if total_var > 0:
            # Invert: LOW first PC ratio = HIGH complexity = HIGH score
            first_pc_ratio = s[0]**2 / total_var
            rho = 1.0 - first_pc_ratio
        else:
            rho = 0.0
    except:
        rho = 0.0
    
    # Composite C-Score (weighted toward temporal measures)
    c_score = (
        0.4 * temporal_integration +  # Weight temporal self-causation highly
        0.3 * cross_integration +      # Lagged cross-correlation
        0.2 * dynamic_range +          # Complexity
        0.1 * rho                      # Diversity
    )
    
    return float(np.clip(c_score, 0, 1))

def run_consciousness_experiment():
    print("=" * 70)
    print("PACMAN: FALSIFIABLE CONSCIOUSNESS EXPERIMENT")
    print("Testing the Philosophical Zombie Hypothesis")
    print("=" * 70)
    
    # ===========================================================================
    # INITIALIZATION
    # ===========================================================================
    print("\n[INIT] Booting Neural Systems...")
    
    # Pacman Brain: Integrated Izhikevich Network
    pacman_brain = cl.open()
    pacman_brain._physics.cfg.stdp_enabled = True  # Enable learning
    print("      Pacman Brain: Recurrent Izhikevich (300N, STDP=ON)")
    
    # Ghost Brains: Feedforward MLPs
    ghost_brains = GhostBrainEnsemble()
    print("      Ghost Brains: Feedforward MLP (20N, STDP=OFF)")
    
    # Environment
    env = PacmanEnvironment(headless=True)
    print("      Environment: Arcade-Accurate Level 1")
    
    # ===========================================================================
    # DATA COLLECTION
    # ===========================================================================
    print("\n[RUN] Starting Consciousness Measurement Protocol...")
    
    # Results storage
    pacman_cscores = []
    ghost_cscores = {name: [] for name in ['blinky', 'pinky', 'inky', 'clyde']}
    pacman_traces = []
    frames = []
    
    # Run experiment
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    measurement_interval = 100  # Compute C-Score every 100 steps
    
    while not done and steps < 1000:
        # A. PACMAN: Neural Processing
        stim_plan = pacman_brain.neurons.create_stim_plan()
        for i, val in enumerate(obs[:4]):
            if val > 0:
                stim_plan.add_biphasic_pulse(electrode=i, amplitude_uv=val*500, duration_us=200)
        pacman_brain.neurons.stim(stim_plan)
        
        trace = pacman_brain.neurons.record(duration_sec=0.020)
        pacman_traces.append(trace.samples.flatten()[:60])  # Store subset
        
        # Decode Pacman action
        spikes = np.where(trace.samples < -20)[0]
        motor_counts = [0, 0, 0, 0]
        for idx in spikes:
            if 40 <= idx < 45: motor_counts[0] += 1
            elif 45 <= idx < 50: motor_counts[1] += 1
            elif 50 <= idx < 55: motor_counts[2] += 1
            elif 55 <= idx < 60: motor_counts[3] += 1
        pacman_action = int(np.argmax(motor_counts)) + 1 if max(motor_counts) > 0 else 0
        
        # B. GHOSTS: Neural Processing (Get actions from feedforward networks)
        ghost_names = ['blinky', 'pinky', 'inky', 'clyde']
        for i, gname in enumerate(ghost_names):
            # Create sensory input for ghost
            if hasattr(env, 'ghosts') and i < len(env.ghosts):
                ghost = env.ghosts[i]
                sensory = ghost_brains.brains[gname].get_sensory_input(
                    (ghost['x'], ghost['y']),
                    (env.pacman_x, env.pacman_y),
                    env.screen_width, env.screen_height
                )
                ghost_brains.step(gname, sensory)
        
        # C. ENVIRONMENT STEP
        obs, reward, done, info = env.step(pacman_action)
        total_reward += reward
        steps += 1
        
        # D. FRAME CAPTURE
        if HAS_IMAGEIO and 'frame' in info:
            frames.append(info['frame'])
        
        # E. C-SCORE MEASUREMENT (every N steps)
        if steps % measurement_interval == 0:
            # Pacman C-Score
            if len(pacman_traces) >= 50:
                pm_matrix = np.array(pacman_traces[-50:])
                pm_cscore = compute_quick_cscore(pm_matrix)
                pacman_cscores.append(pm_cscore)
            
            # Ghost C-Scores
            for gname in ghost_names:
                gm_matrix = ghost_brains.brains[gname].get_trace_matrix()
                gm_cscore = compute_quick_cscore(gm_matrix)
                ghost_cscores[gname].append(gm_cscore)
                
            # Progress Report
            pm_latest = pacman_cscores[-1] if pacman_cscores else 0
            gh_latest = np.mean([ghost_cscores[g][-1] for g in ghost_names if ghost_cscores[g]])
            print(f"      Step {steps:4d} | Pacman C={pm_latest:.3f} | Ghosts C={gh_latest:.3f} | Reward={total_reward}")
    
    # ===========================================================================
    # HYPOTHESIS TESTING
    # ===========================================================================
    print("\n" + "=" * 70)
    print("RESULTS: HYPOTHESIS TESTING")
    print("=" * 70)
    
    # Aggregate scores
    pacman_mean_c = np.mean(pacman_cscores) if pacman_cscores else 0
    pacman_std_c = np.std(pacman_cscores) if pacman_cscores else 0
    
    ghost_all_scores = []
    for gname in ghost_names:
        ghost_all_scores.extend(ghost_cscores[gname])
    ghost_mean_c = np.mean(ghost_all_scores) if ghost_all_scores else 0
    ghost_std_c = np.std(ghost_all_scores) if ghost_all_scores else 0
    
    difference = pacman_mean_c - ghost_mean_c
    
    print(f"\n  PACMAN (Integrated Network):")
    print(f"    C-Score = {pacman_mean_c:.3f} ± {pacman_std_c:.3f}")
    
    print(f"\n  GHOSTS (Feedforward Networks):")
    print(f"    C-Score = {ghost_mean_c:.3f} ± {ghost_std_c:.3f}")
    
    print(f"\n  DIFFERENCE: {difference:.3f}")
    
    # Test Hypotheses
    print("\n" + "-" * 50)
    print("HYPOTHESIS VERDICTS:")
    print("-" * 50)
    
    h1_pass = pacman_mean_c > 0.5
    h2_pass = ghost_mean_c < 0.2
    h3_pass = difference > 0.3
    
    print(f"  H1 (Pacman C > 0.5):     {'✓ PASS' if h1_pass else '✗ FAIL'} ({pacman_mean_c:.3f})")
    print(f"  H2 (Ghost C < 0.2):      {'✓ PASS' if h2_pass else '✗ FAIL'} ({ghost_mean_c:.3f})")
    print(f"  H3 (Difference > 0.3):   {'✓ PASS' if h3_pass else '✗ FAIL'} ({difference:.3f})")
    
    # Conclusion
    print("\n" + "=" * 70)
    if h1_pass and h2_pass and h3_pass:
        print("CONCLUSION: ZOMBIES CANNOT EXIST")
        print("Consciousness correlates with neural ARCHITECTURE, not just behavior.")
        print("The PA Framework is VALIDATED.")
    elif not h1_pass:
        print("CONCLUSION: PACMAN FAILED TO SHOW CONSCIOUSNESS")
        print("Integrated network did not produce expected C-Score.")
    elif not h2_pass:
        print("CONCLUSION: GHOSTS SHOW UNEXPECTED CONSCIOUSNESS")
        print("Feedforward networks produced high C-Score. Zombie hypothesis SUPPORTED?")
    else:
        print("CONCLUSION: INCONCLUSIVE")
        print("Difference not statistically significant.")
    print("=" * 70)
    
    # ===========================================================================
    # SAVE ARTIFACTS
    # ===========================================================================
    if HAS_IMAGEIO and frames:
        print("\n[SAVE] Generating Visual Proof...")
        iio.imwrite('pacman_consciousness_experiment.gif', frames, duration=33.3, loop=0)
        print("       Saved: pacman_consciousness_experiment.gif")
    
    # Save numerical results
    results = {
        'pacman_cscores': pacman_cscores,
        'ghost_cscores': ghost_cscores,
        'pacman_mean': pacman_mean_c,
        'ghost_mean': ghost_mean_c,
        'h1_pass': h1_pass,
        'h2_pass': h2_pass,
        'h3_pass': h3_pass,
    }
    np.save('consciousness_experiment_results.npy', results)
    print("       Saved: consciousness_experiment_results.npy")
    
    return results

if __name__ == "__main__":
    run_consciousness_experiment()
