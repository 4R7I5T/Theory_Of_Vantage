#!/usr/bin/env python3
"""
PA FRAMEWORK TEST: Perspectival Analysis Measures
==================================================
Testing the PA framework's three operational measures of consciousness:
1. Causal Closure - Fraction of influence internal to system
2. λ₂-Integration - Spectral gap of Laplacian (integration)
3. Self-Model Fraction - Recurrent self-representation

PA Hypotheses (Falsifiable):
- H1: All three measures are necessary (multi-component)
- H2: C-Score > any single component (composite superiority)
- H3: Phase transitions, not gradual changes

If PA claims fail, this is a valid experimental finding.
"""

import sys
import os
import numpy as np
from scipy import linalg
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cl_emulation as cl
from cl_emulation.ghost_brain import GhostBrain, GhostBrainEnsemble


def calculate_closure(weight_matrix):
    """
    PA Measure 1: Causal Closure
    
    Closure(S) = Σ|W_ij| / (Σ|W_ij| + Σ|W_ik| + Σ|W_ki|)
    
    For a self-contained system, this simplifies to internal / (internal + boundary).
    Since our systems are isolated (no external), Closure ≈ 1.0.
    For feedforward, we measure how much loops back.
    """
    W = np.abs(weight_matrix)
    n = W.shape[0]
    
    # Total internal connections
    internal = np.sum(W)
    
    # For recurrent: count bidirectional connections
    recurrent_strength = 0
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0 and W[j, i] > 0:
                recurrent_strength += min(W[i, j], W[j, i])
    
    # Closure = recurrent / total (how much loops back)
    if internal > 0:
        closure = recurrent_strength / internal
    else:
        closure = 0.0
    
    return closure


def calculate_lambda2(weight_matrix):
    """
    PA Measure 2: λ₂-Integration (Spectral Gap)
    
    Second smallest eigenvalue of the graph Laplacian.
    Higher λ₂ = more connected = harder to partition.
    """
    W = np.abs(weight_matrix)
    n = W.shape[0]
    
    # Degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    # Laplacian L = D - W
    L = D - W
    
    # Eigenvalues (sorted ascending)
    eigenvalues = np.sort(linalg.eigvalsh(L))
    
    # λ₂ is the second smallest (first is always 0 for connected graph)
    lambda2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    
    # Normalize by max eigenvalue for comparability
    lambda_max = eigenvalues[-1] if eigenvalues[-1] > 0 else 1.0
    lambda2_normalized = lambda2 / lambda_max
    
    return lambda2_normalized


def calculate_self_model_fraction(weight_matrix, trace_matrix=None):
    """
    PA Measure 3: Self-Model Fraction (ρ)
    
    Proportion of system that represents its own internal state.
    Approximated by: recurrent connections / total connections.
    """
    W = np.abs(weight_matrix)
    n = W.shape[0]
    
    # Count self-loops and recurrent pathways
    self_connections = np.trace(W)  # Diagonal = self-loops
    
    # Count neurons that receive input from themselves (via 2-step paths)
    W2 = W @ W
    recurrent_paths = np.trace(W2)
    
    total_paths = np.sum(W2) if np.sum(W2) > 0 else 1.0
    
    rho = (self_connections + recurrent_paths) / (np.sum(W) + total_paths)
    rho = min(1.0, rho)  # Cap at 1.0
    
    return rho


def calculate_cscore(closure, lambda2, rho, alpha=0.33, beta=0.33, gamma=0.34):
    """
    PA Composite: C-Score
    
    C-Score = α·Closure + β·λ₂ + γ·ρ
    
    Default: equal weights (agnostic prior)
    """
    return alpha * closure + beta * lambda2 + gamma * rho


def get_ghost_connectivity():
    """
    Create connectivity matrix for feedforward Ghost brain.
    """
    ghost = GhostBrain(seed=42)
    # Feedforward: input -> hidden -> output (no recurrence)
    n_total = ghost.n_input + ghost.n_hidden + ghost.n_output
    W = np.zeros((n_total, n_total))
    
    # Input to hidden
    W[:ghost.n_input, ghost.n_input:ghost.n_input + ghost.n_hidden] = np.abs(ghost.W1)
    # Hidden to output
    W[ghost.n_input:ghost.n_input + ghost.n_hidden, 
      ghost.n_input + ghost.n_hidden:] = np.abs(ghost.W2)
    
    return W


def run_pa_test():
    print("="*60)
    print("PA FRAMEWORK TEST: Perspectival Analysis Measures")
    print("============================================================")
    
    # 1. SETUP
    print("\n[INIT] Extracting connectivity matrices...")
    
    # Pacman (Recurrent Izhikevich)
    pacman = cl.open()
    pacman_W = np.abs(pacman._physics.S)
    print(f"      Pacman: {pacman_W.shape[0]} neurons, {np.sum(pacman_W > 0)} connections")
    
    # Ghost (Feedforward MLP)
    ghost_W = get_ghost_connectivity()
    print(f"      Ghost:  {ghost_W.shape[0]} neurons, {np.sum(ghost_W > 0)} connections")
    
    # 2. COMPUTE PA MEASURES
    print("\n[TEST] Computing PA Measures...")
    
    # Pacman measures
    p_closure = calculate_closure(pacman_W)
    p_lambda2 = calculate_lambda2(pacman_W)
    p_rho = calculate_self_model_fraction(pacman_W)
    p_cscore = calculate_cscore(p_closure, p_lambda2, p_rho)
    
    print(f"      Pacman Closure: {p_closure:.4f}")
    print(f"      Pacman λ₂:      {p_lambda2:.4f}")
    print(f"      Pacman ρ:       {p_rho:.4f}")
    print(f"      Pacman C-Score: {p_cscore:.4f}")
    
    # Ghost measures
    g_closure = calculate_closure(ghost_W)
    g_lambda2 = calculate_lambda2(ghost_W)
    g_rho = calculate_self_model_fraction(ghost_W)
    g_cscore = calculate_cscore(g_closure, g_lambda2, g_rho)
    
    print(f"\n      Ghost Closure:  {g_closure:.4f}")
    print(f"      Ghost λ₂:       {g_lambda2:.4f}")
    print(f"      Ghost ρ:        {g_rho:.4f}")
    print(f"      Ghost C-Score:  {g_cscore:.4f}")
    
    # 3. PA HYPOTHESIS TESTS
    print("\n" + "="*60)
    print("PA HYPOTHESIS FALSIFICATION")
    print("="*60)
    
    # H1: Multi-Component Necessity
    # Test: Does any single component separate Pacman/Ghost as well as C-Score?
    components = {
        'Closure': (p_closure, g_closure),
        'λ₂': (p_lambda2, g_lambda2),
        'ρ': (p_rho, g_rho),
        'C-Score': (p_cscore, g_cscore)
    }
    
    separations = {}
    for name, (p_val, g_val) in components.items():
        sep = abs(p_val - g_val)
        separations[name] = sep
        print(f"      {name}: Pacman={p_val:.4f}, Ghost={g_val:.4f}, Separation={sep:.4f}")
    
    # Find best single component
    best_single = max(['Closure', 'λ₂', 'ρ'], key=lambda x: separations[x])
    cscore_sep = separations['C-Score']
    best_single_sep = separations[best_single]
    
    print(f"\n[H1] Multi-Component Necessity:")
    print(f"      Best single component: {best_single} (sep={best_single_sep:.4f})")
    print(f"      C-Score separation:    {cscore_sep:.4f}")
    
    if cscore_sep > best_single_sep * 1.1:  # C-Score 10% better
        print("      VERDICT: PA H1 SUPPORTED - C-Score > single component")
    else:
        print("      VERDICT: PA H1 FALSIFIED - Single component works as well")
    
    # H2: Composite Superiority
    print(f"\n[H2] Composite Superiority:")
    if p_cscore > p_closure and p_cscore > p_lambda2 and p_cscore > p_rho:
        print("      VERDICT: PA H2 SUPPORTED - C-Score > all components for Pacman")
    else:
        print("      VERDICT: PA H2 FALSIFIED - Some component >= C-Score")
    
    # H3: Phase Transition (would need gradual lesion study)
    print(f"\n[H3] Phase Transition: REQUIRES GRADUAL LESION STUDY")
    
    # 4. SUMMARY
    print("\n" + "="*60)
    print("SUMMARY: PA FRAMEWORK vs ARCHITECTURE")
    print("="*60)
    
    print(f"""
    | Measure    | Pacman (Recurrent) | Ghost (Feedforward) |
    |------------|-------------------|---------------------|
    | Closure    | {p_closure:.4f}            | {g_closure:.4f}               |
    | λ₂         | {p_lambda2:.4f}            | {g_lambda2:.4f}               |
    | ρ          | {p_rho:.4f}            | {g_rho:.4f}               |
    | C-Score    | {p_cscore:.4f}            | {g_cscore:.4f}               |
    """)
    
    if p_cscore > g_cscore:
        print("CONCLUSION: PA measures correctly identify recurrent > feedforward")
    else:
        print("CONCLUSION: PA measures FAIL to distinguish architectures")


if __name__ == "__main__":
    run_pa_test()
