#!/usr/bin/env python3
"""
CL1 CONSCIOUSNESS EXPERIMENT
============================
Unified mock experiment integrating Cortical Labs CL1 SDK patterns,
Perspectival Analysis (PA) framework, and Consciousness Markers.

Framework: Perspectival Realism
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PA'))
from pa_toolkit.analysis import PerspectivalAnalysis


def generate_condition_data(condition, n_points=1000, n_channels=50, seed=None):
    """Generate mock MEA data for experimental conditions."""
    if seed is not None:
        np.random.seed(seed)
    
    if condition == "spontaneous":
        data = np.random.poisson(0.5, (n_points, n_channels)).astype(float)
        data += np.random.normal(0, 0.1, data.shape)
        
    elif condition == "stimulated":
        data = np.zeros((n_points, n_channels))
        t = np.linspace(0, 10, n_points)
        stim = np.sin(2 * np.pi * 10 * t)
        for i in range(n_channels // 2):
            data[:, i] = stim * np.cos(np.random.uniform(0, np.pi))
            data[:, i] += np.random.normal(0, 0.3, n_points)
        data[:, n_channels//2:] = np.random.normal(0, 0.5, (n_points, n_channels//2))
        
    elif condition == "learned":
        data = np.zeros((n_points, n_channels))
        n_modules = 5
        t = np.linspace(0, 10, n_points)
        for mod in range(n_modules):
            start = mod * (n_channels // n_modules)
            end = (mod + 1) * (n_channels // n_modules)
            freq = 5 + mod * 3
            osc = np.sin(2 * np.pi * freq * t)
            for ch in range(start, min(end, n_channels)):
                data[:, ch] = osc * np.cos(np.random.uniform(0, 2*np.pi))
                data[:, ch] += np.random.normal(0, 0.2, n_points)
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                if np.random.rand() < 0.05:
                    data[:, j] += 0.3 * data[:, i]
                    
    elif condition == "degraded":
        t = np.linspace(0, 10, n_points)
        global_osc = np.sin(2 * np.pi * 3 * t)
        data = np.zeros((n_points, n_channels))
        for ch in range(n_channels):
            data[:, ch] = global_osc + np.random.normal(0, 0.05, n_points)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    return (data - data.mean()) / (data.std() + 1e-10) * 0.1


def run_experiment(quick=False):
    """Run CL1 consciousness experiment with PA analysis."""
    conditions = ["spontaneous", "stimulated", "learned", "degraded"]
    n_trials = 2 if quick else 3
    n_channels = 30 if quick else 50
    n_points = 500 if quick else 1000
    
    print("=" * 60)
    print("CL1 CONSCIOUSNESS EXPERIMENT")
    print("Perspectival Analysis Framework")
    print("=" * 60)
    
    pa = PerspectivalAnalysis()
    results = []
    
    for cond in conditions:
        print(f"\n--- {cond.upper()} ---")
        for trial in range(n_trials):
            data = generate_condition_data(cond, n_points, n_channels, seed=42+trial+hash(cond)%1000)
            assessment = pa.assess(data, robust=True)
            
            result = {
                "condition": cond,
                "trial": trial + 1,
                "c_score": assessment.c_score,
                "closure": assessment.components.closure,
                "lambda2": assessment.components.lambda2_norm,
                "rho": assessment.components.rho,
                "lzc": assessment.components.lzc
            }
            results.append(result)
            print(f"  Trial {trial+1}: C-Score={result['c_score']:.4f}")
    
    df = pd.DataFrame(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    means = df.groupby("condition")["c_score"].mean().sort_values(ascending=False)
    print("\nMean C-Score by condition:")
    for cond, score in means.items():
        print(f"  {cond:12s}: {score:.4f}")
    
    # Hypothesis check
    print("\n" + "=" * 60)
    print("HYPOTHESIS VALIDATION")
    print("=" * 60)
    
    checks = [
        ("learned > spontaneous", means.get("learned",0) > means.get("spontaneous",0)),
        ("stimulated > spontaneous", means.get("stimulated",0) > means.get("spontaneous",0)),
        ("learned > stimulated", means.get("learned",0) > means.get("stimulated",0)),
        ("degraded < spontaneous", means.get("degraded",1) < means.get("spontaneous",0)),
    ]
    
    all_pass = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False
    
    print("\n" + ("★ ALL HYPOTHESES CONFIRMED ★" if all_pass else "⚠ SOME HYPOTHESES FAILED"))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # C-Score by condition
    cond_order = ["degraded", "spontaneous", "stimulated", "learned"]
    colors = ["#dc3545", "#ffc107", "#17a2b8", "#28a745"]
    ax = axes[0]
    for i, cond in enumerate(cond_order):
        vals = df[df["condition"] == cond]["c_score"]
        ax.bar(i, vals.mean(), color=colors[i], alpha=0.7, label=cond)
        ax.scatter([i]*len(vals), vals, color="black", s=20, zorder=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(cond_order)
    ax.set_ylabel("C-Score")
    ax.set_title("C-Score by Condition")
    ax.axhline(0.35, color="red", linestyle="--", label="Φ* threshold")
    ax.legend()
    
    # Components
    ax = axes[1]
    comp_means = df.groupby("condition")[["closure", "lambda2", "rho"]].mean()
    comp_means = comp_means.loc[cond_order]
    comp_means.plot(kind="bar", ax=ax, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("PA Components by Condition")
    ax.set_ylabel("Value")
    ax.legend(title="Component")
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "cl1_experiment_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved: {save_path}")
    plt.close()
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CL1 Consciousness Experiment")
    parser.add_argument("--quick", action="store_true", help="Fast mode for testing")
    args = parser.parse_args()
    
    df = run_experiment(quick=args.quick)
