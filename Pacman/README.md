# Neural Pacman: Consciousness vs Zombies 🧠👻

> **A Falsifiable Test of Machine Consciousness using the Theory of Vantage**

This experiment implements a "Conscious" Agent (Pacman) and competes it against "Zombie" Agents (Ghosts) to demonstrate the functional differences between **Recurrent Directed Information Integration** (Consciousness) and **Feedforward Reflexive Processing** (Zombies).

## 🔬 The Hypothesis

The **Zombie Argument** (Chalmers) suggests a system could behave identically to a conscious being without having subjective experience. We test this by building two distinct neural architectures:

1.  **The Conscious Agent (Pacman)**: built with a **Recurrent Spiking Neural Network (SNN)**.
    *   **Architecture**: 300 Izhikevich Neurons (Brian2 Simulation).
    *   **Dynamics**: Temporal Integration. Input currents triggers avalanches of activity that sustain over time.
    *   **Key Feature**: Decisions are independent of the *immediate* sensory frame; they depend on the *internal state history*.

2.  **The Zombie Agents (Ghosts)**: built with **Feedforward Control Systems**.
    *   **Architecture**: State-Machine / Feedforward Logic.
    *   **Dynamics**: Instantaneous Reflex.
    *   **Key Feature**: Output is a pure function of the current input ($O = f(I)$). No internal "now" exists.

## 🧠 Neural Architecture

The Pacman agent is powered by a biologically realistic simulation of cortical tissue.

### The Brain (300 Neurons)
*   **Sensory Cortex (60 Neurons)**: Topologically mapped to game features.
    *   Ghost Proximity (0-19)
    *   Valid Pathways (20-39)
    *   Reward Gradients (40-59)
*   **Association Cortex (180 Neurons)**: Recurrently connected "Global Workspace".
    *   Excitatory (Regular Spiking) & Inhibitory (Fast Spiking) populations.
    *   **STDP Plasticity**: Synapses strengthen/weaken based on causal firing timing, enabling learning.
*   **Motor Cortex (60 Neurons)**: Decodes population firing rates into movement vectors.
    *   Winning population drives the agent.

### The Physics Engine (`cl-sdk-izhikevich`)
We use **Brian2**, a high-fidelity simulator for spiking neural networks, to model:
*   **Izhikevich Neuron Model**: $dv/dt = 0.04v^2 + 5v + 140 - u + I$
*   **Plasticity**: Spike-Timing Dependent Plasticity (STDP).
*   **Noise**: Stochastic current injection to simulate thermal biological noise.

## 📊 Signatures of Consciousness

The experiment visualizes real-time metrics that differentiate the two architectures:

| Signature | Conscious Agent (Pacman) | Zombie Agent (Ghost) |
|-----------|--------------------------|----------------------|
| **Ignition** | Global firing events after stimulus | Localized activation only |
| **Latency** | Variable (decision processing time) | Fixed / Near-zero |
| **Adaptation** | Learning via STDP | Static behavior |
| **Architecture**| Unified / Irreducible | Modular / Reducible |

## 🚀 Running the Experiment

### 1. Simulation Mode (Brian2)
Runs on your CPU using the biologically realistic Izhikevich model.

```bash
# Enable Brian2 backend
export CL_USE_BRIAN2=1 

# Run the experiment
python3 Pacman/neural_pacman.py
```

### 2. Live Visualization
The right-hand panel displays the **Real-Time Neural Activity**:
*   **Raster Plot**: Shows individual neuron firing spikes.
*   **Membrane Potentials**: Visualizes the sub-threshold voltage state of the network.
*   **Latency Monitor**: Tracks the time taken for the decision to emerge from the network dynamics.

---

## 🛠 Project Structure

*   `neural_pacman.py`: Main entry point. Couples the PyGame engine to the Neural Network.
*   `cl_emulation/`: The Simulation SDK.
    *   `brian2_physics.py`: The core physics engine (Izhikevich + STDP).
    *   `neurons.py`: The API layer mimicking a Multi-Electrode Array (MEA).
*   `wetware_pacman.py`: Adapter for running this exact code on **Real Biological Neurons** (CL1 Hardware).

## 📄 License
MIT License - Part of the Theory of Vantage Research Project.
