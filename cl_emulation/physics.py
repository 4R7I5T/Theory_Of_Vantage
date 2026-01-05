"""
Neural Physics Engine
=====================
Vectorized Izhikevich neuron model with STDP plasticity.
Simulates ~300 neurons in real-time.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class PhysicsConfig:
    n_neurons: int = 300     # Reduced from 1000 for performance
    n_exc: int = 240         # 80% Excitatory
    n_inh: int = 60          # 20% Inhibitory
    dt: float = 0.5          # Time step (ms)
    
    # Izhikevich parameters
    # Regular spiking (RS) for excitatory
    a_exc: float = 0.02
    b_exc: float = 0.2
    c_exc: float = -65.0
    d_exc: float = 8.0
    
    # Fast spiking (FS) for inhibitory
    a_inh: float = 0.1
    b_inh: float = 0.2
    c_inh: float = -65.0
    d_inh: float = 2.0
    
    # Synaptic Plasticity
    stdp_enabled: bool = True
    learning_rate: float = 0.05
    max_weight: float = 10.0


class NeuralPhysics:
    def __init__(self, config: PhysicsConfig = PhysicsConfig()):
        self.cfg = config
        self.t = 0.0
        
        # Initialize State Variables
        self.v = -65.0 * np.ones(config.n_neurons)
        self.u = self.v * config.b_exc
        
        # Neuron Parameters (Vectorized)
        self.a = np.concatenate([np.ones(config.n_exc)*config.a_exc, np.ones(config.n_inh)*config.a_inh])
        self.b = np.concatenate([np.ones(config.n_exc)*config.b_exc, np.ones(config.n_inh)*config.b_inh])
        self.c = np.concatenate([np.ones(config.n_exc)*config.c_exc, np.ones(config.n_inh)*config.c_inh])
        self.d = np.concatenate([np.ones(config.n_exc)*config.d_exc, np.ones(config.n_inh)*config.d_inh])
        
        # Synaptic Weights
        self.S = np.random.rand(config.n_neurons, config.n_neurons) 
        mask = np.random.rand(config.n_neurons, config.n_neurons) < 0.2 # 20% connectivity
        self.S *= mask
        self.S *= config.max_weight
        self.S[:, config.n_exc:] *= -1.0
        
        self.last_spike_times = -1000.0 * np.ones(config.n_neurons)
        self.current_input = np.zeros(config.n_neurons)
        self.fired = np.array([], dtype=int)

    def step(self, external_current: np.ndarray = None) -> np.ndarray:
        dt = self.cfg.dt
        n = self.cfg.n_neurons
        
        # 1. Input
        I = self.current_input.copy() + np.random.normal(0, 0.5, n)
        if external_current is not None:
            I += external_current
            
        # 2. Dynamics
        v, u = self.v, self.u
        dv = (0.04 * v**2 + 5 * v + 140 - u + I)
        v_next = v + dv * dt
        du = self.a * (self.b * v - u)
        u_next = u + du * dt
        
        # 3. Spiking
        fired_indices = np.where(v_next >= 30.0)[0]
        
        if len(fired_indices) > 0:
            v_next[fired_indices] = self.c[fired_indices]
            u_next[fired_indices] += self.d[fired_indices]
            
            # Synaptic Input
            incoming_spikes = np.zeros(n)
            incoming_spikes[fired_indices] = 1.0
            self.current_input = self.S @ incoming_spikes
            
            # STDP
            if self.cfg.stdp_enabled:
                self._apply_stdp(fired_indices)
                
            self.last_spike_times[fired_indices] = self.t

        else:
            self.current_input *= 0.9
            
        self.v = v_next
        self.u = u_next
        self.fired = fired_indices
        self.t += dt
        
        return fired_indices

    def _apply_stdp(self, fired_indices: np.ndarray):
        """Vectorized STDP."""
        t = self.t
        window = 20.0
        lr = self.cfg.learning_rate
        
        # Find presynaptic spikes in window
        delta_t = t - self.last_spike_times
        pre_indices = np.where((delta_t > 0) & (delta_t < window))[0]
        
        if len(pre_indices) == 0:
            return

        # Restrict to excitatory pre-synaptic neurons
        pre_indices = pre_indices[pre_indices < self.cfg.n_exc]
        
        if len(pre_indices) == 0:
            return
            
        # Calculate weight changes
        # dw[j] = lr * exp(-delta_t[j]/10)
        dw = lr * np.exp(-delta_t[pre_indices] / 10.0)
        
        # Update S[post, pre] for all combinations
        rows = fired_indices
        cols = pre_indices
        
        # Vectorized block update using advanced indexing
        current_weights = self.S[rows[:, None], cols]
        new_weights = np.minimum(current_weights + dw, self.cfg.max_weight)
        self.S[rows[:, None], cols] = new_weights

    def set_weights_scale(self, scale: float):
        self.S *= scale
