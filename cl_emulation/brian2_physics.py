"""
Brian2 Physics Engine
=====================
Brian2-based Izhikevich neuron model with STDP plasticity.
API-compatible with NeuralPhysics for drop-in replacement.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import os

# Brian2 import with fallback
try:
    import brian2 as b2
    BRIAN2_AVAILABLE = True
except ImportError:
    BRIAN2_AVAILABLE = False
    print("[WARNING] Brian2 not available, falling back to numpy physics")


@dataclass
class PhysicsConfig:
    """Configuration matching original NeuralPhysics."""
    n_neurons: int = 300
    n_exc: int = 240         # 80% Excitatory
    n_inh: int = 60          # 20% Inhibitory
    dt: float = 0.5          # Time step (ms)
    
    # Izhikevich parameters - Regular spiking (RS) for excitatory
    a_exc: float = 0.02
    b_exc: float = 0.2
    c_exc: float = -65.0
    d_exc: float = 8.0
    
    # Fast spiking (FS) for inhibitory
    a_inh: float = 0.1
    b_inh: float = 0.2
    c_inh: float = -65.0
    d_inh: float = 2.0
    
    # STDP
    stdp_enabled: bool = True
    learning_rate: float = 0.05
    max_weight: float = 10.0


class Brian2Physics:
    """
    Brian2-powered Izhikevich physics engine.
    Exposes same API as NeuralPhysics for compatibility.
    """
    
    def __init__(self, config: PhysicsConfig = None):
        if not BRIAN2_AVAILABLE:
            raise ImportError("Brian2 is required for Brian2Physics")
        
        self.cfg = config or PhysicsConfig()
        self.t = 0.0
        
        # Use numpy codegen for compatibility
        b2.prefs.codegen.target = "numpy"
        b2.start_scope()
        
        # Set clock
        b2.defaultclock.dt = self.cfg.dt * b2.ms
        
        # Izhikevich equations (dimensionless form)
        eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I_ext + I_syn)/ms : 1
        du/dt = a*(b*v - u)/ms : 1
        I_ext : 1  # External current injection
        I_syn : 1  # Synaptic current
        a : 1
        b : 1
        c : 1
        d : 1
        '''
        
        # Create neuron group
        self._group = b2.NeuronGroup(
            self.cfg.n_neurons, 
            eqs, 
            threshold='v >= 30', 
            reset='v = c; u = u + d',
            method='euler'
        )
        
        # Set parameters per neuron type
        n_exc = self.cfg.n_exc
        
        # Excitatory neurons (Regular Spiking)
        self._group.v[:] = -65
        self._group.u[:] = self._group.v[:] * self.cfg.b_exc
        
        self._group.a[:n_exc] = self.cfg.a_exc
        self._group.b[:n_exc] = self.cfg.b_exc
        self._group.c[:n_exc] = self.cfg.c_exc
        self._group.d[:n_exc] = self.cfg.d_exc
        
        # Inhibitory neurons (Fast Spiking)
        self._group.a[n_exc:] = self.cfg.a_inh
        self._group.b[n_exc:] = self.cfg.b_inh
        self._group.c[n_exc:] = self.cfg.c_inh
        self._group.d[n_exc:] = self.cfg.d_inh
        
        self._group.I_ext = 0
        self._group.I_syn = 0
        
        # Create synapses with STDP
        self._create_synapses()
        
        # Spike monitor
        self._spike_mon = b2.SpikeMonitor(self._group)
        
        # Build network
        self._net = b2.Network(self._group, self._synapses, self._spike_mon)
        self._net.store('initial')
        
        # Track last spike check time
        self._last_spike_count = 0
        
        # Cached membrane potentials
        self._v_cache = np.array(self._group.v[:])
        self._u_cache = np.array(self._group.u[:])
        
    def _create_synapses(self):
        """Create synapses with STDP plasticity."""
        # Synapse model with STDP
        if self.cfg.stdp_enabled:
            synapse_eqs = '''
            w : 1  # Synaptic weight
            dApre/dt = -Apre / (20*ms) : 1 (event-driven)
            dApost/dt = -Apost / (20*ms) : 1 (event-driven)
            '''
            
            on_pre = '''
            I_syn_post += w
            Apre += 0.01
            w = clip(w + Apost * {lr}, 0, {max_w})
            '''.format(lr=self.cfg.learning_rate, max_w=self.cfg.max_weight)
            
            on_post = '''
            Apost += -0.01
            w = clip(w + Apre * {lr}, 0, {max_w})
            '''.format(lr=self.cfg.learning_rate, max_w=self.cfg.max_weight)
            
            self._synapses = b2.Synapses(
                self._group, self._group,
                synapse_eqs,
                on_pre=on_pre,
                on_post=on_post
            )
        else:
            self._synapses = b2.Synapses(
                self._group, self._group,
                'w : 1',
                on_pre='I_syn_post += w'
            )
        
        # Random 20% connectivity
        self._synapses.connect(p=0.2)
        
        # Initialize weights
        self._synapses.w[:] = np.random.rand(len(self._synapses)) * self.cfg.max_weight
        
        # Inhibitory synapses are negative (from neurons n_exc onwards)
        n_exc = self.cfg.n_exc
        inh_mask = self._synapses.i[:] >= n_exc
        self._synapses.w[inh_mask] *= -1
        
    @property
    def v(self) -> np.ndarray:
        """Membrane potentials (compatibility with NeuralPhysics)."""
        return self._v_cache
    
    @property
    def u(self) -> np.ndarray:
        """Recovery variables (compatibility with NeuralPhysics)."""
        return self._u_cache
    
    @property
    def S(self) -> np.ndarray:
        """Synaptic weight matrix (for compatibility, returns dense matrix)."""
        n = self.cfg.n_neurons
        S = np.zeros((n, n))
        S[self._synapses.j[:], self._synapses.i[:]] = self._synapses.w[:]
        return S
    
    def step(self, external_current: np.ndarray = None) -> np.ndarray:
        """
        Run one time step of the simulation.
        
        Args:
            external_current: Array of currents to inject (length n_neurons)
            
        Returns:
            Array of neuron indices that fired this step
        """
        # Apply external current
        if external_current is not None:
            # Add noise like original
            noisy_input = external_current + np.random.normal(0, 2.0, self.cfg.n_neurons)
            self._group.I_ext[:] = noisy_input
        else:
            self._group.I_ext[:] = np.random.normal(0, 2.0, self.cfg.n_neurons)
        
        # Decay synaptic current
        self._group.I_syn[:] *= 0.9
        
        # Run one step
        self._net.run(self.cfg.dt * b2.ms)
        
        # Update time
        self.t += self.cfg.dt
        
        # Update cached values
        self._v_cache = np.array(self._group.v[:])
        self._u_cache = np.array(self._group.u[:])
        
        # Get spikes from this step
        all_spikes_i = np.array(self._spike_mon.i[:])
        current_count = len(all_spikes_i)
        
        if current_count > self._last_spike_count:
            fired = all_spikes_i[self._last_spike_count:]
        else:
            fired = np.array([], dtype=int)
            
        self._last_spike_count = current_count
        
        return fired
    
    def set_weights_scale(self, scale: float):
        """Scale all synaptic weights."""
        self._synapses.w[:] *= scale
    
    @property
    def current_input(self) -> np.ndarray:
        """Current input (for compatibility)."""
        return np.array(self._group.I_syn[:])
    
    @property 
    def last_spike_times(self) -> np.ndarray:
        """Last spike times for each neuron."""
        result = -1000.0 * np.ones(self.cfg.n_neurons)
        if len(self._spike_mon.i) > 0:
            for i, t in zip(self._spike_mon.i[:], self._spike_mon.t[:]):
                result[i] = float(t / b2.ms)
        return result
    
    @property
    def fired(self) -> np.ndarray:
        """Indices of neurons that fired in last step."""
        all_i = np.array(self._spike_mon.i[:])
        if self._last_spike_count > 0 and len(all_i) >= self._last_spike_count:
            return all_i[max(0, self._last_spike_count - 1):]
        return np.array([], dtype=int)
