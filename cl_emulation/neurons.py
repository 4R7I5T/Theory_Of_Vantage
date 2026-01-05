"""
CL1 Neurons Interface (Mock)
============================
Mimics the `cl.neurons` API, forwarding calls to the physics engine.
"""

import numpy as np
from typing import List, Callable, Optional, NamedTuple
from dataclasses import dataclass
from .physics import NeuralPhysics

class Spike(NamedTuple):
    time: int
    electrode: int

class LoopTick(NamedTuple):
    spikes: List[Spike]
    elapsed_time: float

@dataclass
class EmulatedRecording:
    samples: np.ndarray

class Neurons:
    def __init__(self, physics: NeuralPhysics):
        self._physics = physics
        self.sampling_rate = 25000 # Hz
        
        # Map 1000 neurons to 59 electrodes
        # Simple spatial mapping: electrode i listens to neurons [i*10 : i*10+10]
        # This means we only record from ~600 neurons, others are "deep tissue"
        self.n_electrodes = 59
        self.sim_neurons_per_elec = 10 
        self.neuron_to_elec_map = {}
        for e in range(self.n_electrodes):
            start = e * self.sim_neurons_per_elec
            for offset in range(self.sim_neurons_per_elec):
                n_idx = start + offset
                if n_idx < physics.cfg.n_neurons:
                    self.neuron_to_elec_map[n_idx] = e

    def create_stim_plan(self):
        """Mock stim plan creator."""
        return StimPlan()

    def stim(self, plan):
        """Apply stimulation plan to physics engine."""
        # Map electrode pulses to current injection
        batch_current = np.zeros(self._physics.cfg.n_neurons)
        
        for pulse in plan.pulses:
            # Stimulate all neurons near this electrode
            e_idx = pulse['electrode']
            start = e_idx * self.sim_neurons_per_elec
            end = start + self.sim_neurons_per_elec
            
            # Amplitude in uV -> Current in pA (Simulated scaling)
            current = pulse['amplitude_uv'] * 0.5 
            batch_current[start:end] += current
            
        # Apply for duration (simplified: just one big kick for now)
        # Ideally we'd distribute this over time steps
        self._physics.step(batch_current)

    def record(self, duration_sec: float) -> EmulatedRecording:
        """Record samples from the physics engine."""
        n_samples = int(duration_sec * self.sampling_rate)
        n_sim_steps = int(duration_sec * 1000 / self._physics.cfg.dt) # dt is in ms
        
        # We need to downsample physics (2kHz) to MEA (25kHz) - wait, physics is usually slower?
        # Physics dt = 0.5ms -> 2kHz. MEA is 25kHz.
        # We'll just interpolate or repeat samples. 
        # Actually for LFP (Local Field Potential), we sum synaptic currents.
        
        traces = np.zeros((self.n_electrodes, n_samples))
        
        # Run simulation
        samples_per_step = max(1, int(self.sampling_rate * (self._physics.cfg.dt / 1000)))
        current_sample = 0
        
        for _ in range(n_sim_steps):
            fired = self._physics.step()
            
            # Generate LFP for this step
            # LFP ~ sum of absolute synaptic currents in the cluster
            # Simplified: just use Voltage mean of cluster
            step_lfp = np.zeros(self.n_electrodes)
            
            for e in range(self.n_electrodes):
                start = e * self.sim_neurons_per_elec
                end = start + self.sim_neurons_per_elec
                # Get mean voltage of neurons near electrode
                if start < self._physics.cfg.n_neurons:
                    cluster_v = self._physics.v[start:end]
                    step_lfp[e] = np.mean(cluster_v)
            
            # Fill traces
            end_sample = min(current_sample + samples_per_step, n_samples)
            for s in range(current_sample, end_sample):
                traces[:, s] = step_lfp + np.random.normal(0, 2.0, self.n_electrodes) # Add instruction noise
            current_sample = end_sample
            
        return EmulatedRecording(samples=traces)

    def loop(self, tick_frames: int, n_ticks: int, callback: Callable):
        """
        Run real-time closed loop.
        tick_frames: number of 40us frames per tick.
        """
        dt_ms = self._physics.cfg.dt
        tick_ms = tick_frames * 40 / 1000 # 40us per frame
        steps_per_tick = int(tick_ms / dt_ms)
        
        for t in range(n_ticks):
            # 1. Run Physics for this Tick
            tick_spikes = []
            
            for _ in range(steps_per_tick):
                fired = self._physics.step()
                for n_idx in fired:
                    if n_idx in self.neuron_to_elec_map:
                        e_idx = self.neuron_to_elec_map[n_idx]
                        # Time in samples (approx)
                        time_sample = int(self._physics.t * 25) # simplified
                        tick_spikes.append(Spike(time=time_sample, electrode=e_idx))
            
            # 2. Invoke User Code
            elapsed_sec = t * tick_ms / 1000.0
            loop_tick = LoopTick(spikes=tick_spikes, elapsed_time=elapsed_sec)
            
            callback(loop_tick)

class StimPlan:
    def __init__(self):
        self.pulses = []
        
    def add_biphasic_pulse(self, electrode, amplitude_uv, duration_us, charge_balance):
        self.pulses.append({
            'electrode': electrode,
            'amplitude_uv': amplitude_uv,
            'duration_us': duration_us
        })
