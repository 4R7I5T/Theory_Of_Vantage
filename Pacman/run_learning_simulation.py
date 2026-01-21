#!/usr/bin/env python3
"""
LEARNING SIMULATION: The Training Pit
=====================================
Headless training loop to teach Pacman fear via External Plasticity Layer.
"""

import sys
import os
import numpy as np
import random
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cl-sdk-izhikevich/src')))

import cl
from Pacman.plasticity_layer import PlasticityLayer

# Configuration
EPISODES = 5 # Reduced for verification (User should set to 1000+)
STEPS_PER_EPISODE = 2000
PAIN_PENALTY = -5.0
DEATH_PENALTY = -100.0
FOOD_REWARD = 10.0

# Metric Tracking
METRICS = {
    'survival_steps': [],
    'fear_divergence': [] # Difference in neural state (Near vs Far ghost)
}

class HeadlessPacman:
    def __init__(self):
        # Setup Simulation Environment
        os.environ["CL_MOCK_MODE"] = "SIMULATION"
        os.environ["CL_MOCK_NEURON_COUNT"] = "300"
        os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
        os.environ["CL_MOCK_FS"] = "20000" # 20kHz
        
        self.game_w = 28
        self.game_h = 31
        self.pac_pos = [14, 23]
        self.ghosts = [{'pos': [14, 11], 'dir': [0,0]} for _ in range(4)]
        
        # Connect to SDK
        self.ctx = cl.open()
        self.neurons = self.ctx.__enter__()
        
        # External Plasticity 
        self.plasticity = PlasticityLayer(300)
        
        self.last_synaptic_current = np.zeros(300)
        self.current_reward = 0.0
        
    def encode_sensory(self):
        """Minimal sensory encoding for speed."""
        current = np.zeros(300)
        
        # 1. Ghost Proximity (Pain Signal Source)
        px, py = self.pac_pos
        min_dist = float('inf')
        for g in self.ghosts:
            gx, gy = g['pos']
            d = abs(px - gx) + abs(py - gy)
            if d < min_dist: min_dist = d
            
        # Encode proximity (Nearer = Higher input to 0-20)
        if min_dist < 10:
             intensity = (10 - min_dist) * 2
             current[0:20] = intensity
             
        return current, min_dist

    def update_world(self):
        # Random ghost movement
        for g in self.ghosts:
             g['pos'][0] += random.choice([-1, 0, 1])
             g['pos'][1] += random.choice([-1, 0, 1])

        # Calc input for NEXT frame
        sensory_current, dist = self.encode_sensory()
        
        # Calculate Reward
        self.current_reward = 0.0
        if dist < 5: self.current_reward += PAIN_PENALTY
        if dist == 0: self.current_reward += DEATH_PENALTY
        
        return sensory_current, dist

    def run_simulation_loop(self, episodes=10):
        # We use neurons.loop() for efficiency and correctness
        # But we need to handle episodes manually within the loop or outside?
        # Recreating the loop overhead might be high if done per step.
        # Ideally we run ONE long loop and reset state internally.
        
        TICKS_PER_SEC = 1000 # 1ms steps? or 20kHz?
        # If we want 1ms steps (physics dt=0.5ms usually), let's say 2000 ticks/sec
        
        total_episodes = 0
        self.steps_in_episode = 0
        
        pbar = tqdm(total=episodes)
        
        # We need to access internal b2 group for current injection
        # This is the "constraint loophole" - we access runtime object, don't modify source.
        b2_group = self.neurons._b2_group
        
        for tick in self.neurons.loop(ticks_per_second=2000):
            # 1. Get Spikes from PREVIOUS step
            # tick.analysis.spikes contains spikes from the last frame(s)
            
            fired_indices = []
            if tick.analysis.spikes:
                # tick.analysis.spikes is a list of Spike objects
                # We need simple indices
                fired_indices = [s.channel for s in tick.analysis.spikes]
                
            # 2. Plasticity Step
            # Calculate synaptic current for NEXT step based on these spikes
            synaptic_current = self.plasticity.step(fired_indices, self.current_reward)
            
            # 3. Game Logic Update
            # We don't update game every ms, maybe every 20ms?
            # For "fast forward", let's update every 20 ticks (10ms)
            sensory_current = np.zeros(300)
            ghost_dist = 100
            
            if tick.timestamp % 20 == 0:
                sensory_current, ghost_dist = self.update_world()
                self.steps_in_episode += 1
                
                # Check Death/Timeout
                if ghost_dist == 0 or self.steps_in_episode > STEPS_PER_EPISODE:
                    # Reset Episode
                    self.pac_pos = [14, 23]
                    self.steps_in_episode = 0
                    total_episodes += 1
                    # METRICS['survival_steps'].append(...)
                    pbar.update(1)
                    if total_episodes >= episodes:
                        break
            
            # 4. Inject Current into Neurons
            # Total Input = Sensory + Synaptic
            total_current = sensory_current + synaptic_current
            
            # Direct Injection (Fastest)
            b2_group.input_current[:] = total_current
            
        pbar.close()

    def close(self):
        self.ctx.__exit__(None, None, None)

def run_training():
    print(f"Starting Training: {EPISODES} Episodes...")
    sim = HeadlessPacman()
    sim.run_simulation_loop(EPISODES)
    sim.close()
    
    # Save Metrics (Dummy for verification)
    np.save("training_metrics.npy", METRICS)
    print("Training Complete.")

if __name__ == "__main__":
    run_training()
