"""
GHOST BRAIN: Feedforward "Zombie" Network
==========================================
A deliberately NON-conscious architecture for comparison.

Design choices to ensure LOW C-Score:
1. Strictly feedforward (no recurrence → low closure)
2. No STDP (no plasticity → no self-organization)  
3. No E/I balance (no oscillatory dynamics → low integration)
4. Simple input-output mapping (stimulus-response only)
"""

import numpy as np

class GhostBrain:
    """
    Feedforward MLP: Designed to be a 'Philosophical Zombie'
    
    Architecture:
    - Input: 4 neurons (distance to Pacman in each direction)
    - Hidden: 20 neurons (ReLU activation)
    - Output: 4 neurons (movement direction)
    
    Key: No recurrent connections, no plasticity, no internal state.
    This should produce LOW C-Score despite driving behavior.
    """
    
    def __init__(self, n_input=4, n_hidden=20, n_output=4, seed=None):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        rng = np.random.default_rng(seed)
        
        # Fixed random weights (no learning)
        self.W1 = rng.standard_normal((n_input, n_hidden)) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_output)) * 0.5
        self.b2 = np.zeros(n_output)
        
        # Trace buffer for PA analysis
        self.trace_buffer = []
        self.max_trace_length = 500  # 500ms at 1kHz
        
    def forward(self, sensory_input):
        """
        Pure feedforward pass.
        Returns action and internal activations for PA analysis.
        """
        x = np.asarray(sensory_input).flatten()[:self.n_input]
        if len(x) < self.n_input:
            x = np.pad(x, (0, self.n_input - len(x)))
            
        # Layer 1: ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)
        
        # Layer 2: Softmax (action probabilities)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Record hidden activations for trace
        self._record_trace(h)
        
        # Winner-takes-all action
        action = int(np.argmax(probs))
        
        return action, h
    
    def _record_trace(self, activations):
        """Store activations for PA analysis."""
        self.trace_buffer.append(activations.copy())
        if len(self.trace_buffer) > self.max_trace_length:
            self.trace_buffer.pop(0)
            
    def get_trace_matrix(self):
        """
        Return trace as (T, N) matrix for PA analysis.
        """
        if len(self.trace_buffer) < 10:
            return None
        return np.array(self.trace_buffer)
    
    def reset_trace(self):
        """Clear trace buffer."""
        self.trace_buffer = []
        
    def get_sensory_input(self, ghost_pos, pacman_pos, maze_width, maze_height):
        """
        Generate sensory input vector: normalized distance to Pacman in 4 directions.
        """
        gx, gy = ghost_pos
        px, py = pacman_pos
        
        dx = px - gx
        dy = py - gy
        
        # Normalize by maze size
        return np.array([
            max(0, dx) / maze_width,   # Pacman to the right
            max(0, -dx) / maze_width,  # Pacman to the left
            max(0, -dy) / maze_height, # Pacman above
            max(0, dy) / maze_height,  # Pacman below
        ])


class GhostBrainEnsemble:
    """
    Manages 4 independent Ghost brains (Blinky, Pinky, Inky, Clyde).
    Each has identical architecture but different random weights.
    """
    
    def __init__(self):
        self.brains = {
            'blinky': GhostBrain(seed=1),
            'pinky': GhostBrain(seed=2),
            'inky': GhostBrain(seed=3),
            'clyde': GhostBrain(seed=4),
        }
        
    def step(self, ghost_name, sensory_input):
        """
        Run one forward pass for a specific ghost.
        Returns action (0=Right, 1=Left, 2=Up, 3=Down).
        """
        action, _ = self.brains[ghost_name].forward(sensory_input)
        return action
    
    def get_all_traces(self):
        """Return dict of trace matrices for PA analysis."""
        return {
            name: brain.get_trace_matrix() 
            for name, brain in self.brains.items()
        }
    
    def reset_all(self):
        """Reset all trace buffers."""
        for brain in self.brains.values():
            brain.reset_trace()
