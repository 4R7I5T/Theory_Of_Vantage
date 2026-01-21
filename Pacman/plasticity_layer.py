import numpy as np

class PlasticityLayer:
    def __init__(self, n_neurons, connectivity=0.2):
        self.n = n_neurons
        
        # -----------------------------------------------------------------
        # Synaptic Weights: W[post, pre]
        # -----------------------------------------------------------------
        # Initial random weights (small positive values)
        self.W = np.random.rand(n_neurons, n_neurons) * 5.0
        
        # Sparse connectivity mask
        # Neurons only connect to 20% of other neurons
        self.mask = np.random.rand(n_neurons, n_neurons) < connectivity
        
        # Enforce self-connections? Or avoid them? Usually avoid.
        np.fill_diagonal(self.mask, 0)
        
        self.W *= self.mask
        
        # -----------------------------------------------------------------
        # Plasticity State
        # -----------------------------------------------------------------
        # Eligibility trace: c[post, pre]
        # Tracks which synapses are eligible for change
        self.c = np.zeros((n_neurons, n_neurons))
        
        # Parameters
        self.tau_c = 10.0      # Eligibility trace decay (ms)
        self.learning_rate = 0.01 
        self.max_weight = 15.0 # Saturation
        self.dt = 0.5          # Simulation step size (ms) - match Physics
        
        # Dopamine
        self.dopamine_level = 0.0     # Instantaneous DA
        self.mean_dopamine = 0.0      # Baseline DA (for subtraction)
        self.tau_d = 20.0             # Dopamine baseline decay (ms)

    def step(self, fired_indices, dopamine_reward):
        """
        Advances the plasticity layer by one step.
        1. Decays traces.
        2. Updates traces based on spikes.
        3. Updates weights based on Dopamine-STDP.
        4. Calculates output synaptic currents for the NEXT step.
        """
        # ---------------------------------------------------------
        # 1. Decay Dynamics
        # ---------------------------------------------------------
        decay_c = np.exp(-self.dt / self.tau_c)
        self.c *= decay_c
        
        # Mean dopamine decay
        decay_d = np.exp(-self.dt / self.tau_d)
        self.mean_dopamine = (self.mean_dopamine * decay_d) + (self.dopamine_level * (1 - decay_d))
        
        # Update current instantaneous dopamine
        self.dopamine_level = dopamine_reward
        
        # ---------------------------------------------------------
        # 2. STDP Trace Update (On Pre-Synaptic Spike)
        # ---------------------------------------------------------
        # When a neuron fires, it acts as PRE-synaptic to all its targets.
        # We increment the trace for all synapses COMING FROM this neuron.
        # W[post, pre] -> We update column 'pre'
        if len(fired_indices) > 0:
            # Add 1.0 to the trace for all connections emanating from fired neurons
            # Use broadcasting or fancy indexing
            self.c[:, fired_indices] += 1.0
            
            # Re-apply mask to ensure no ghost traces on non-existent synapses
            self.c *= self.mask

        # ---------------------------------------------------------
        # 3. Weight Update (DA-STDP)
        # ---------------------------------------------------------
        # dw/dt = c * (DA - mean_DA)
        # We integrate this over dt. 
        # Using a simple Euler step: dw = c * (DA - mean_DA) * LR * dt
        
        delta_da = self.dopamine_level #- self.mean_dopamine # Simple prediction error
        
        # Optimization: continuous update is heavy (N^2). 
        # But for N=300, N^2=90,000 ops. Trivial for numpy.
        
        weight_change = self.c * delta_da * self.learning_rate * (self.dt / 10.0)
        
        self.W += weight_change
        
        # Clip weights
        np.clip(self.W, 0, self.max_weight, out=self.W)
        self.W *= self.mask # Enforce sparsity
        
        # ---------------------------------------------------------
        # 4. Compute Output Current
        # ---------------------------------------------------------
        # I_post = Sum(W[post, pre]) for all firing 'pre'
        # This current will be injected into the Post-synaptic neurons
        
        input_currents = np.zeros(self.n)
        
        if len(fired_indices) > 0:
            # Sum the columns corresponding to firing neurons
            # axis=1 sums across columns (pre) for each row (post)
            input_currents = np.sum(self.W[:, fired_indices], axis=1)
            
        return input_currents
