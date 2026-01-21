import numpy as np
from plasticity_layer import PlasticityLayer

def test_plasticity():
    print("Testing Plasticity Layer...")
    layer = PlasticityLayer(n_neurons=10, connectivity=1.0) # Fully connected
    
    # 1. Test Forward Pass
    spikes = [0, 1]
    current = layer.step(spikes, dopamine_reward=0.0)
    print(f"Initial Current: {current}")
    assert len(current) == 10
    
    # 2. Test Trace Accumulation
    # Neuron 0 fired. Trace for synapses originating from 0 should increase.
    # c[post, pre]. So column 0 should be > 0 (except diagonal).
    print(f"Trace (Col 0): {layer.c[:, 0]}")
    # Mask out diagonal for check
    trace_col = layer.c[:, 0].copy()
    trace_col[0] = 1.0 # Ignore diagonal
    assert np.all(trace_col > 0)
    
    # 3. Test Weight Update (Reward)
    # Inject Reward (+10)
    # Weights for active traces (Col 0) should INCREASE.
    old_w = layer.W.copy()
    layer.step([], dopamine_reward=10.0) # No new spikes, just reward
    
    delta_w = layer.W - old_w
    print(f"Weight Change (Col 0): {np.mean(delta_w[:, 0])}")
    print(f"Weight Change (Col 5): {np.mean(delta_w[:, 5])}")
    
    if np.mean(delta_w[:, 0]) > 0:
        print("PASS: Weights increased with reward.")
    else:
        print("FAIL: Weights did not increase.")
        
    # 4. Test Weight Update (Punishment)
    layer.step([5], dopamine_reward=0.0) # Neuron 5 fires
    old_w = layer.W.copy()
    layer.step([], dopamine_reward=-10.0) # Punishment
    
    delta_w = layer.W - old_w
    print(f"Weight Change (Col 5, Punishment): {np.mean(delta_w[:, 5])}")
    
    if np.mean(delta_w[:, 5]) < 0:
        print("PASS: Weights decreased with punishment.")
    else:
        print("FAIL: Weights did not decrease.")

if __name__ == "__main__":
    test_plasticity()
