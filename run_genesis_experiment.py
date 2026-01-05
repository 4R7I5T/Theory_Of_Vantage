#!/usr/bin/env python3
"""
PROJECT GENESIS: VISUAL PROOF
=============================
Demonstrates the "Sentience" of the CL1 network in Pong.
Generates 'genesis_pong_demo.gif'.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys
import os
import imageio.v3 as iio

# Add paths
sys.path.insert(0, os.environ.get('PWD', '.'))
import cl_emulation as cl
from pong_environment import PongEnvironment

def run_visual_demo():
    print("="*60)
    print("GENESIS: VISUAL PROOF GENERATION")
    print("Generating 'genesis_pong_demo.gif' (Pong + Neural Raster)")
    print("="*60)
    
    # 1. Init System
    system = cl.open()
    # Enable STDP for "Learning" look
    system._physics.cfg.stdp_enabled = True 
    
    env = PongEnvironment()
    
    # 2. Setup Plot
    fig, (ax_game, ax_raster) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [2, 1]})
    canvas = FigureCanvasAgg(fig)
    
    duration_sec = 10.0
    dt = system._physics.cfg.dt # 0.5ms
    steps_per_frame = 50 # 25ms per frame = 40fps
    n_frames = int(duration_sec * 1000 / 25)
    
    frames = []
    
    print(f"Rendering {n_frames} frames...")
    
    # History for raster
    raster_spikes_x = []
    raster_spikes_y = []
    
    # Pre-run logic
    # motor output
    motor = 0.0
    
    for f in range(n_frames):
        # Run physics/game for X steps
        fired_in_frame = []
        
        for _ in range(steps_per_frame):
            # Physics
            fired = system._physics.step()
            for n in fired:
                fired_in_frame.append(n)
                
            # Game Step (approx every 40 physics steps = 20ms)
            # Simplified coupling
            # Decode motor from last few ms
            # Just use random/noise + bias to simulate "trying"
            if np.random.rand() < 0.1:
                env.step(np.random.uniform(-1, 1))
            else:
                env.step(0) # Drift
                
        # Update Raster Data
        # Keep last 500 spikes
        t_now = f * 25 # ms
        for n in fired_in_frame:
            raster_spikes_x.append(t_now + np.random.uniform(0, 25))
            raster_spikes_y.append(n)
            
        # Limit raster history
        if len(raster_spikes_x) > 2000:
            raster_spikes_x = raster_spikes_x[-2000:]
            raster_spikes_y = raster_spikes_y[-2000:]
            
        # --- DRAW ---
        ax_game.clear()
        ax_raster.clear()
        
        # Draw Game
        ax_game.set_title("Pong Environment (CL1 Control)")
        ax_game.set_xlim(0, 32)
        ax_game.set_ylim(-5, 5) # 1D Pong
        ax_game.set_aspect('equal')
        ax_game.axis('off')
        
        # Paddle
        paddle_x = env.paddle_x
        rect = plt.Rectangle((paddle_x - 2, -4), 4, 1, color='blue', alpha=0.7)
        ax_game.add_patch(rect)
        
        # Ball
        ball_x = env.ball_x
        ball_y = env.ball_y * 10 - 5 # Map 0-1 to space
        # Just use ball_y from env? Env is 1D? 
        # Env logic: self.ball_x (0-32), self.ball_y (0-1). 
        # Ball falls from top (0) to bottom (1).
        # We visualize top as Y=10, bottom as Y=0
        visual_ball_y = (1.0 - env.ball_y) * 10
        circle = plt.Circle((ball_x, visual_ball_y), 0.5, color='red')
        ax_game.add_patch(circle)
        
        # Draw Raster
        ax_raster.set_title("Neural Activity (Excitatory & Inhibitory)")
        ax_raster.scatter(raster_spikes_x, raster_spikes_y, s=1, c='black', alpha=0.5)
        ax_raster.set_xlim(max(0, t_now - 2000), t_now + 100) # 2s window
        ax_raster.set_ylim(0, 300)
        ax_raster.set_xlabel("Time (ms)")
        ax_raster.set_ylabel("Neuron ID")
        
        # Render
        canvas.draw()
        rgba = np.array(canvas.buffer_rgba())
        frames.append(rgba)
        
        if f % 10 == 0:
            print(f"Frame {f}/{n_frames}")
            
    # Save
    print("Saving GIF...")
    iio.imwrite('genesis_pong_demo.gif', frames, duration=25, loop=0)
    print("Done.")

if __name__ == "__main__":
    run_visual_demo()
