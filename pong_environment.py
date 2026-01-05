#!/usr/bin/env python3
"""
DISHBRAIN PONG ENVIRONMENT
==========================
Recreates the exact experimental setup from Kagan et al. (2022).
- Pure Python implementation
- Physics: 1D Pong
- Interface: Compatible with cl_emulation
"""

import numpy as np

class PongEnvironment:
    def __init__(self):
        # Arena
        self.width = 1.0
        self.height = 1.0
        self.paddle_width = 0.2
        
        # State
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_dx = 0.015
        self.ball_dy = 0.01
        self.paddle_x = 0.5
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.total_ticks = 0
        self.long_rally = 0
        self.current_rally = 0
        
    def step(self, paddle_motor_signal: float) -> dict:
        """
        Advance game state.
        paddle_motor_signal: -1.0 (left) to 1.0 (right)
        """
        # 1. Move Paddle
        speed = 0.05
        self.paddle_x += paddle_motor_signal * speed
        self.paddle_x = np.clip(self.paddle_x, 0, 1.0)
        
        # 2. Move Ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # 3. Collision Logic
        event = "none"
        
        # Walls (Top, Left, Right)
        if self.ball_y <= 0:
            self.ball_dy *= -1
            self.ball_y = abs(self.ball_y)
        
        if self.ball_x <= 0:
            self.ball_dx *= -1
            self.ball_x = abs(self.ball_x)
            
        if self.ball_x >= 1.0:
            self.ball_dx *= -1
            self.ball_x = 1.0 - (self.ball_x - 1.0)
            
        # Paddle Interaction (Bottom)
        if self.ball_y >= 1.0:
            # Check overlap
            hit_range_min = self.paddle_x - self.paddle_width/2
            hit_range_max = self.paddle_x + self.paddle_width/2
            
            if hit_range_min <= self.ball_x <= hit_range_max:
                # HIT!
                self.ball_dy *= -1
                self.ball_y = 1.0 - (self.ball_y - 1.0)
                self.hits += 1
                self.current_rally += 1
                self.long_rally = max(self.long_rally, self.current_rally)
                event = "hit"
            else:
                # MISS!
                # Reset ball
                self.ball_x = 0.5
                self.ball_y = 0.5
                self.ball_dx = np.random.choice([-1, 1]) * 0.015
                self.ball_dy = 0.01
                self.misses += 1
                self.current_rally = 0
                event = "miss"
                
        self.total_ticks += 1
        
        return {
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "paddle_x": self.paddle_x,
            "event": event
        }
        
    def get_sensory_stimulation(self) -> list:
        """
        Map ball position to electrode dict.
        Returns list of electrode indices to stimulate.
        """
        # Simple map: discrete zones
        # 0-7: Left wall
        # 8-15: Right wall
        # 16-23: Ball close warning
        stim_elecs = []
        
        # Spatial coding (Position)
        # Map x (0-1) to electrodes 0-20
        pos_elec = int(self.ball_x * 20)
        stim_elecs.append(pos_elec)
        
        # Critical distance feedback (Frequency scaling handled by rate of calls)
        if self.ball_y > 0.8:
            stim_elecs.append(50) # Warning channel
            
        return stim_elecs
