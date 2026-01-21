#!/usr/bin/env python3
"""
NEURAL PACMAN: Consciousness vs Zombies Demo
=============================================
Combines classic mechanics (ghost house, release timers) 
with modern Sprite structure and Neural Network brains.

- PACMAN: Conscious Autonomous Agent (Neural Decision Integration)
- GHOSTS: Feedforward MLP Zombies
  - State Machine: In_House -> Exiting -> Scatter/Chase -> Frightened

Press ESC to quit. Press R to restart.
"""

import pygame
import random
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cl_emulation.ghost_brain import GhostBrainEnsemble
import cl_emulation as cl


class PacmanBrain:
    """
    Conscious Agent Brain - Izhikevich Recurrent Neural Network
    ===========================================================
    Uses 300 spiking neurons with STDP plasticity for decision making.
    This creates temporal integration - the hallmark of conscious processing.
    """
    
    def __init__(self):
        self.system = cl.open()
        self.physics = self.system._physics  # 300 neurons
        
        # Neuron population assignments (for encoding/decoding)
        # Input: neurons 0-59 (sensory)
        # Processing: neurons 60-239 (recurrent core)
        # Output: neurons 240-299 (motor)
        self.n_input = 60
        self.n_output = 60
        
        # Output neuron groups for each direction
        # RIGHT=240-254, LEFT=255-269, UP=270-284, DOWN=285-299
        self.output_ranges = {
            'right': (240, 255),
            'left': (255, 270),
            'up': (270, 285),
            'down': (285, 300)
        }
        
        # Metrics tracking
        self.total_spikes = 0
        self.decision_spikes = 0
        self.last_decision_time = 0
        self.last_latency = 0.0  # Decision latency in ms
        
    def encode_sensory(self, pac_pos, ghosts, pellets, valid_dirs, frightened):
        """Convert game state to input currents for sensory neurons."""
        current = np.zeros(self.physics.cfg.n_neurons)
        
        px, py = pac_pos
        
        # 1. Danger encoding (neurons 0-19): Ghost proximity
        for i, ghost in enumerate(ghosts):
            if i >= 4:
                break
            gx, gy = ghost.rect.centerx, ghost.rect.centery
            dist = abs(px - gx) + abs(py - gy)
            
            # Closer = more activation (inverted)
            if dist < 200:
                intensity = (200 - dist) / 200 * 15
                if frightened:
                    intensity *= -0.5  # Inverted when we can eat them
                current[i * 5:(i + 1) * 5] = intensity
        
        # 2. Direction encoding (neurons 20-39): Valid moves
        dir_map = {'right': 0, 'left': 1, 'up': 2, 'down': 3}
        for dir_name, idx in dir_map.items():
            # Check if direction is in valid moves
            speed = 2
            dir_vec = {'right': (speed, 0), 'left': (-speed, 0), 
                       'up': (0, -speed), 'down': (0, speed)}[dir_name]
            if dir_vec in valid_dirs:
                current[20 + idx * 5:20 + (idx + 1) * 5] = 8.0
                
        # 3. Reward encoding (neurons 40-59): Nearest pellet direction
        if pellets:
            # Find nearest pellet
            nearest = None
            min_dist = float('inf')
            for p in pellets:
                d = abs(px - p.rect.x) + abs(py - p.rect.y)
                if d < min_dist:
                    min_dist = d
                    nearest = p
                    
            if nearest:
                dx = nearest.rect.x - px
                dy = nearest.rect.y - py
                
                # Encode direction to pellet
                if dx > 0:
                    current[40:45] = 10.0  # Right
                elif dx < 0:
                    current[45:50] = 10.0  # Left
                if dy < 0:
                    current[50:55] = 10.0  # Up
                elif dy > 0:
                    current[55:60] = 10.0  # Down
                    
        return current
    
    def decode_action(self):
        """Decode output neuron activity into movement direction."""
        spike_counts = {}
        
        for dir_name, (start, end) in self.output_ranges.items():
            # Count recent spikes in each output group
            # Using membrane potential as proxy for activity
            activity = np.sum(self.physics.v[start:end] > -50)
            spike_counts[dir_name] = activity
            
        # Winner-take-all
        if sum(spike_counts.values()) == 0:
            return None
            
        best_dir = max(spike_counts, key=spike_counts.get)
        
        speed = 2
        dir_vec = {'right': (speed, 0), 'left': (-speed, 0), 
                   'up': (0, -speed), 'down': (0, speed)}
        
        return dir_vec.get(best_dir)
    
    def decide(self, pac_pos, ghosts, pellets, valid_dirs, frightened):
        """
        Main decision function - runs neural integration.
        Returns (direction_tuple, spike_count)
        """
        import time
        start_time = time.perf_counter()
        
        # Encode sensory input
        input_current = self.encode_sensory(pac_pos, ghosts, pellets, valid_dirs, frightened)
        
        # Run multiple physics steps for temporal integration
        # This is the KEY difference from feedforward zombies
        integration_steps = 5
        total_spikes = 0
        
        for _ in range(integration_steps):
            fired = self.physics.step(input_current)
            total_spikes += len(fired)
            
        self.decision_spikes = total_spikes
        self.total_spikes += total_spikes
        
        # Track latency (in ms)
        self.last_latency = (time.perf_counter() - start_time) * 1000
        
        # Decode output
        action = self.decode_action()
        
        # Validate against valid moves
        if action and action in valid_dirs:
            return action, total_spikes
        elif valid_dirs:
            return random.choice(valid_dirs), total_spikes
        else:
            return (0, 0), total_spikes

# Settings
CHAR_SIZE = 20  # Slightly smaller for classic 28x31 board
PLAYER_SPEED = 2
GHOST_SPEED = 2
FPS = 60
FRIGHTENED_TIME = 600

# Colors
BLACK = (0, 0, 0)
MIDNIGHT_BLUE = (25, 25, 166)
BLUE = (33, 33, 222)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 184, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 184, 82)
BLUE_FRIGHT = (33, 33, 255)

# Classic 28x31 Maze
# 1=Wall, 0=Dot, 3=Power, 8=Gate, 9=GhostHouse, 2=Empty
MAP = [
    "1111111111111111111111111111",
    "1000000000000110000000000001",
    "1011110111110110111110111101",
    "1311110111110110111110111131",
    "1011110111110110111110111101",
    "1000000000000000000000000001",
    "1011110110111111110110111101",
    "1011110110111111110110111101",
    "1000000110000110000110000001",
    "1111110111112112111110111111",
    "2222210111112112111110122222",
    "2222210112222222222110122222",
    "2222210112111881112110122222",
    "1111110112199999912110111111",
    "2222220002199999912000222222",
    "1111110112111111112110111111",
    "2222210112222222222110122222",
    "2222210112111111112110122222",
    "2222210112111111112110122222",
    "1111110110000000000110111111",
    "1000000000000110000000000001",
    "1011110111110110111110111101",
    "1011110111110110111110111101",
    "1300110000000000000000110031",
    "1110110110111111110110110111",
    "1110110110111111110110110111",
    "1000000110000110000110000001",
    "1011111111110110111111111101",
    "1011111111110110111111111101",
    "1000000000000000000000000001",
    "1111111111111111111111111111",
]

# Coordinates (in tiles)
PACMAN_START = (14, 23)
# Ghost House Exit point (above gate)
HOUSE_EXIT_TARGET = (14, 11)  # Target tile outside gate

# (Start Tile X, Start Tile Y, Release Time Frames)
GHOST_CONFIG = {
    'blinky': {'start': (14, 11), 'release': 0, 'color': RED},    # Starts outside
    'pinky':  {'start': (14, 14), 'release': 60, 'color': PINK},  # Center
    'inky':   {'start': (12, 14), 'release': 200, 'color': CYAN}, # Left
    'clyde':  {'start': (16, 14), 'release': 400, 'color': ORANGE}# Right
}


class Cell(pygame.sprite.Sprite):
    def __init__(self, x, y, char):
        super().__init__()
        self.image = pygame.Surface((CHAR_SIZE, CHAR_SIZE))
        if char == '1':
            self.image.fill(MIDNIGHT_BLUE)
            # Add detail loop for board look
            pygame.draw.rect(self.image, (0,0,0), (4,4,CHAR_SIZE-8,CHAR_SIZE-8))
        elif char == '8':  # Gate
            self.image.fill((0, 0, 0))
            pygame.draw.line(self.image, PINK, (0, CHAR_SIZE//2), (CHAR_SIZE, CHAR_SIZE//2), 3)
        self.rect = self.image.get_rect(topleft=(x*CHAR_SIZE, y*CHAR_SIZE))


class Berry(pygame.sprite.Sprite):
    def __init__(self, x, y, is_power=False):
        super().__init__()
        self.is_power = is_power
        self.image = pygame.Surface((CHAR_SIZE, CHAR_SIZE), pygame.SRCALPHA)
        center = (CHAR_SIZE // 2, CHAR_SIZE // 2)
        if is_power:
            pygame.draw.circle(self.image, WHITE, center, 7)
        else:
            pygame.draw.circle(self.image, (255, 184, 174), center, 2)
        self.rect = self.image.get_rect(topleft=(x*CHAR_SIZE, y*CHAR_SIZE))


class Ghost(pygame.sprite.Sprite):
    def __init__(self, name, config, brain):
        super().__init__()
        self.name = name
        self.start_pos = (config['start'][0]*CHAR_SIZE, config['start'][1]*CHAR_SIZE)
        self.base_color = config['color']
        self.brain = brain
        self.release_time = config['release']
        self.speed = GHOST_SPEED
        
        # Init visual
        self.image = pygame.Surface((CHAR_SIZE, CHAR_SIZE), pygame.SRCALPHA)
        self.rect = self.image.get_rect(topleft=self.start_pos)
        
        # Logic init
        self.move_dir = (0, 0)
        self.keys = ['left', 'right', 'up', 'down']
        self.dirs = {
            'left': (-self.speed, 0), 'right': (self.speed, 0),
            'up': (0, -self.speed), 'down': (0, self.speed)
        }
        self.reset()
        
    def reset(self):
        self.rect.topleft = self.start_pos
        self.timer = 0
        self.frightened = False
        self.dead = False
        
        # State machine
        if self.release_time == 0:
            self.state = 'active'
            self.move_dir = random.choice(['left', 'right'])
        else:
            self.state = 'in_house'
            self.move_dir = 'up' if random.random() < 0.5 else 'down'
            
        self.update_image()
        
    def update_image(self):
        self.image.fill((0,0,0,0))
        color = BLUE_FRIGHT if self.frightened else self.base_color
        if self.dead: color = (100, 100, 100) # Eyes only ideally, but grey for now
        
        cx, cy = CHAR_SIZE//2, CHAR_SIZE//2
        r = CHAR_SIZE//2 - 2
        
        pygame.draw.circle(self.image, color, (cx, cy-2), r)
        pygame.draw.rect(self.image, color, (cx-r, cy, r*2, r))
        
        # Eyes
        if not self.frightened or self.dead:
            pygame.draw.circle(self.image, WHITE, (cx-4, cy-4), 4)
            pygame.draw.circle(self.image, WHITE, (cx+4, cy-4), 4)
            
            # Pupil direction
            pdx, pdy = 0, 0
            if self.move_dir == 'left': pdx = -2
            if self.move_dir == 'right': pdx = 2
            if self.move_dir == 'up': pdy = -2
            if self.move_dir == 'down': pdy = 2
            
            pygame.draw.circle(self.image, BLUE, (cx-4+pdx, cy-4+pdy), 2)
            pygame.draw.circle(self.image, BLUE, (cx+4+pdx, cy-4+pdy), 2)
        else:
            # Scared face
            pygame.draw.line(self.image, (255,200,200), (cx-5, cy), (cx-2, cy+2), 2)
            pygame.draw.line(self.image, (255,200,200), (cx-2, cy+2), (cx+1, cy), 2)
            
    def get_valid_moves(self, walls, allow_reverse=False):
        moves = []
        # Tunnel Constraint: If off-screen, horizontal only
        is_in_tunnel = self.rect.left < 0 or self.rect.right > len(MAP[0]) * CHAR_SIZE

        for d in self.keys:
            if is_in_tunnel and d in ['up', 'down']:
                continue

            if not allow_reverse:
                # Basic reverse check
                if (d == 'left' and self.move_dir == 'right') or \
                   (d == 'right' and self.move_dir == 'left') or \
                   (d == 'up' and self.move_dir == 'down') or \
                   (d == 'down' and self.move_dir == 'up'):
                    continue
            
            # Prediction check
            vx, vy = self.dirs[d]
            test_rect = self.rect.move(vx, vy)
            if test_rect.collidelist(walls) == -1:
                moves.append(d)
        return moves

    def update(self, walls, pac_rect):
        self.timer += 1
        self.last_activations = None # Reset activations
        
        # 1. HOUSE LOGIC
        if self.state == 'in_house':
            # Bounce
            if self.timer % 20 < 10:
                self.rect.y -= 1
            else:
                self.rect.y += 1
                
            if self.timer > self.release_time:
                self.state = 'exiting'
                # Center horizontally first
                self.target_x = 14 * CHAR_SIZE
                self.target_y = 11 * CHAR_SIZE # Outside gate
                
            self.update_image()
            return

        # 2. EXITING LOGIC
        if self.state == 'exiting':
            # Simple lerp-like movement to target
            if self.rect.x < self.target_x: self.rect.x += 1
            elif self.rect.x > self.target_x: self.rect.x -= 1
            elif self.rect.y > self.target_y: self.rect.y -= 1
            else:
                self.state = 'active'
                self.move_dir = random.choice(['left', 'right'])
            self.update_image()
            return
            
        # 3. ACTIVE LOGIC
        # Align to grid
        on_grid = (self.rect.x % CHAR_SIZE == 0) and (self.rect.y % CHAR_SIZE == 0)
        
        if on_grid:
            valid = self.get_valid_moves(walls)
            if not valid:
                # Dead end or stuck, allow reverse
                valid = self.get_valid_moves(walls, allow_reverse=True)
            
            if valid:
                if self.frightened and not self.dead:
                    # Run away
                    # Pick move that maximizes distance
                    best_move = max(valid, key=lambda m: 
                        abs((self.rect.x + self.dirs[m][0]*10) - pac_rect.x) + 
                        abs((self.rect.y + self.dirs[m][1]*10) - pac_rect.y))
                    
                    if random.random() < 0.8: # Erratic
                        self.move_dir = best_move
                    else:
                        self.move_dir = random.choice(valid)
                else:
                    # Neural Chase
                    # Convert to grid coords
                    gx, gy = self.rect.x // CHAR_SIZE, self.rect.y // CHAR_SIZE
                    px, py = pac_rect.x // CHAR_SIZE, pac_rect.y // CHAR_SIZE
                    w, h = len(MAP[0]), len(MAP)
                    
                    sensory = self.brain.get_sensory_input((gx, gy), (px, py), w, h)
                    action, activations = self.brain.forward(sensory)
                    self.last_activations = activations # Store for visualization
                    
                    # 0=R, 1=L, 2=U, 3=D
                    net_map = {0:'right', 1:'left', 2:'up', 3:'down'}
                    choice = net_map.get(action, 'right')
                    
                    if choice in valid and random.random() < 0.65:
                        self.move_dir = choice
                    else:
                        self.move_dir = random.choice(valid)
        
        # Apply movement
        vx, vy = self.dirs.get(self.move_dir, (0,0))
        
        # Collision check before move (pixel perfect)
        test_rect = self.rect.move(vx, vy)
        if test_rect.collidelist(walls) == -1:
            self.rect = test_rect
        else:
            # Re-align if hit wall
            self.rect.x = (self.rect.centerx // CHAR_SIZE) * CHAR_SIZE
            self.rect.y = (self.rect.centery // CHAR_SIZE) * CHAR_SIZE
        
        # Tunnel
        if self.rect.right <= 0: self.rect.x = len(MAP[0]) * CHAR_SIZE
        if self.rect.left >= len(MAP[0]) * CHAR_SIZE: self.rect.x = 0
        
        self.update_image()


class Pacman(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((CHAR_SIZE, CHAR_SIZE), pygame.SRCALPHA)
        self.rect = self.image.get_rect(topleft=(PACMAN_START[0]*CHAR_SIZE, PACMAN_START[1]*CHAR_SIZE))
        
        self.speed = PLAYER_SPEED
        self.move_dir = (0, 0)
        self.next_dir = (0, 0)
        
        self.lives = 3
        self.score = 0
        self.frame = 0
        
        # CONSCIOUS BRAIN - Izhikevich Network
        self.brain = PacmanBrain()
        self.last_spikes = 0  # For display
        
        self.reset_pos()
        
    def reset_pos(self):
        self.rect.topleft = (PACMAN_START[0]*CHAR_SIZE, PACMAN_START[1]*CHAR_SIZE)
        self.move_dir = (0, 0)
        self.next_dir = (0, 0)
        
    def get_valid_moves(self, walls):
        moves = []
        is_in_tunnel = self.rect.left < 0 or self.rect.right > len(MAP[0]) * CHAR_SIZE
        
        dirs = {'left':(-self.speed, 0), 'right':(self.speed, 0), 
                'up':(0, -self.speed), 'down':(0, self.speed)}
        
        for name, d in dirs.items():
            if is_in_tunnel and name in ['up', 'down']:
                continue
            if self.rect.move(d).collidelist(walls) == -1:
                moves.append(d)
        return moves
        
    def update(self, walls, ghosts, pellets, frightened=False):
        self.frame += 1
        
        # Neural Decision - replaces heuristic AI
        # Make decision every 10 frames or if stopped
        if self.frame % 10 == 0 or self.move_dir == (0, 0):
            valid = self.get_valid_moves(walls)
            if valid:
                # Use Izhikevich network brain
                pac_pos = (self.rect.centerx, self.rect.centery)
                action, spikes = self.brain.decide(
                    pac_pos, ghosts, pellets, valid, frightened
                )
                self.last_spikes = spikes
                self.next_dir = action
        
        # Attempt to turn (on grid alignment)
        if self.next_dir != (0, 0):
            if self.rect.x % CHAR_SIZE == 0 and self.rect.y % CHAR_SIZE == 0:
                if self.rect.move(self.next_dir).collidelist(walls) == -1:
                    self.move_dir = self.next_dir
                    self.next_dir = (0, 0)
            # Allow immediate reverse
            if self.next_dir[0] == -self.move_dir[0] and self.next_dir[1] == -self.move_dir[1]:
                self.move_dir = self.next_dir
                self.next_dir = (0, 0)

        # Move
        if self.rect.move(self.move_dir).collidelist(walls) == -1:
            self.rect.move_ip(self.move_dir)
        else:
            self.rect.x = (self.rect.x // CHAR_SIZE) * CHAR_SIZE
            self.rect.y = (self.rect.y // CHAR_SIZE) * CHAR_SIZE

        # Tunnel wrap
        if self.rect.right <= 0: 
            self.rect.x = len(MAP[0]) * CHAR_SIZE
        if self.rect.left >= len(MAP[0]) * CHAR_SIZE: 
            self.rect.x = 0
        
        # Draw
        self._draw()
        
    def _draw(self):
        self.image.fill((0, 0, 0, 0))
        pygame.draw.circle(self.image, YELLOW, (CHAR_SIZE//2, CHAR_SIZE//2), CHAR_SIZE//2-2)
        
        # Mouth animation
        if self.frame % 10 < 5 and self.move_dir != (0, 0):
            angle = 0
            if self.move_dir[0] < 0: angle = 180
            elif self.move_dir[1] < 0: angle = 90
            elif self.move_dir[1] > 0: angle = 270
            
            pts = [(CHAR_SIZE//2, CHAR_SIZE//2)]
            for a in [angle - 40, angle + 40]:
                r = np.radians(a)
                pts.append((CHAR_SIZE//2 + int(np.cos(r)*12), CHAR_SIZE//2 - int(np.sin(r)*12)))
            pygame.draw.polygon(self.image, BLACK, pts)


class Game:
    def __init__(self):
        pygame.init()
        
        # Game area + visualization panel
        # Enlarged panel to fit 2 brains side by side or stacked
        self.game_w = len(MAP[0]) * CHAR_SIZE
        self.panel_w = 400  # Enlarged from 220 to 400
        self.screen_w = self.game_w + self.panel_w
        self.screen_h = len(MAP) * CHAR_SIZE + 50
        
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Neural Pacman: Consciousness vs Zombies Experiment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 24)
        
        # Sprite Groups
        self.walls = pygame.sprite.Group()
        self.pellets = pygame.sprite.Group()
        self.ghosts = pygame.sprite.Group()
        self.pacman = Pacman()
        self.all_sprites = pygame.sprite.Group()
        
        self.ghost_brains = GhostBrainEnsemble()
        self.load_map()
        
        self.frightened_timer = 0
        self.game_over = False
        
        # Visualization data
        self.spike_history = []  # List of spike counts per frame
        self.ghost_spike_history = []
        self.max_history = 100  # Rolling window
        
    def load_map(self):
        self.walls.empty()
        self.pellets.empty()
        self.ghosts.empty()
        
        # Determine actual walls rects strictly
        wall_rects = []
        
        for y, row in enumerate(MAP):
            for x, char in enumerate(row):
                if char in ['1', '9']: # Wall or ghost house
                    w = Cell(x, y, char)
                    self.walls.add(w)
                    if char == '1':
                        wall_rects.append(w.rect)
                elif char == '8': # Gate
                    # Visual only, not physical for ghosts (handled by logic)
                    # For pacman, treated as wall
                    w = Cell(x, y, char)
                    self.walls.add(w)
                    wall_rects.append(w.rect)
                elif char == '0':
                    self.pellets.add(Berry(x, y, False))
                elif char == '3':
                    self.pellets.add(Berry(x, y, True))
                    
        # Create ghosts
        for name, cfg in GHOST_CONFIG.items():
            g = Ghost(name, cfg, self.ghost_brains.brains[name])
            self.ghosts.add(g)
            
        self.pacman.reset_pos()
        self.wall_list = wall_rects # For collision
        
    def run(self):
        running = True
        frame_count = 0
        capture_frames = True  # Capture first 120 frames for GIF
        import os
        capture_dir = "/tmp/pacman_frames"
        os.makedirs(capture_dir, exist_ok=True)
        
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    if event.key == pygame.K_r: 
                        self.load_map()
                        self.game_over = False
                        self.pacman.lives = 3
                        self.pacman.score = 0
                        
            if not self.game_over:
                self.update()
                
            self.draw()
            
            # Capture frames for GIF (every 4th frame for 120 captures = ~8 seconds)
            if capture_frames and frame_count < 480:
                if frame_count % 4 == 0:
                    pygame.image.save(self.screen, f"{capture_dir}/frame_{frame_count//4:04d}.png")
                frame_count += 1
            elif capture_frames and frame_count >= 480:
                capture_frames = False
                print(f"[INFO] Captured 120 frames to {capture_dir}/")
                
            self.clock.tick(FPS)
        pygame.quit()
        
    def update(self):
        # Update Timer
        if self.frightened_timer > 0:
            self.frightened_timer -= 1
            if self.frightened_timer == 0:
                for g in self.ghosts: g.frightened = False
                
        # Update Sprites
        frightened = (self.frightened_timer > 0)
        self.pacman.update(self.wall_list, self.ghosts, self.pellets, frightened)
        for g in self.ghosts:
            # Ghosts can pass through gate (Cell 8) if exiting
            # So pass full wall list, but logic handles it
            g.frightened = (self.frightened_timer > 0)
            g.update(self.wall_list, self.pacman.rect)
            
        # Pellet Collision
        hit_list = pygame.sprite.spritecollide(self.pacman, self.pellets, True)
        for p in hit_list:
            if p.is_power:
                self.pacman.score += 50
                self.frightened_timer = FRIGHTENED_TIME
                for g in self.ghosts: g.frightened = True
            else:
                self.pacman.score += 10
                
        # Ghost Collision
        hit_ghosts = pygame.sprite.spritecollide(self.pacman, self.ghosts, False)
        for g in hit_ghosts:
            if g.frightened:
                g.dead = True # Simple reset for now
                g.reset()
                self.pacman.score += 200
            else:
                self.pacman.lives -= 1
                if self.pacman.lives == 0:
                    self.game_over = True
                else:
                    self.pacman.reset_pos()
                    for gh in self.ghosts: gh.reset()
                break
                
        # Level Complete
        if len(self.pellets) == 0:
            self.load_map() 
                
    def draw(self):
        self.screen.fill(BLACK)
        self.walls.draw(self.screen)
        self.pellets.draw(self.screen)
        self.ghosts.draw(self.screen)
        self.screen.blit(self.pacman.image, self.pacman.rect)
        
        # Dashboard
        base_y = self.screen_h - 40
        score_t = self.font.render(f"SCORE: {self.pacman.score}", True, WHITE)
        self.screen.blit(score_t, (10, base_y))
        
        for i in range(self.pacman.lives):
            pygame.draw.circle(self.screen, YELLOW, (150 + i*22, base_y+8), 8)
            
        if self.game_over:
            end_t = self.title_font.render("GAME OVER - PRESS R", True, RED)
            self.screen.blit(end_t, (self.game_w//2 - 90, self.screen_h//2))
        
        # Draw Neural Visualization Panel
        self.draw_neural_panel()
              
        pygame.display.flip()
        
    def draw_neural_panel(self):
        """Draw the neural activity visualization panel on the right side."""
        panel_x = self.game_w + 5
        panel_w = (self.panel_w - 10) // 2 
        
        # Panel background
        pygame.draw.rect(self.screen, (20, 20, 30), (self.game_w, 0, self.panel_w, self.screen_h))
        pygame.draw.line(self.screen, (60, 60, 80), (self.game_w, 0), (self.game_w, self.screen_h), 2)
        
        # Title
        title = self.title_font.render("NEURAL ARCHITECTURE TEST", True, (200, 200, 255))
        self.screen.blit(title, (self.game_w + 20, 10))
        
        # --- LEFT SUB-PANEL: PACMAN (Recurrent) ---
        col1_x = self.game_w + 10
        y = 50
        
        label = self.font.render("PACMAN (CONSCIOUS)", True, YELLOW)
        self.screen.blit(label, (col1_x, y))
        y += 20
        sub_label = self.font.render("Recurrent SNN (Brian2)", True, (150, 150, 100))
        self.screen.blit(sub_label, (col1_x, y))
        y += 25
        
        # Spikes/Metrics
        spikes = self.pacman.last_spikes
        spike_text = self.font.render(f"Spikes/Step: {spikes}", True, WHITE)
        self.screen.blit(spike_text, (col1_x, y)); y += 20
        
        latency = getattr(self.pacman.brain, 'last_latency', 0)
        lat_text = self.font.render(f"Integration: {latency:.1f}ms", True, WHITE)
        self.screen.blit(lat_text, (col1_x, y)); y += 20

        # Spike Rate Bar
        bar_w = 160
        ratio = min(1.0, spikes / 50.0)
        pygame.draw.rect(self.screen, (40, 40, 60), (col1_x, y, bar_w, 10))
        pygame.draw.rect(self.screen, (255, 255, 100), (col1_x, y, bar_w * ratio, 10))
        y += 20
        
        # Raster Plot (Temporal Integration)
        raster_h = 200
        pygame.draw.rect(self.screen, (10, 10, 15), (col1_x, y, bar_w, raster_h))
        
        # Draw membrane potentials
        physics = self.pacman.brain.physics
        n_show = 60
        for i in range(n_show):
            neuron_idx = i * 5
            if neuron_idx < len(physics.v):
                v = physics.v[neuron_idx]
                # Map -80..30 to 0..raster_h
                ny = int(((-v + 30) / 110) * raster_h)
                ny = max(0, min(raster_h - 2, ny))
                
                color = (100, 255, 100) if v > -50 else (30, 80, 30)
                if neuron_idx >= physics.cfg.n_exc:
                    color = (255, 100, 100) if v > -50 else (80, 30, 30)
                
                nx = col1_x + int(i * bar_w / n_show)
                pygame.draw.circle(self.screen, color, (nx, y + ny), 2)
        y += raster_h + 10
        
        # --- RIGHT SUB-PANEL: GHOST (Feedforward) ---
        col2_x = self.game_w + 200
        y = 50
        
        # Find active ghost (closest to pacman)
        pac_center = self.pacman.rect.center
        active_ghost = min(self.ghosts, key=lambda g: 
            abs(g.rect.centerx - pac_center[0]) + abs(g.rect.centery - pac_center[1]))
        
        label_g = self.font.render(f"{active_ghost.name.upper()} (ZOMBIE)", True, active_ghost.base_color)
        self.screen.blit(label_g, (col2_x, y))
        y += 20
        sub_label_g = self.font.render("Feedforward MLP", True, (150, 150, 150))
        self.screen.blit(sub_label_g, (col2_x, y))
        y += 25
        
        # Metrics
        act_text = self.font.render("State: REFLEX", True, WHITE)
        self.screen.blit(act_text, (col2_x, y)); y += 20
        
        delay_text = self.font.render("Integration: 0.0ms", True, (255, 100, 100))
        self.screen.blit(delay_text, (col2_x, y)); y += 20
        
        # Feedforward Visualization
        # Draw a simple MLP diagram: Input(4) -> Hidden(20) -> Output(4)
        y += 10
        mlp_w = 160
        mlp_h = 200
        pygame.draw.rect(self.screen, (10, 10, 15), (col2_x, y, mlp_w, mlp_h))
        
        activations = getattr(active_ghost, 'last_activations', None)
        
        # Architecture positions
        cx_in = col2_x + 20
        cx_hid = col2_x + 80
        cx_out = col2_x + 140
        
        cy_start = y + 20
        space_in = 40
        space_hid = 8
        space_out = 40
        
        # Draw connections (dim)
        for i in range(4):
            for j in range(20):
                # Input -> Hidden
                start = (cx_in, cy_start + i*space_in)
                end = (cx_hid, y + 10 + j*space_hid)
                # If active, brighten line
                val = activations[j] if activations is not None else 0
                alpha = int(min(255, val * 50))
                color = (alpha, alpha, alpha) if alpha > 20 else (30,30,30)
                if alpha > 20: 
                    pygame.draw.line(self.screen, color, start, end, 1)

        # Draw Nodes
        # INPUT NODES
        for i in range(4):
            pos = (cx_in, cy_start + i*space_in)
            pygame.draw.circle(self.screen, (100, 100, 255), pos, 5)
            
        # HIDDEN NODES (Light up if active)
        for i in range(20):
            pos = (cx_hid, y + 10 + i*space_hid)
            val = activations[i] if activations is not None else 0
            # Relu activation usually 0 to ~3
            intensity = min(255, int(val * 100))
            color = (intensity, intensity, intensity)
            pygame.draw.circle(self.screen, color, pos, 3)
            
        # OUTPUT NODES
        for i in range(4):
            pos = (cx_out, cy_start + i*space_out)
            # Highlight chosen direction
            is_chosen = False
            # Map dir to index 0=R,1=L,2=U,3=D (from ghost logic)
            # active_ghost.move_dir matches index?
            # 'right'=0, 'left'=1, 'up'=2, 'down'=3
            idx_map = {'right':0, 'left':1, 'up':2, 'down':3}
            if idx_map.get(active_ghost.move_dir) == i:
                is_chosen = True
                
            color = (255, 0, 0) if is_chosen else (50, 0, 0)
            pygame.draw.circle(self.screen, color, pos, 6)
            if is_chosen:
                 pygame.draw.circle(self.screen, WHITE, pos, 2)
        
        # Spike history graph
        label = self.font.render("Activity History:", True, (150, 150, 150))
        self.screen.blit(label, (panel_x, y))
        y += 16
        
        graph_h = 40
        pygame.draw.rect(self.screen, (10, 10, 20), (panel_x, y, panel_w - 10, graph_h))
        
        if len(self.spike_history) > 1:
            max_spike = max(self.spike_history) if max(self.spike_history) > 0 else 1
            points = []
            for i, s in enumerate(self.spike_history):
                px = panel_x + int(i * (panel_w - 10) / self.max_history)
                py = y + graph_h - int(s / max_spike * graph_h)
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(self.screen, YELLOW, False, points, 2)
        y += graph_h + 15
        
        # === ZOMBIE AGENTS (Ghosts) ===
        pygame.draw.line(self.screen, (60, 60, 100), (panel_x, y - 5), (panel_x + panel_w - 10, y - 5))
        label = self.font.render("GHOSTS (ZOMBIE MLP)", True, RED)
        self.screen.blit(label, (panel_x, y))
        y += 18
        
        # Ghost spikes (simulated - feedforward is instant)
        ghost_spikes = 12  # Fixed low value to demonstrate contrast
        spike_text = self.font.render(f"Spikes/Decision: ~{ghost_spikes}", True, WHITE)
        self.screen.blit(spike_text, (panel_x, y))
        y += 18
        
        # Ghost bar (much smaller to show contrast)
        bar_w = ghost_spikes * 2
        pygame.draw.rect(self.screen, (40, 40, 60), (panel_x, y, panel_w - 10, 12))
        pygame.draw.rect(self.screen, (255, 100, 100), (panel_x, y, bar_w, 12))
        y += 20
        
        # Comparison text
        y += 10
        pygame.draw.line(self.screen, (60, 60, 100), (panel_x, y), (panel_x + panel_w - 10, y))
        y += 10
        label = self.font.render("COMPARISON:", True, (100, 200, 255))
        self.screen.blit(label, (panel_x, y))
        y += 18
        
        # Integration ratio
        avg_pac = sum(self.spike_history) / len(self.spike_history) if self.spike_history else 0
        ratio = avg_pac / ghost_spikes if ghost_spikes > 0 else 0
        ratio_text = self.font.render(f"Integration: {ratio:.1f}x", True, WHITE)
        self.screen.blit(ratio_text, (panel_x, y))
        y += 18
        
        # Verdict
        if ratio > 3:
            verdict = "CONSCIOUS: High Integration"
            color = (100, 255, 100)
        elif ratio > 1.5:
            verdict = "CONSCIOUS: Moderate"
            color = (255, 255, 100)
        else:
            verdict = "Warming up..."
            color = (150, 150, 150)
            
        v_text = self.font.render(verdict, True, color)
        self.screen.blit(v_text, (panel_x, y))

if __name__ == "__main__":
    Game().run()
