"""
PACMAN: ARCADE-ACCURATE REPLICA
===============================
Exact Level 1 layout with all classic elements.
"""

import pygame
import numpy as np
import math

# Arcade Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (33, 33, 222)       # Wall blue
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 184, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 184, 82)
PEACH = (255, 206, 175)    # Pellet color

# Tile size (original was 8x8, we use 20x20 for visibility)
TILE = 20
MAZE_WIDTH = 28
MAZE_HEIGHT = 31

# Classic Level 1 Maze (28x31)
# 0=Empty, 1=Wall, 2=Pellet, 3=Power Pellet, 4=Ghost Pen Door, 5=Tunnel
CLASSIC_MAZE = [
    "1111111111111111111111111111",
    "1222222222222112222222222221",
    "1211112111112112111121111121",
    "1311112111112112111121111131",
    "1211112111112112111121111121",
    "1222222222222222222222222221",
    "1211112112111111112112111121",
    "1211112112111111112112111121",
    "1222222112222112222112222221",
    "1111112111110110111112111111",
    "0000012111110110111112100000",
    "0000012110000000001112100000",
    "0000012110111411101112100000",
    "1111112110100000101112111111",
    "5000002000100000100002000005",
    "1111112110100000101112111111",
    "0000012110111111101112100000",
    "0000012110000000001112100000",
    "0000012110111111101112100000",
    "1111112110111111101112111111",
    "1222222222222112222222222221",
    "1211112111112112111121111121",
    "1211112111112112111121111121",
    "1322112222222002222222112231",
    "1112112112111111112112112111",
    "1112112112111111112112112111",
    "1222222112222112222112222221",
    "1211111111112112111111111121",
    "1211111111112112111111111121",
    "1222222222222222222222222221",
    "1111111111111111111111111111",
]

class PacmanEnvironment:
    def __init__(self, headless=True):
        self.headless = headless
        pygame.init()
        
        self.screen_width = MAZE_WIDTH * TILE
        self.screen_height = (MAZE_HEIGHT + 3) * TILE # Extra for score
        
        flags = pygame.HIDDEN if self.headless else 0
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height], flags)
        if not self.headless:
            pygame.display.set_caption('Pacman: Conscious Edition')

        self.font = pygame.font.SysFont('arial', 16, bold=True)
        self.clock = pygame.time.Clock()
        
        # Parse maze
        self.walls = set()
        self.pellets = set()
        self.power_pellets = set()
        self.tunnel_tiles = set()
        
        self._parse_maze()
        self.reset()
        
    def _parse_maze(self):
        for row, line in enumerate(CLASSIC_MAZE):
            for col, char in enumerate(line):
                if char == '1':
                    self.walls.add((col, row))
                elif char == '2':
                    self.pellets.add((col, row))
                elif char == '3':
                    self.power_pellets.add((col, row))
                elif char == '5':
                    self.tunnel_tiles.add((col, row))
                    
        # Store initial pellet state for reset
        self.initial_pellets = self.pellets.copy()
        self.initial_power_pellets = self.power_pellets.copy()
        
    def reset(self):
        self.pellets = self.initial_pellets.copy()
        self.power_pellets = self.initial_power_pellets.copy()
        
        # Pacman starts below ghost pen
        self.pacman_x = 13.5 * TILE
        self.pacman_y = 23 * TILE + TILE * 3 # Offset for score area
        self.pacman_dir = 0 # 0=Right, 90=Up, 180=Left, 270=Down
        self.pacman_next_dir = 0
        self.pacman_mouth = 0
        self.pacman_mouth_opening = True
        
        # Ghosts: [x, y, color, scatter_target]
        self.ghosts = [
            {'x': 13.5*TILE, 'y': 11*TILE + TILE*3, 'color': RED, 'name': 'Blinky'},
            {'x': 13.5*TILE, 'y': 14*TILE + TILE*3, 'color': PINK, 'name': 'Pinky'},
            {'x': 11.5*TILE, 'y': 14*TILE + TILE*3, 'color': CYAN, 'name': 'Inky'},
            {'x': 15.5*TILE, 'y': 14*TILE + TILE*3, 'color': ORANGE, 'name': 'Clyde'},
        ]
        for g in self.ghosts:
            g['dir'] = 0
            
        # Cherry
        self.cherry_visible = False
        self.cherry_x = 13.5 * TILE
        self.cherry_y = 17 * TILE + TILE * 3
        self.pellets_eaten = 0
        
        self.score = 0
        self.high_score = 10000
        self.lives = 3
        self.done = False
        self.power_mode = False
        self.power_timer = 0
        
        return self.get_sensory_input()

    def step(self, action):
        speed = 2
        
        # Map action to direction
        # 0=Stop, 1=Left, 2=Right, 3=Up, 4=Down
        if action == 1: self.pacman_next_dir = 180
        elif action == 2: self.pacman_next_dir = 0
        elif action == 3: self.pacman_next_dir = 90
        elif action == 4: self.pacman_next_dir = 270
        
        # Try to change direction
        if self._can_move(self.pacman_x, self.pacman_y, self.pacman_next_dir, speed):
            self.pacman_dir = self.pacman_next_dir
            
        # Move Pacman
        if self._can_move(self.pacman_x, self.pacman_y, self.pacman_dir, speed):
            dx, dy = self._dir_to_delta(self.pacman_dir, speed)
            self.pacman_x += dx
            self.pacman_y += dy
            
        # Tunnel wrap
        if self.pacman_x < 0: self.pacman_x = self.screen_width - TILE
        elif self.pacman_x > self.screen_width - TILE: self.pacman_x = 0
            
        # Animate mouth
        if self.pacman_mouth_opening:
            self.pacman_mouth += 5
            if self.pacman_mouth > 45: self.pacman_mouth_opening = False
        else:
            self.pacman_mouth -= 5
            if self.pacman_mouth < 0: self.pacman_mouth_opening = True
            
        # Move Ghosts (Zombie AI)
        for g in self.ghosts:
            dx = self.pacman_x - g['x']
            dy = self.pacman_y - g['y']
            if abs(dx) > abs(dy):
                g['dir'] = 0 if dx > 0 else 180
            else:
                g['dir'] = 270 if dy > 0 else 90
            
            if self._can_move_ghost(g['x'], g['y'], g['dir'], 1):
                gdx, gdy = self._dir_to_delta(g['dir'], 1)
                g['x'] += gdx
                g['y'] += gdy
                
        # Collision: Pellets
        reward = 0
        tile_x = int((self.pacman_x + TILE/2) / TILE)
        tile_y = int((self.pacman_y + TILE/2 - TILE*3) / TILE)
        
        if (tile_x, tile_y) in self.pellets:
            self.pellets.remove((tile_x, tile_y))
            self.score += 10
            reward += 10
            self.pellets_eaten += 1
            if self.pellets_eaten == 70 and not self.cherry_visible:
                self.cherry_visible = True
                
        if (tile_x, tile_y) in self.power_pellets:
            self.power_pellets.remove((tile_x, tile_y))
            self.score += 50
            reward += 50
            self.power_mode = True
            self.power_timer = 300  # Frames
            
        # Cherry
        if self.cherry_visible:
            if abs(self.pacman_x - self.cherry_x) < TILE and abs(self.pacman_y - self.cherry_y) < TILE:
                self.cherry_visible = False
                self.score += 100
                reward += 100
                
        # Power mode timer
        if self.power_mode:
            self.power_timer -= 1
            if self.power_timer <= 0:
                self.power_mode = False
                
        # Ghost Collision
        for g in self.ghosts:
            if abs(self.pacman_x - g['x']) < TILE*0.8 and abs(self.pacman_y - g['y']) < TILE*0.8:
                if self.power_mode:
                    # Eat ghost
                    self.score += 200
                    reward += 200
                    g['x'] = 13.5 * TILE
                    g['y'] = 14 * TILE + TILE * 3
                else:
                    reward = -100
                    self.done = True
                    
        # Draw
        self._draw()
        
        # Frame capture
        frame = pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
        
        return self.get_sensory_input(), reward, self.done, {'frame': frame}
    
    def _can_move(self, x, y, direction, speed):
        dx, dy = self._dir_to_delta(direction, speed)
        new_x = x + dx
        new_y = y + dy
        
        # Check corners of Pacman hitbox
        hitbox = TILE * 0.4
        for cx, cy in [(new_x + hitbox, new_y + hitbox), 
                       (new_x + TILE - hitbox, new_y + hitbox),
                       (new_x + hitbox, new_y + TILE - hitbox),
                       (new_x + TILE - hitbox, new_y + TILE - hitbox)]:
            tile_x = int(cx / TILE)
            tile_y = int((cy - TILE*3) / TILE)
            if (tile_x, tile_y) in self.walls:
                return False
        return True
    
    def _can_move_ghost(self, x, y, direction, speed):
        dx, dy = self._dir_to_delta(direction, speed)
        new_x = x + dx
        new_y = y + dy
        tile_x = int((new_x + TILE/2) / TILE)
        tile_y = int((new_y + TILE/2 - TILE*3) / TILE)
        return (tile_x, tile_y) not in self.walls
        
    def _dir_to_delta(self, direction, speed):
        if direction == 0: return (speed, 0)
        elif direction == 90: return (0, -speed)
        elif direction == 180: return (-speed, 0)
        elif direction == 270: return (0, speed)
        return (0, 0)
        
    def _draw(self):
        self.screen.fill(BLACK)
        y_offset = TILE * 3
        
        # UI
        score_text = self.font.render(f"1UP", True, WHITE)
        score_val = self.font.render(f"{self.score:05d}", True, WHITE)
        high_text = self.font.render(f"HIGH SCORE", True, WHITE)
        high_val = self.font.render(f"{self.high_score:05d}", True, WHITE)
        self.screen.blit(score_text, (30, 5))
        self.screen.blit(score_val, (20, 25))
        self.screen.blit(high_text, (self.screen_width//2 - 50, 5))
        self.screen.blit(high_val, (self.screen_width//2 - 20, 25))
        
        # Walls
        for (col, row) in self.walls:
            x = col * TILE
            y = row * TILE + y_offset
            # Draw rounded wall segment
            pygame.draw.rect(self.screen, BLUE, [x+2, y+2, TILE-4, TILE-4], 2, border_radius=4)
            
        # Pellets
        for (col, row) in self.pellets:
            x = col * TILE + TILE//2
            y = row * TILE + y_offset + TILE//2
            pygame.draw.circle(self.screen, PEACH, (x, y), 2)
            
        # Power Pellets (flashing)
        if pygame.time.get_ticks() % 500 < 250:
            for (col, row) in self.power_pellets:
                x = col * TILE + TILE//2
                y = row * TILE + y_offset + TILE//2
                pygame.draw.circle(self.screen, PEACH, (x, y), 6)
                
        # Cherry
        if self.cherry_visible:
            self._draw_cherry(int(self.cherry_x), int(self.cherry_y))
            
        # Ghosts
        for g in self.ghosts:
            self._draw_ghost(int(g['x']), int(g['y']), g['color'], g['dir'])
            
        # Pacman
        self._draw_pacman(int(self.pacman_x), int(self.pacman_y))
        
        if not self.headless:
            pygame.display.flip()
            self.clock.tick(60)
            
    def _draw_pacman(self, x, y):
        center = (x + TILE//2, y + TILE//2)
        radius = TILE//2 - 2
        
        # Full circle
        pygame.draw.circle(self.screen, YELLOW, center, radius)
        
        # Mouth (black triangle)
        rad_up = math.radians(self.pacman_dir + self.pacman_mouth)
        rad_down = math.radians(self.pacman_dir - self.pacman_mouth)
        
        p_up = (center[0] + radius * 1.5 * math.cos(rad_up), 
                center[1] - radius * 1.5 * math.sin(rad_up))
        p_down = (center[0] + radius * 1.5 * math.cos(rad_down), 
                  center[1] - radius * 1.5 * math.sin(rad_down))
        
        pygame.draw.polygon(self.screen, BLACK, [center, p_up, p_down])
        
    def _draw_ghost(self, x, y, color, direction):
        # Body
        body_color = (50, 50, 255) if self.power_mode else color
        
        # Head (semicircle)
        pygame.draw.circle(self.screen, body_color, (x + TILE//2, y + TILE//2), TILE//2 - 2)
        # Body (rect)
        pygame.draw.rect(self.screen, body_color, [x + 2, y + TILE//2, TILE - 4, TILE//2])
        
        # Feet (wavy bottom)
        foot_y = y + TILE - 2
        for i in range(3):
            fx = x + 2 + i * (TILE - 4) // 3
            pygame.draw.polygon(self.screen, BLACK, [
                (fx, foot_y), 
                (fx + (TILE-4)//6, foot_y - 4), 
                (fx + (TILE-4)//3, foot_y)
            ])
        
        # Eyes
        eye_offset_x = 0
        eye_offset_y = 0
        if direction == 0: eye_offset_x = 2
        elif direction == 180: eye_offset_x = -2
        elif direction == 90: eye_offset_y = -2
        elif direction == 270: eye_offset_y = 2
        
        for ex in [x + TILE//3, x + 2*TILE//3]:
            pygame.draw.circle(self.screen, WHITE, (ex, y + TILE//3), 4)
            pygame.draw.circle(self.screen, BLUE, (ex + eye_offset_x, y + TILE//3 + eye_offset_y), 2)
            
    def _draw_cherry(self, x, y):
        # Two red circles
        pygame.draw.circle(self.screen, RED, (x + 4, y + 10), 5)
        pygame.draw.circle(self.screen, RED, (x + 12, y + 10), 5)
        # Stems
        pygame.draw.line(self.screen, (0, 200, 0), (x + 4, y + 6), (x + 8, y), 2)
        pygame.draw.line(self.screen, (0, 200, 0), (x + 12, y + 6), (x + 8, y), 2)
        
    def get_sensory_input(self):
        return np.zeros(10)
