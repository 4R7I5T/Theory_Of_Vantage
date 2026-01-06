#!/usr/bin/env python3
"""
NEURAL PACMAN: Consciousness vs Zombies Demo
=============================================
A live demonstration of the Theory of Vantage.

- PACMAN: Driven by Recurrent Izhikevich Network (Conscious Candidate)
- GHOSTS: Driven by Feedforward MLP (Philosophical Zombies)

This illustrates that "Zombies" work via pure stimulus-response,
while consciousness requires integrated recurrent processing.
"""

import pygame
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cl_emulation.ghost_brain import GhostBrain, GhostBrainEnsemble

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 184, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 184, 82)

# Screen dimensions
CELL_SIZE = 24
MAZE_WIDTH = 19
MAZE_HEIGHT = 21
SCREEN_WIDTH = MAZE_WIDTH * CELL_SIZE
SCREEN_HEIGHT = MAZE_HEIGHT * CELL_SIZE + 60  # Extra for HUD

# Simple maze layout (0=path, 1=wall, 2=ghost_house)
MAZE = [
    "1111111111111111111",
    "1000000001000000001",
    "1011011101011101101",
    "1000000000000000001",
    "1011010111110101101",
    "1000010001000100001",
    "1111011101011101111",
    "0000010000000100000",
    "1111010111110101111",
    "0000000122210000000",
    "1111010122210101111",
    "0000010000000100000",
    "1111010111110101111",
    "1000000001000000001",
    "1011011101011101101",
    "1001000000000001001",
    "1101010111110101011",
    "1000010001000100001",
    "1011111101011111101",
    "1000000000000000001",
    "1111111111111111111",
]

class Ghost:
    def __init__(self, name, color, brain, start_x, start_y):
        self.name = name
        self.color = color
        self.brain = brain
        self.x = float(start_x)
        self.y = float(start_y)
        self.speed = 0.08
        self.direction = 0  # 0=right, 1=left, 2=up, 3=down
        # Start moving immediately if possible
        self.direction = 0
        
    def get_sensory_input(self, pacman_x, pacman_y):
        """Generate input for neural network."""
        return self.brain.get_sensory_input(
            (self.x, self.y),
            (pacman_x, pacman_y),
            MAZE_WIDTH, MAZE_HEIGHT
        )
    
    def can_move(self, x, y, direction, maze):
        dx, dy = [(1, 0), (-1, 0), (0, -1), (0, 1)][direction]
        next_x = int(round(x + dx))
        next_y = int(round(y + dy))
        if 0 <= next_x < MAZE_WIDTH and 0 <= next_y < MAZE_HEIGHT:
            return maze[next_y][next_x] != '1'
        return False

    def update(self, pacman_x, pacman_y, maze):
        # Determine if centered in tile (within small threshold)
        # We use a slightly wider threshold to catch the center crossing
        threshold = self.speed * 0.9
        
        # Current integer position
        curr_grid_x = int(round(self.x))
        curr_grid_y = int(round(self.y))
        
        # Check if we are at the center of the tile
        dist_x = abs(self.x - curr_grid_x)
        dist_y = abs(self.y - curr_grid_y)
        
        at_center = dist_x < threshold and dist_y < threshold
        
        if at_center:
            # Snap to exact center to prevent drift
            self.x = float(curr_grid_x)
            self.y = float(curr_grid_y)
            
            # Get available moves at this intersection
            valid_moves = []
            for d in range(4):
                if self.can_move(self.x, self.y, d, maze):
                    valid_moves.append(d)
            
            # Neural decision
            sensory = self.get_sensory_input(pacman_x, pacman_y)
            # Override network output: only consider valid moves
            # We get raw output from brain to see preferences, but mask invalid ones
            # For the simple GhostBrain (MLP), we just get the best action.
            # But the Zombie MLP might output an invalid wall move.
            # We need to pick the highest output that IS valid.
            
            # Assuming brain.forward returns (action, _) where action is argmax
            # We need raw logits if possible, but GhostBrain returns action.
            # Let's just trust the brain but fallback if invalid.
            nm_action, _ = self.brain.forward(sensory)
            
            if nm_action in valid_moves:
                self.direction = nm_action
            elif valid_moves:
                # If network chose wall, pick random valid move (Zombie stupidity/fallback)
                # Or pick valid move closest to desired (simplification: random)
                self.direction = valid_moves[np.random.randint(len(valid_moves))]
            else:
                # Stuck (shouldn't happen in standard maze)
                pass

        # Execute movement if valid
        if self.can_move(self.x, self.y, self.direction, maze):
            dx, dy = [(1, 0), (-1, 0), (0, -1), (0, 1)][self.direction]
            self.x += dx * self.speed
            self.y += dy * self.speed
        else:
            # Hit wall, stop/center
            self.x = float(int(round(self.x)))
            self.y = float(int(round(self.y)))
    
    def draw(self, screen):
        px = int(self.x * CELL_SIZE + CELL_SIZE // 2)
        py = int(self.y * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, self.color, (px, py), CELL_SIZE // 2 - 2)
        # Eyes
        pygame.draw.circle(screen, WHITE, (px - 4, py - 2), 4)
        pygame.draw.circle(screen, WHITE, (px + 4, py - 2), 4)
        pygame.draw.circle(screen, BLUE, (px - 4, py - 2), 2)
        pygame.draw.circle(screen, BLUE, (px + 4, py - 2), 2)


class Pacman:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.speed = 0.12
        self.direction = 0     # Current moving direction
        self.next_direction = 0 # Buffered input
        self.moving = False
        self.mouth_open = True
        self.mouth_timer = 0
        
    def can_move(self, x, y, direction, maze):
        dx, dy = [(1, 0), (-1, 0), (0, -1), (0, 1)][direction]
        next_x = int(round(x + dx))
        next_y = int(round(y + dy))
        if 0 <= next_x < MAZE_WIDTH and 0 <= next_y < MAZE_HEIGHT:
            return maze[next_y][next_x] != '1'
        return False
        
    def update(self, keys, maze):
        # 1. Handle Input (Buffered)
        if keys[pygame.K_LEFT]:
            self.next_direction = 1
            self.moving = True
        elif keys[pygame.K_RIGHT]:
            self.next_direction = 0
            self.moving = True
        elif keys[pygame.K_UP]:
            self.next_direction = 2
            self.moving = True
        elif keys[pygame.K_DOWN]:
            self.next_direction = 3
            self.moving = True
            
        # 2. Movement Logic (Grid-Locked)
        threshold = self.speed * 0.95
        curr_grid_x = int(round(self.x))
        curr_grid_y = int(round(self.y))
        
        dist_x = abs(self.x - curr_grid_x)
        dist_y = abs(self.y - curr_grid_y)
        at_center = dist_x < threshold and dist_y < threshold

        if at_center:
            # We are at an intersection/tile center: Try to switch to buffered direction
            self.x = float(curr_grid_x)
            self.y = float(curr_grid_y)
            
            if self.moving and self.can_move(self.x, self.y, self.next_direction, maze):
                self.direction = self.next_direction
            
            # Check if we can continue in current direction
            if not self.can_move(self.x, self.y, self.direction, maze):
                self.moving = False # Stop if hitting wall

        # 3. Apply Velocity
        if self.moving:
            if self.can_move(self.x, self.y, self.direction, maze) or not at_center:
                 dx, dy = [(1, 0), (-1, 0), (0, -1), (0, 1)][self.direction]
                 self.x += dx * self.speed
                 self.y += dy * self.speed
            else:
                 # Snap to center if stuck
                 self.x = float(curr_grid_x)
                 self.y = float(curr_grid_y)
        
        # Animate mouth
        self.mouth_timer += 1
        if self.mouth_timer > 5:
            self.mouth_open = not self.mouth_open
            self.mouth_timer = 0
    
    def draw(self, screen):
        px = int(self.x * CELL_SIZE + CELL_SIZE // 2)
        py = int(self.y * CELL_SIZE + CELL_SIZE // 2)
        
        if self.mouth_open:
            # Draw pacman with open mouth
            start_angle = [0.25, 0.75, 0.5, 0][self.direction]
            pygame.draw.circle(screen, YELLOW, (px, py), CELL_SIZE // 2 - 2)
            # Draw mouth as black wedge
            mouth_points = [(px, py)]
            angle_offset = [0, 3.14, 1.57, -1.57][self.direction]
            for a in np.linspace(-0.5, 0.5, 10):
                mx = px + int(np.cos(angle_offset + a) * CELL_SIZE // 2)
                my = py + int(np.sin(angle_offset + a) * CELL_SIZE // 2)
                mouth_points.append((mx, my))
            pygame.draw.polygon(screen, BLACK, mouth_points)
        else:
            pygame.draw.circle(screen, YELLOW, (px, py), CELL_SIZE // 2 - 2)


def draw_maze(screen, maze):
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if cell == '1':
                pygame.draw.rect(screen, BLUE, rect)
            elif cell == '2':
                pygame.draw.rect(screen, (50, 50, 50), rect)


def draw_hud(screen, font, ghost_brains):
    """Draw HUD showing that ghosts are 'Zombies'."""
    y_offset = MAZE_HEIGHT * CELL_SIZE + 5
    
    # Title
    title = font.render("CONSCIOUSNESS vs ZOMBIES", True, WHITE)
    screen.blit(title, (10, y_offset))
    
    # Ghost status
    zombie_text = font.render("GHOSTS: Feedforward MLP (ZOMBIE)", True, RED)
    screen.blit(zombie_text, (10, y_offset + 25))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Pacman: Consciousness vs Zombies")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)
    
    # Initialize ghost brains (Zombies)
    ghost_brains = GhostBrainEnsemble()
    
    # Create ghosts at ghost house positions
    ghosts = [
        Ghost("blinky", RED, ghost_brains.brains['blinky'], 9, 9),
        Ghost("pinky", PINK, ghost_brains.brains['pinky'], 8, 10),
        Ghost("inky", CYAN, ghost_brains.brains['inky'], 10, 10),
        Ghost("clyde", ORANGE, ghost_brains.brains['clyde'], 9, 10),
    ]
    
    # Create pacman
    pacman = Pacman(9, 15)
    
    # Make mutable maze
    maze = [list(row) for row in MAZE]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        
        # Update
        pacman.update(keys, maze)
        for ghost in ghosts:
            ghost.update(pacman.x, pacman.y, maze)
        
        # Draw
        screen.fill(BLACK)
        draw_maze(screen, maze)
        
        # Draw pellets (simple version)
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == '0':
                    px = x * CELL_SIZE + CELL_SIZE // 2
                    py = y * CELL_SIZE + CELL_SIZE // 2
                    pygame.draw.circle(screen, WHITE, (px, py), 2)
        
        pacman.draw(screen)
        for ghost in ghosts:
            ghost.draw(screen)
        
        draw_hud(screen, font, ghost_brains)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
