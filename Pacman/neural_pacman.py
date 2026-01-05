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
        self.x = start_x
        self.y = start_y
        self.target_x = start_x
        self.target_y = start_y
        self.speed = 0.08  # Slower than pacman
        self.direction = 0  # 0=right, 1=left, 2=up, 3=down
        
    def get_sensory_input(self, pacman_x, pacman_y):
        """Generate input for neural network."""
        return self.brain.get_sensory_input(
            (self.x, self.y),
            (pacman_x, pacman_y),
            MAZE_WIDTH, MAZE_HEIGHT
        )
    
    def update(self, pacman_x, pacman_y, maze):
        # Only decide new direction when reaching a cell center
        if abs(self.x - self.target_x) < 0.05 and abs(self.y - self.target_y) < 0.05:
            self.x = self.target_x
            self.y = self.target_y
            
            # Get neural network decision
            sensory = self.get_sensory_input(pacman_x, pacman_y)
            action, _ = self.brain.forward(sensory)
            
            # Try to move in the decided direction
            dx, dy = [(1, 0), (-1, 0), (0, -1), (0, 1)][action]
            new_x, new_y = int(self.x + dx), int(self.y + dy)
            
            # Check if valid move
            if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT:
                if maze[new_y][new_x] != '1':
                    self.target_x = new_x
                    self.target_y = new_y
                    self.direction = action
                else:
                    # Try alternative directions
                    for alt_action in [0, 1, 2, 3]:
                        if alt_action == action:
                            continue
                        dx, dy = [(1, 0), (-1, 0), (0, -1), (0, 1)][alt_action]
                        new_x, new_y = int(self.x + dx), int(self.y + dy)
                        if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT:
                            if maze[new_y][new_x] != '1':
                                self.target_x = new_x
                                self.target_y = new_y
                                self.direction = alt_action
                                break
        
        # Smooth movement towards target
        if self.x < self.target_x:
            self.x = min(self.x + self.speed, self.target_x)
        elif self.x > self.target_x:
            self.x = max(self.x - self.speed, self.target_x)
        if self.y < self.target_y:
            self.y = min(self.y + self.speed, self.target_y)
        elif self.y > self.target_y:
            self.y = max(self.y - self.speed, self.target_y)
    
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
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.speed = 0.12
        self.direction = 0
        self.mouth_open = True
        self.mouth_timer = 0
        
    def update(self, keys, maze):
        # Handle input
        if abs(self.x - self.target_x) < 0.05 and abs(self.y - self.target_y) < 0.05:
            self.x = self.target_x
            self.y = self.target_y
            
            dx, dy = 0, 0
            if keys[pygame.K_LEFT]:
                dx, dy = -1, 0
                self.direction = 1
            elif keys[pygame.K_RIGHT]:
                dx, dy = 1, 0
                self.direction = 0
            elif keys[pygame.K_UP]:
                dx, dy = 0, -1
                self.direction = 2
            elif keys[pygame.K_DOWN]:
                dx, dy = 0, 1
                self.direction = 3
            
            new_x, new_y = int(self.x + dx), int(self.y + dy)
            if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT:
                if maze[new_y][new_x] != '1':
                    self.target_x = new_x
                    self.target_y = new_y
        
        # Movement
        if self.x < self.target_x:
            self.x = min(self.x + self.speed, self.target_x)
        elif self.x > self.target_x:
            self.x = max(self.x - self.speed, self.target_x)
        if self.y < self.target_y:
            self.y = min(self.y + self.speed, self.target_y)
        elif self.y > self.target_y:
            self.y = max(self.y - self.speed, self.target_y)
        
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
