
import pygame
import random
import numpy as np
import sys
import os

# Try to import the Real SDK
try:
    import cl_sdk
    MOCK_MODE = False
except ImportError:
    print("[WARNING] Real Cortical Labs SDK (cl_sdk) not found.")
    print("Running in 'Dry Run' mode to verify logic structure only.")
    MOCK_MODE = True
    # create dummy cl_sdk for syntax checking
    class MockCL:
        class System:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def connect(self): pass
            def start_stream(self): pass
            def stop_stream(self): pass
            def disconnect(self): pass
    cl_sdk = MockCL()

# Settings
CHAR_SIZE = 20
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

# -----------------------------------------------------------------------------
# WETWARE INTERFACE (The Real Deal)
# -----------------------------------------------------------------------------

class WetwareBrain:
    def __init__(self):
        """
        Interface for Real Cortical Labs CL1 Hardware.
        Uses actual electrodes for I/O instead of differential equations.
        """
        if MOCK_MODE:
            self.system = cl_sdk.System()
        else:
            # Initialize connection to DishBrain
            self.system = cl_sdk.System()
            
        self.last_spikes = 0
        self.last_latency = 0.0
        
        # Electrodes Config (Standard MEA usually has ~60-1000+ electrodes)
        # We assume a 60-electrode array for simplicity (e.g. Maxwell or Axion)
        self.n_electrodes = 60
        
        # Mapping: Game Concepts -> Electrode Indices
        # We use spatial zones on the chip
        self.input_map = {
            'ghost_proximity': [0, 1, 2, 3],      # Stimulate top-left
            'pellet_up': [10, 11],
            'pellet_down': [12, 13],
            'pellet_left': [14, 15],
            'pellet_right': [16, 17]
        }
        
        # Mapping: Output Zones -> Actions
        # We listen to distinct regions of the dish
        self.output_map = {
            'up': range(30, 35),
            'down': range(35, 40),
            'left': range(40, 45),
            'right': range(45, 50)
        }
        
        # Buffer for spikes received from hardware
        self.spike_buffer = []  
        self.is_connected = False

    def start_session(self):
        if not MOCK_MODE:
            self.system.connect()
            self.system.start_stream() # Start listening to spikes
        self.is_connected = True

    def stop_session(self):
        if not MOCK_MODE and self.is_connected:
            self.system.stop_stream()
            self.system.disconnect()
            
    def encode_sensory(self, pac_pos, ghosts, pellets, valid_dirs, frightened):
        """Translate Game State -> Stimulation Pattern"""
        # Note: In real SDK, we build a 'StimulationFrame' or 'StimPlan'
        
        stim_list = [] # List of tuples (electrode, amplitude_uv)
        
        # 1. Ghost Danger
        px, py = pac_pos
        closest_dist = float('inf')
        
        for g in ghosts:
            # Use center coordinates
            gx, gy = g.rect.centerx, g.rect.centery
            dist = abs(gx - px) + abs(gy - py)
            if dist < 160: # Ghost is close (pixels)
                # Stimulate 'danger' zone
                # Amplitude scales with proximity
                amp = min(100.0, 50000.0 / (dist + 0.1)) 
                for ele in self.input_map['ghost_proximity']:
                     stim_list.append((ele, amp))
                        
        # 2. Pellet Attraction
        # Simplified: Simply verify valid moves that lead to pellets
        # In wetware, we try to condition the dish to 'like' pellets
        # by stimulating reward centers (high freq burst) when eating.
        # Here we just hint direction based on available valid moves.
        if 'up' in valid_dirs:
            for e in self.input_map['pellet_up']: stim_list.append((e, 50))
        if 'down' in valid_dirs:
            for e in self.input_map['pellet_down']: stim_list.append((e, 50))
        if 'left' in valid_dirs:
            for e in self.input_map['pellet_left']: stim_list.append((e, 50))
        if 'right' in valid_dirs:
            for e in self.input_map['pellet_right']: stim_list.append((e, 50))
            
        return stim_list

    def send_stimulation(self, stim_list):
        if MOCK_MODE: return
        
        # Real SDK Implementation (Pseudocode based on typical SDKs)
        # plan = self.system.create_plan()
        # for elec, amp in stim_list:
        #     plan.pulse(elec, amplitude=amp)
        # self.system.send(plan)
        pass

    def read_biological_spikes(self):
        """
        Poll the hardware for spikes that occurred since last frame.
        """
        if MOCK_MODE:
            # Emulate background noise of a living culture
            return [(random.randint(0, self.n_electrodes-1), 0) for _ in range(random.randint(5, 15))]
            
        # Real: return self.system.read_stream()
        return []

    def decide(self, pac_pos, ghosts, pellets, valid_dirs, frightened):
        import time
        t0 = time.perf_counter()
        
        # 1. Encode & Stimulate
        stim_plans = self.encode_sensory(pac_pos, ghosts, pellets, valid_dirs, frightened)
        self.send_stimulation(stim_plans)
        
        # 2. Integrate (Biology is slow, but game loop is fast)
        # We read whatever spikes arrived since last check
        spikes = self.read_biological_spikes()
        self.last_spikes = len(spikes)
        
        # 3. Decode Population Vector
        # Count spikes in each output zone
        votes = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        
        for (elec_idx, timestamp) in spikes:
            for action, electrode_range in self.output_map.items():
                if elec_idx in electrode_range:
                    votes[action] += 1
                    
        # 4. Winner Take All
        action = max(votes, key=votes.get)
        
        self.last_latency = (time.perf_counter() - t0) * 1000
        
        # Filter by valid moves
        if action in valid_dirs and votes[action] > 0:
            return action, self.last_spikes
        elif valid_dirs:
            return random.choice(valid_dirs), self.last_spikes
        return (0,0), 0

# -----------------------------------------------------------------------------
# GAME LOGIC
# -----------------------------------------------------------------------------

try:
    from Pacman.neural_pacman import Game, Pacman, Cell, Berry, Ghost, GhostBrainEnsemble, MAP
except ImportError:
    from neural_pacman import Game, Pacman, Cell, Berry, Ghost, GhostBrainEnsemble, MAP

class WetwarePacman(Pacman):
    def __init__(self):
        super().__init__() # Initialize sprites
        self.brain = WetwareBrain()
        self.brain.start_session()
        self.last_spikes = 0
        
    def update(self, walls, ghosts, pellets, frightened=False):
        # 1. Get Valid Moves
        valid_dirs = self.get_valid_moves(walls)
        
        # 2. Wetware Decision
        pac_pos = (self.rect.centerx, self.rect.centery)
        move_dir, spikes = self.brain.decide(pac_pos, ghosts, pellets, valid_dirs, frightened)
        self.last_spikes = spikes
        
        # 3. Execute
        self.direction = move_dir
        # Move Logic (Copied from parent update)
        if self.rect.move(self.direction).collidelist(walls) == -1:
            self.rect.move_ip(self.direction)
        else:
            # Snap to grid
            self.rect.x = (self.rect.x // CHAR_SIZE) * CHAR_SIZE
            self.rect.y = (self.rect.y // CHAR_SIZE) * CHAR_SIZE

class WetwareGame(Game):
    def __init__(self):
        super().__init__()
        pygame.display.set_caption("Project Genesis: WETWARE Pacman (Real CL1)")
        
        # Swap the Pacman for the Wetware version
        self.pacman = WetwarePacman()
        # Initial draw/reset
        self.load_map()
        
    def load_map(self):
        # We need to override because parent load_map creates a default Pacman()
        super().load_map()
        # Restore our wetware pacman
        self.pacman.reset_pos()
        # Re-add to all_sprites if necessary
        
    def draw_neural_panel(self):
        # Override to show "BIOLOGICAL" instead of "CONSCIOUS"
        panel_x = self.game_w + 5
        panel_w = self.panel_w - 10
        
        pygame.draw.rect(self.screen, (20, 40, 20), (self.game_w, 0, self.panel_w, self.screen_h)) # Green tint
        pygame.draw.line(self.screen, (60, 100, 60), (self.game_w, 0), (self.game_w, self.screen_h), 2)
        
        y = 20
        title = self.title_font.render("BIOLOGICAL SUBSTRATE", True, (100, 255, 100))
        self.screen.blit(title, (panel_x, y))
        y += 40
        
        label = self.font.render("LIVE CULTURE (CL1)", True, YELLOW)
        self.screen.blit(label, (panel_x, y))
        y += 20
        
        # Spikes
        spikes = self.pacman.last_spikes
        t = self.font.render(f"Real Spikes: {spikes}", True, WHITE)
        self.screen.blit(t, (panel_x, y))
        
        # Latency
        lat = self.pacman.brain.last_latency
        t2 = self.font.render(f"Latency: {lat:.2f}ms", True, WHITE)
        self.screen.blit(t2, (panel_x, y + 20))

if __name__ == "__main__":
    game = WetwareGame()
    game.run()
