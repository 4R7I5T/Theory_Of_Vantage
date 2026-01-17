"""
Cortical Labs SDK Emulation Package
===================================
Provides a drop-in replacement for the cl-sdk that runs on top of a 
high-fidelity Izhikevich neural simulation engine.

Set CL_USE_BRIAN2=1 to use Brian2 backend (requires brian2 package).
Default: numpy-based vectorized physics.
"""

import os

from .system import open
from .physics import NeuralPhysics

# Brian2 backend (optional)
USE_BRIAN2 = os.getenv("CL_USE_BRIAN2", "0") == "1"

if USE_BRIAN2:
    try:
        from .brian2_physics import Brian2Physics
        print("[CL_EMULATION] Using Brian2 physics backend")
    except ImportError as e:
        print(f"[CL_EMULATION] Brian2 not available: {e}, falling back to numpy")
        USE_BRIAN2 = False

