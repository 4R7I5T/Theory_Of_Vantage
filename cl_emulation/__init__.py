"""
Cortical Labs SDK Emulation Package
===================================
Provides a drop-in replacement for the cl-sdk that runs on top of a 
high-fidelity Izhikevich neural simulation engine.
"""

from .system import open
from .physics import NeuralPhysics
