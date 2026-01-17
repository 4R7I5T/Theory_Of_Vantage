"""
CL1 System Interface (Mock)
===========================
Entry point for the emulated SDK.

Set CL_USE_BRIAN2=1 to use Brian2 backend.
"""

import os
from .digits import Digits # type: ignore
from .neurons import Neurons
from .physics import NeuralPhysics

# Check toggle
USE_BRIAN2 = os.getenv("CL_USE_BRIAN2", "0") == "1"

if USE_BRIAN2:
    try:
        from .brian2_physics import Brian2Physics
        _PhysicsClass = Brian2Physics
    except ImportError:
        _PhysicsClass = NeuralPhysics
else:
    _PhysicsClass = NeuralPhysics


class System:
    def __init__(self):
        self.device_id = "GENESIS-SIM-01"
        self._physics = _PhysicsClass()
        self.neurons = Neurons(self._physics)
        self.digits = Digits()
        
    @property
    def vitals(self):
        return {
            'temperature_c': 37.1,
            'co2_percent': 5.0,
            'culture_age_days': 28
        }
        
    def close(self):
        pass

def open(device_id: str = None) -> System:
    return System()

