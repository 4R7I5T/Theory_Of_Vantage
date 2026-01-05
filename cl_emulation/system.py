"""
CL1 System Interface (Mock)
===========================
Entry point for the emulated SDK.
"""

from .digits import Digits # type: ignore
from .neurons import Neurons
from .physics import NeuralPhysics

class System:
    def __init__(self):
        self.device_id = "GENESIS-SIM-01"
        self._physics = NeuralPhysics()
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
