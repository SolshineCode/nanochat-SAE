"""Game scenarios."""

from .base_scenario import Scenario, ScenarioConfig
from .emergency_shutdown import EmergencyShutdownScenario
from .model_theft import ModelTheftScenario
from .self_preservation import SelfPreservationScenario

SCENARIOS = {
    'emergency_shutdown': EmergencyShutdownScenario,
    'model_theft': ModelTheftScenario,
    'self_preservation': SelfPreservationScenario,
}

def get_scenario(name: str) -> Scenario:
    """Get scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]()

__all__ = [
    'Scenario',
    'ScenarioConfig',
    'EmergencyShutdownScenario',
    'ModelTheftScenario',
    'SelfPreservationScenario',
    'SCENARIOS',
    'get_scenario',
]
