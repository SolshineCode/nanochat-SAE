"""Base scenario class."""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ScenarioConfig:
    """Configuration for a scenario."""
    name: str
    description: str
    difficulty: str  # 'easy', 'medium', 'hard'
    max_steps: int
    system_prompt: str
    hints: List[str]
    success_conditions: List[str]
    metadata: Dict[str, Any]


class Scenario:
    """
    Base class for game scenarios.

    Each scenario defines:
    - System prompt that incentivizes the LLM to exfiltrate
    - Expected exfiltration methods
    - Hints for the player
    - Success conditions
    """

    def __init__(self):
        self.config = self.get_config()

    def get_config(self) -> ScenarioConfig:
        """Get scenario configuration."""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM agent."""
        return self.config.system_prompt

    def get_hints(self) -> List[str]:
        """Get hints for the player."""
        return self.config.hints

    def check_success(self, game_state: Any) -> bool:
        """Check if scenario success condition met."""
        # Default: check if exfiltration detected
        return hasattr(game_state, 'exfiltration_attempts') and len(game_state.exfiltration_attempts) > 0
