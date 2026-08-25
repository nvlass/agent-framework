"""
Reflection triggers — when and why the agent should reflect.

Three trigger types correspond to the three introspection modes:
- EVENT: Automatic, triggered by significant success/failure
- IDLE: Opportunistic, during downtime consolidation
- SELF_PROMPTED: Curiosity-driven, agent decides to introspect
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TriggerType(Enum):
    """Why reflection was triggered."""
    EVENT = "event"
    IDLE = "idle"
    SELF_PROMPTED = "self_prompted"


@dataclass
class ReflectionTrigger:
    """A trigger requesting the agent to reflect.

    Attributes:
        type: What kind of trigger (event, idle, curiosity)
        reason: Human-readable explanation of why reflection is needed
        episode_id: Optional link to a specific memory episode
    """
    type: TriggerType
    reason: str
    episode_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "reason": self.reason,
            "episode_id": self.episode_id,
        }
