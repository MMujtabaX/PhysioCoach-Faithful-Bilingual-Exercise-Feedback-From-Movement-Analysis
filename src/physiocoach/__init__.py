"""PhysioCoach-NLP: faithful exercise coaching feedback from structured movement analysis."""

from .config import load_config
from .faithfulness import CheckResult, check_rep, check_session
from .linearize import linearize_rep, linearize_session
from .schema import RepRecord, SessionRecord, SessionStats
from .templates import TemplateGenerator

__version__ = "0.1.0"

__all__ = [
    "CheckResult",
    "RepRecord",
    "SessionRecord",
    "SessionStats",
    "TemplateGenerator",
    "check_rep",
    "check_session",
    "linearize_rep",
    "linearize_session",
    "load_config",
]
