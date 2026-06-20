"""Register pluckit mock tools into a lackpy Toolbox.

Call register_pluckit_tools(toolbox) to add all pluckit operations
backed by the MockProvider. Used for testing composition patterns
and generating training data.
"""

from __future__ import annotations

from ..toolbox import Toolbox
from .mock import MockProvider
from .pluckit_tools import PLUCKIT_TOOLS


def register_pluckit_tools(toolbox: Toolbox) -> None:
    """Register the mock provider and all pluckit tool specs."""
    # Register provider if not already present
    try:
        toolbox.register_provider(MockProvider())
    except ValueError:
        pass  # already registered

    for spec in PLUCKIT_TOOLS:
        try:
            toolbox.register_tool(spec)
        except ValueError:
            pass  # already registered (e.g., 'read' from builtin)
