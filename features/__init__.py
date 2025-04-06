"""
Features Package for SimpleRTOS

This package contains the main feature modules that make up the SimpleRTOS application.
Each module represents a major functionality area and provides a Streamlit interface
for its specific feature set.

Modules included:
- system_overview: System hardware and software information display
- process_manager: Process monitoring and management interface
- performance_graphs: Real-time performance monitoring with graphs
- battery_management: Battery status and power management interface
- task_planning: Task and project management tools
- collaboration_tools: Team collaboration and communication features

Each module exposes a main function that is called from the SimpleRTOS application router.
"""

from .system_overview import system_overview
from .process_manager import process_manager
from .performance_graphs import performance_graphs
from .battery_management import battery_management
from .task_planning import task_planning
from .collaboration_tools import collaboration_tools

__all__ = [
    "system_overview",
    "process_manager",
    "performance_graphs",
    "battery_management",
    "task_planning",
    "collaboration_tools"
]
