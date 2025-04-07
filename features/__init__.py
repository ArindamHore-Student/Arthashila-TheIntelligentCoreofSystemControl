"""
Features Package for Arthashila

This package contains the main feature modules that make up the Arthashila application.
Each module represents a major functionality area and provides a Streamlit interface
for its specific feature set.

Modules included:
- system_overview: System hardware and software information display
- process_manager: Process monitoring and management interface
- performance_graphs: Real-time performance monitoring with graphs
- ai_analytics: AI-powered system analytics and process optimization
- battery_management: Battery status and power management interface
- task_planning: Task and project management tools

Each module exposes a main function that is called from the Arthashila application router.
"""

from .system_overview import system_overview
from .process_manager import process_manager
from .performance_graphs import performance_graphs
from .ai_analytics import ai_analytics
from .battery_management import battery_management
from .task_planning import task_planning

__all__ = [
    "system_overview",
    "process_manager",
    "performance_graphs",
    "ai_analytics",
    "battery_management",
    "task_planning"
]
