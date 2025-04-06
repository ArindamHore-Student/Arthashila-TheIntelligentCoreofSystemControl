# SimpleRTOS Developer Guide

This guide provides detailed information for developers who want to understand, modify, or extend the SimpleRTOS application.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Core Components](#core-components)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Common Issues](#common-issues)

## Architecture Overview

SimpleRTOS is built using a modular architecture with the following key components:

1. **Main Application (main.py)**
   - Serves as the entry point and initializes the Streamlit UI
   - Manages navigation between different features
   - Applies global styling and UI components
   - Orchestrates the routing to feature modules

2. **Feature Modules (features/)**
   - Self-contained modules implementing specific functionality
   - Each module is responsible for its own UI rendering and data processing
   - Modules communicate with system resources through utility functions

3. **Utility Modules (utils/)**
   - Shared helper functions used across multiple feature modules
   - Visualization utilities for creating consistent charts and graphs
   - Common helper functions for data formatting and processing

## Project Structure

```
SimpleRTOS/
├── main.py                  # Application entry point
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation configuration
├── README.md                # User documentation
├── DEVELOPERS.md            # Developer documentation
├── features/                # Feature modules
│   ├── __init__.py          # Feature module exports
│   ├── system_overview.py   # System info display
│   ├── process_manager.py   # Process management
│   ├── performance_graphs.py # Performance monitoring
│   ├── battery_management.py # Battery monitoring
│   ├── task_planning.py     # Task planning
│   └── collaboration_tools.py # Team collaboration
└── utils/                   # Utility functions
    ├── __init__.py          # Utility module exports
    ├── helpers.py           # Common helper functions
    └── visualization.py     # Visualization utilities
```

## Coding Standards

### Python Conventions

- Follow PEP 8 style guidelines
- Use Google-style docstrings for all functions and modules
- Include type hints for function parameters and return values
- Maintain 4-space indentation
- Maximum line length of 100 characters

### Documentation

- All modules should have a module-level docstring explaining their purpose
- All functions should have docstrings with parameters, return values, and examples when appropriate
- Document complex algorithms with comments explaining the approach

### UI Components

- Follow Streamlit best practices for component layout
- Maintain consistent styling across all feature modules
- Use the provided CSS classes for UI components

## Core Components

### Main Application (main.py)

The main application file sets up the Streamlit UI framework and handles navigation between different features. Key functions:

- `load_custom_css()`: Applies custom CSS styling to enhance the UI
- `render_sidebar()`: Creates the navigation sidebar and returns the selected option
- `render_system_status()`: Displays system status metrics in the sidebar
- `render_footer()`: Displays the application footer
- `main()`: Entry point that configures the page and routes to appropriate modules

### Feature Modules

Each feature module is responsible for implementing a specific functionality:

1. **system_overview.py**
   - Displays system information, CPU details, and memory information
   - Renders system metrics using clean, visual components

2. **process_manager.py**
   - Lists running processes with their resource usage
   - Allows terminating processes and managing process priority

3. **performance_graphs.py**
   - Creates and displays interactive charts for system performance metrics
   - Tracks historical data for CPU, memory, and disk usage

4. **battery_management.py**
   - Monitors battery status and provides power-saving recommendations
   - Displays battery statistics and remaining time estimates

5. **task_planning.py**
   - Implements a task management system with creation, editing, and deletion
   - Includes task categorization, prioritization, and filtering

6. **collaboration_tools.py**
   - Provides features for team coordination and communication
   - Includes messaging and shared task management capabilities

### Utility Modules

Utility modules provide shared functionality used across the application:

1. **helpers.py**
   - Contains common helper functions for data formatting and processing
   - Includes system information retrieval functions

2. **visualization.py**
   - Provides functions for creating consistent charts and graphs
   - Ensures consistent styling and behavior across visualizations

## Adding New Features

### Creating a New Feature Module

1. Create a new Python file in the `features` directory, e.g., `feature_name.py`
2. Import necessary Streamlit and utility modules
3. Create a main function (e.g., `feature_name()`) that will be called from `main.py`
4. Implement the feature functionality using Streamlit components
5. Add the feature to the import statements and navigation options in `main.py`

Example template for a new feature module:

```python
"""
Feature Name - Purpose description

This module implements a new feature for SimpleRTOS that provides specific functionality.
It includes UI components and data processing for [purpose].
"""

import streamlit as st
import psutil  # or other relevant libraries
from utils.helpers import some_helper_function
from utils.visualization import create_bar_chart

def feature_name():
    """
    Main function for the feature.
    Renders UI components and processes data for the feature.
    """
    st.title("Feature Name")
    
    # Feature implementation
    data = get_feature_data()
    display_feature_ui(data)
    
def get_feature_data():
    """Get data required for the feature."""
    # Implementation
    pass
    
def display_feature_ui(data):
    """Display UI components for the feature."""
    # Implementation
    pass

# Helper functions specific to this feature
def feature_specific_function():
    """Specific functionality for this feature."""
    # Implementation
    pass
```

### Integration with Main Application

To integrate your new feature into the main application:

1. Add an import for your feature module in `main.py`:
   ```python
   from features.feature_name import feature_name
   ```

2. Add your feature to the navigation options in `render_sidebar()`:
   ```python
   options=[
       # Existing options
       "Feature Name"
   ],
   icons=[
       # Existing icons
       "icon-name"  # Choose an appropriate Bootstrap icon
   ]
   ```

3. Add a condition to route to your feature in the `main()` function:
   ```python
   elif selected_option == "Feature Name":
       feature_name()
   ```

## Testing

### Manual Testing

For manual testing of your feature:

1. Implement your feature in its own module
2. Run the application with `streamlit run main.py`
3. Navigate to your feature using the sidebar
4. Test all UI components and functionality
5. Verify that your feature works correctly with different inputs

### Automated Testing

For automated testing (to be implemented):

1. Create test files in a `tests` directory
2. Use pytest for unit and integration tests
3. Mock system dependencies using appropriate libraries
4. Run tests with `pytest tests/`

## Common Issues

### Streamlit Session State

When adding stateful features, use Streamlit's session state to persist data between reruns:

```python
if 'key' not in st.session_state:
    st.session_state['key'] = default_value
```

### Performance Considerations

- Avoid expensive operations in the main UI thread
- Cache data when appropriate using `@st.cache_data`
- Use efficient data structures for large datasets

### UI Responsiveness

- Use columns and containers for layout
- Consider mobile responsiveness with adaptable layouts
- Test UI on different screen sizes

## Extension Points

SimpleRTOS has several extension points where additional functionality can be added:

1. **New Visualization Types**: Add new chart types to `utils/visualization.py`
2. **System Monitoring Extensions**: Extend system monitoring capabilities in system_overview.py
3. **Process Management Features**: Add new process management capabilities
4. **Task Management Enhancements**: Extend task planning with additional features
5. **Data Export/Import**: Add functionality to export and import data

## Contributing

When contributing changes:

1. Follow the coding standards outlined in this document
2. Test your changes thoroughly
3. Update documentation to reflect your changes
4. Submit a pull request with a clear description of your changes

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [psutil Documentation](https://psutil.readthedocs.io/)
- [Plotly Documentation](https://plotly.com/python/) 