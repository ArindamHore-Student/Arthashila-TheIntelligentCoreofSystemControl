"""
Visualization Utilities Module for SimpleRTOS

This module provides visualization functions used throughout the SimpleRTOS application.
It encapsulates Plotly chart creation functions with consistent styling and formatting.
"""

import plotly.graph_objects as go

# Define the exported functions
__all__ = ["create_line_chart", "create_bar_chart"]

def create_line_chart(data, title, x_label, y_label):
    """
    Create a line chart using Plotly with consistent styling.
    
    This function generates a line chart with the SimpleRTOS theme styling applied,
    including dark background, appropriate colors, and consistent font styling.
    The chart includes markers at data points and a filled area under the line.
    
    Parameters:
        data (list of tuples): Data points as [(x1, y1), (x2, y2), ...] pairs
        title (str): Title of the chart
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
    
    Returns:
        plotly.graph_objects.Figure: A fully configured Plotly figure object
        
    Example:
        >>> data = [(1, 10), (2, 15), (3, 13), (4, 17)]
        >>> fig = create_line_chart(data, "Sample Metrics", "Time", "Value")
        >>> st.plotly_chart(fig)
    """
    # Extract x and y values from data tuples
    x_values = [t for t, _ in data]
    y_values = [v for _, v in data]
    
    # Create the figure with Scatter trace
    fig = go.Figure(
        data=go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",  # Show both lines and markers
            line=dict(color="blue"),
            fill="tozeroy",  # Fill area between line and x-axis
            fillcolor="rgba(0, 0, 255, 0.1)",  # Light blue fill
        )
    )
    
    # Update layout with consistent styling
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor="#222222",  # Dark background for the chart paper
        plot_bgcolor="#333333",   # Dark background for the plotting area
        font=dict(color="#FFFFFF"), # White text for better contrast
    )
    
    return fig


def create_bar_chart(categories, values, title, x_label, y_label):
    """
    Create a bar chart using Plotly with consistent styling.
    
    This function generates a bar chart with the SimpleRTOS theme styling applied,
    including dark background, appropriate colors, and consistent font styling.
    
    Parameters:
        categories (list): List of category labels for the x-axis
        values (list): List of numeric values corresponding to each category
        title (str): Title of the chart
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
    
    Returns:
        plotly.graph_objects.Figure: A fully configured Plotly figure object
        
    Example:
        >>> categories = ["Category A", "Category B", "Category C"]
        >>> values = [20, 35, 15]
        >>> fig = create_bar_chart(categories, values, "Sample Distribution", "Categories", "Count")
        >>> st.plotly_chart(fig)
    """
    # Create the figure with Bar trace
    fig = go.Figure(
        data=go.Bar(
            x=categories,
            y=values,
            marker_color="blue",  # Use blue for the bars
            opacity=0.8,          # Slight transparency
        )
    )
    
    # Update layout with consistent styling
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor="#222222",  # Dark background for the chart paper
        plot_bgcolor="#333333",   # Dark background for the plotting area
        font=dict(color="#FFFFFF"), # White text for better contrast
        bargap=0.2,               # Gap between bars
    )
    
    return fig
