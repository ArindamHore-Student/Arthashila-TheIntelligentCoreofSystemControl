"""
Utility Functions for Arthashila

This module provides common utility functions used across the Arthashila application.
All helper functions are consolidated here for easier imports and reduced file count.
"""

import plotly.graph_objects as go

# Define what gets imported with "from utils import *"
__all__ = ["get_size", "format_priority", "create_line_chart", "create_bar_chart"]

def get_size(size_in_bytes, suffix="B"):
    """
    Convert bytes to a human-readable format (e.g., KB, MB, GB).
    
    Args:
        size_in_bytes (int): Size in bytes to convert
        suffix (str): Suffix to append to the unit (default: "B" for Bytes)
        
    Returns:
        str: Formatted string with size and appropriate unit (e.g., "123.45 MB")
    """
    for unit in ["", "K", "M", "G", "T", "P"]:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f}{unit}{suffix}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f}P{suffix}"


def format_priority(priority):
    """
    Format priority levels with color coding for display.
    
    Args:
        priority (str): Priority level ("Low", "Medium", or "High")
        
    Returns:
        str: HTML-formatted string with color coding
    """
    colors = {"Low": "#00FF00", "Medium": "#FFA500", "High": "#FF0000"}
    return f"<span style='color:{colors[priority]};'>{priority}</span>"


def create_line_chart(data, title, x_label, y_label):
    """
    Create a line chart using Plotly with consistent styling.
    
    Parameters:
        data (list of tuples): Data points as [(x1, y1), (x2, y2), ...] pairs
        title (str): Title of the chart
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
    
    Returns:
        plotly.graph_objects.Figure: A fully configured Plotly figure object
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
            line=dict(color="#4f8bf9"),
            fill="tozeroy",  # Fill area between line and x-axis
            fillcolor="rgba(79, 139, 249, 0.1)",  # Light blue fill
        )
    )
    
    # Update layout with consistent styling
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor="#0e1117",  # Dark background for the chart paper
        plot_bgcolor="#1a1f2c",   # Dark background for the plotting area
        font=dict(color="#FFFFFF"), # White text for better contrast
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig


def create_bar_chart(categories, values, title, x_label, y_label):
    """
    Create a bar chart using Plotly with consistent styling.
    
    Parameters:
        categories (list): List of category labels for the x-axis
        values (list): List of numeric values corresponding to each category
        title (str): Title of the chart
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
    
    Returns:
        plotly.graph_objects.Figure: A fully configured Plotly figure object
    """
    # Create the figure with Bar trace
    fig = go.Figure(
        data=go.Bar(
            x=categories,
            y=values,
            marker_color="#4f8bf9",  # Use theme color for bars
            opacity=0.8,          # Slight transparency
        )
    )
    
    # Update layout with consistent styling
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor="#0e1117",  # Dark background for the chart paper
        plot_bgcolor="#1a1f2c",   # Dark background for the plotting area
        font=dict(color="#FFFFFF"), # White text for better contrast
        bargap=0.2,               # Gap between bars
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig
