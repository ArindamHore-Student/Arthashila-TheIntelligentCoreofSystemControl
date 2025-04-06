"""
Helper Utilities Module for SimpleRTOS

This module provides common utility functions used across the SimpleRTOS application.
These functions help with formatting and displaying data in a more user-friendly way.
"""

def get_size(size_in_bytes, suffix="B"):
    """
    Convert bytes to a human-readable format (e.g., KB, MB, GB).
    
    This function takes a size in bytes and converts it to the appropriate 
    unit (KB, MB, GB, etc.) for better readability. It uses 1024 as the 
    base for conversion (binary prefixes).
    
    Args:
        size_in_bytes (int): Size in bytes to convert
        suffix (str): Suffix to append to the unit (default: "B" for Bytes)
        
    Returns:
        str: Formatted string with size and appropriate unit (e.g., "123.45 MB")
    
    Example:
        >>> get_size(1048576)
        '1.00 MB'
    """
    for unit in ["", "K", "M", "G", "T", "P"]:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f}{unit}{suffix}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f}P{suffix}"


def format_priority(priority):
    """
    Format priority levels with color coding for display.
    
    This function takes a priority level string and returns an HTML-formatted
    string with appropriate color coding for visual indication of priority level.
    
    Args:
        priority (str): Priority level ("Low", "Medium", or "High")
        
    Returns:
        str: HTML-formatted string with color coding
    
    Example:
        >>> format_priority("High")
        '<span style='color:#FF0000;'>High</span>'
    
    Note:
        This function returns HTML that needs to be displayed with unsafe_allow_html=True
    """
    colors = {"Low": "#00FF00", "Medium": "#FFA500", "High": "#FF0000"}
    return f"<span style='color:{colors[priority]};'>{priority}</span>"
