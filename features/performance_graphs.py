"""
Performance Graphs Module for SimpleRTOS

This module provides real-time performance monitoring with interactive graphs.
It displays CPU and memory usage over time, with configurable refresh rates.

The module uses a deque data structure to store a fixed number of historical
data points for each metric, which are then displayed using line charts.
"""

import streamlit as st
import psutil
import time
from collections import deque
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization function from utils
from utils.visualization import create_line_chart

# Constants
MAX_DATA_POINTS = 60  # Store up to 60 data points (1 minute at 1-second refresh)
DEFAULT_REFRESH_RATE = 2  # Default refresh rate in seconds

def initialize_session_state():
    """
    Initialize the session state variables for storing time series data.
    Uses deque with a maximum length to automatically discard old data points.
    
    Returns:
        tuple: Tuple containing the CPU and memory data deques
    """
    # Initialize data storage if not already in session state
    if 'cpu_data' not in st.session_state:
        st.session_state.cpu_data = deque(maxlen=MAX_DATA_POINTS)
    
    if 'memory_data' not in st.session_state:
        st.session_state.memory_data = deque(maxlen=MAX_DATA_POINTS)
    
    return st.session_state.cpu_data, st.session_state.memory_data

def update_performance_data():
    """
    Collect current system performance metrics and add them to the data deques.
    
    Returns:
        tuple: Current CPU and memory usage percentages
    """
    current_time = time.time()
    cpu_percent = psutil.cpu_percent(interval=0.1)  # Quick sampling
    memory_percent = psutil.virtual_memory().percent
    
    # Add to data collections
    st.session_state.cpu_data.append((current_time, cpu_percent))
    st.session_state.memory_data.append((current_time, memory_percent))
    
    return cpu_percent, memory_percent

def render_performance_controls():
    """
    Render the control panel for performance monitoring settings.
    
    Returns:
        int: Selected refresh rate in seconds
    """
    # Add refresh rate selector
    refresh_rate = st.slider(
        "Refresh Rate (seconds)", 
        min_value=1, 
        max_value=10, 
        value=DEFAULT_REFRESH_RATE,
        help="How frequently to update the performance graphs"
    )
    
    return refresh_rate

def render_performance_graphs(cpu_data, memory_data):
    """
    Render the CPU and memory usage graphs side by side.
    
    Args:
        cpu_data (deque): Collection of CPU usage data points
        memory_data (deque): Collection of memory usage data points
    """
    # Create two columns for CPU and Memory graphs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CPU Usage")
        cpu_fig = create_line_chart(
            list(cpu_data),
            "CPU Usage Over Time",
            "Time",
            "CPU %"
        )
        st.plotly_chart(cpu_fig, use_container_width=True)
    
    with col2:
        st.subheader("Memory Usage")
        memory_fig = create_line_chart(
            list(memory_data),
            "Memory Usage Over Time",
            "Time",
            "Memory %"
        )
        st.plotly_chart(memory_fig, use_container_width=True)

def performance_graphs():
    """
    Main entry point for the performance graphs feature.
    
    This function:
    1. Initializes data storage
    2. Updates with current performance data
    3. Renders control panel and graphs
    4. Refreshes the display at the specified interval
    """
    st.title("📊 Performance Graphs")
    st.markdown("Monitor system performance metrics in real-time with interactive graphs")
    
    # Initialize session state for data storage
    cpu_data, memory_data = initialize_session_state()
    
    # Render control panel and get settings
    refresh_rate = render_performance_controls()
    
    # Add current performance data
    update_performance_data()
    
    # Show system metrics summary
    cpu_current = cpu_data[-1][1] if cpu_data else 0
    memory_current = memory_data[-1][1] if memory_data else 0
    
    # Display current metrics above the graphs
    col1, col2 = st.columns(2)
    col1.metric("Current CPU Usage", f"{cpu_current}%")
    col2.metric("Current Memory Usage", f"{memory_current}%")
    
    # Display graphs
    render_performance_graphs(cpu_data, memory_data)
    
    # Add information about the graphs
    with st.expander("About Performance Monitoring"):
        st.markdown("""
        ### Understanding Performance Metrics
        
        **CPU Usage**: Shows the percentage of CPU resources currently in use. High sustained 
        CPU usage can indicate processing bottlenecks or resource-intensive applications.
        
        **Memory Usage**: Shows the percentage of RAM currently in use. Memory that stays 
        consistently high may indicate memory leaks or insufficient RAM for your workload.
        
        The graphs show data for the last {0} data points. The refresh rate determines 
        how frequently new data points are collected.
        """.format(MAX_DATA_POINTS))
    
    # Add auto-refresh with the specified rate
    time.sleep(refresh_rate)
    st.rerun()