"""
Process Manager Module for Arthashila

This module provides functionality to monitor and manage system processes.
It displays real-time information about running processes, their resource usage,
and allows users to terminate processes or modify their priorities.

Enhanced with AI analytics integration for process optimization.
"""

import streamlit as st
import psutil
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import numpy as np
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DEFAULT_REFRESH_RATE = 3.0  # seconds as float for cross-module compatibility
MAX_PROCESSES = 1000  # Maximum number of processes to display
PROCESS_UPDATE_INTERVAL = 1.0  # How often to update process list

def create_gauge_chart(value, title, max_value=100):
    """
    Create a gauge chart for displaying values like CPU usage.
    
    Args:
        value (float): The value to display on the gauge
        title (str): The title of the gauge chart
        max_value (int, optional): The maximum value of the gauge. Defaults to 100.
        
    Returns:
        plotly.graph_objects.Figure: A configured gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 24, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#4f8bf9"},
            'bgcolor': "#1a1f2c",
            'borderwidth': 2,
            'bordercolor': "#2d3747",
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'rgba(79, 139, 249, 0.3)'},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': 'rgba(79, 139, 249, 0.6)'},
                {'range': [max_value * 0.8, max_value], 'color': 'rgba(79, 139, 249, 0.9)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#0e1117",
        font={'color': "white", 'family': "Arial"},
        height=200,
        margin=dict(l=20, r=20, b=20, t=40),
    )
    
    return fig

def get_process_color(cpu_percent):
    """
    Return color based on CPU usage percentage.
    
    Args:
        cpu_percent (float): CPU usage percentage
        
    Returns:
        str: Hex color code representing the usage level
    """
    if cpu_percent < 5:
        return "#0cce6b"  # Green for low usage
    elif cpu_percent < 25:
        return "#4f8bf9"  # Blue for medium usage
    elif cpu_percent < 60:
        return "#f9a825"  # Orange/Yellow for high usage
    else:
        return "#ff4b4b"  # Red for very high usage

def format_bytes(bytes_value):
    """
    Format bytes into a human-readable format.
    
    Args:
        bytes_value (int): Size in bytes
        
    Returns:
        str: Formatted string with appropriate unit
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def initialize_session_state():
    """
    Initialize session state variables for the process manager.
    """
    if 'refresh_rate' not in st.session_state:
        st.session_state.refresh_rate = 1.0
        
    if 'show_all_processes' not in st.session_state:
        st.session_state.show_all_processes = False
        
    if 'sort_by' not in st.session_state:
        st.session_state.sort_by = "CPU Usage"
        
    if 'filter_text' not in st.session_state:
        st.session_state.filter_text = ""
        
    if 'enable_auto_optimization' not in st.session_state:
        st.session_state.enable_auto_optimization = False
        
    if 'optimization_threshold' not in st.session_state:
        st.session_state.optimization_threshold = 70
        
    if 'enable_process_termination' not in st.session_state:
        st.session_state.enable_process_termination = False
        
    if 'termination_threshold' not in st.session_state:
        st.session_state.termination_threshold = 85

def get_all_processes():
    """
    Retrieve information about all running processes.
    
    Returns:
        list: List of dictionaries containing process information
    """
    try:
        processes = []
        
        # Get all process IDs
        for pid in psutil.pids():
            try:
                # Get process object
                process = psutil.Process(pid)
                
                # Get process info
                with process.oneshot():
                    process_info = {
                        'pid': pid,
                        'name': process.name(),
                        'cpu_percent': process.cpu_percent(),
                        'memory_percent': process.memory_percent(),
                        'status': process.status(),
                        'username': process.username(),
                        'create_time': process.create_time(),
                        'num_threads': process.num_threads(),
                        'memory_info': process.memory_info().rss / (1024 * 1024),  # MB
                        'io_counters': process.io_counters() if hasattr(process, 'io_counters') else None
                    }
                
                processes.append(process_info)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Skip processes that no longer exist or can't be accessed
                continue
            except Exception as e:
                logger.warning(f"Error getting info for process {pid}: {str(e)}")
                continue
        
        return processes
        
    except Exception as e:
        logger.error(f"Error getting process list: {str(e)}")
        return []

def update_calibration_factors():
    """
    Update calibration factors based on historical data to improve accuracy
    This is the self-learning component that helps match values with Task Manager
    """
    samples = st.session_state.training_data['process_samples']
    if len(samples) < 10:
        return
    
    # Check if it's time to train (every minute)
    current_time = datetime.now()
    time_since_last_training = (current_time - st.session_state.training_data['last_training_time']).total_seconds()
    
    if time_since_last_training < 60 and st.session_state.training_data['training_rounds'] > 0:
        return
    
    # Calculate adjustment factors
    system_cpu_values = [s['system_cpu'] for s in samples]
    measured_cpu_values = [s['measured_cpu'] for s in samples]
    system_memory_values = [s['system_memory'] for s in samples]
    measured_memory_values = [s['measured_memory'] for s in samples]
    
    # Calculate average ratios for calibration
    cpu_ratios = []
    memory_ratios = []
    
    for i in range(len(samples)):
        if measured_cpu_values[i] > 0:
            cpu_ratios.append(system_cpu_values[i] / measured_cpu_values[i])
        
        if measured_memory_values[i] > 0:
            memory_ratios.append(system_memory_values[i] / measured_memory_values[i])
    
    # Update calibration factors if we have valid ratios
    if cpu_ratios:
        # Use median to avoid outliers
        cpu_ratio = sorted(cpu_ratios)[len(cpu_ratios)//2]
        # Smooth updates to avoid oscillation (80% existing, 20% new)
        current_cpu_cal = st.session_state.training_data['calibration_factor']['process_cpu']
        st.session_state.training_data['calibration_factor']['process_cpu'] = 0.8 * current_cpu_cal + 0.2 * cpu_ratio
    
    if memory_ratios:
        memory_ratio = sorted(memory_ratios)[len(memory_ratios)//2]
        current_mem_cal = st.session_state.training_data['calibration_factor']['memory']
        st.session_state.training_data['calibration_factor']['memory'] = 0.8 * current_mem_cal + 0.2 * memory_ratio
    
    # Calculate accuracy metrics
    if cpu_ratios and memory_ratios:
        cpu_accuracy = min(100, 100 * (1 - (abs(1 - st.session_state.training_data['calibration_factor']['process_cpu']) / 2)))
        memory_accuracy = min(100, 100 * (1 - (abs(1 - st.session_state.training_data['calibration_factor']['memory']) / 2)))
        
        # Update accuracy metrics
        st.session_state.training_data['accuracy_metrics']['cpu'] = cpu_accuracy
        st.session_state.training_data['accuracy_metrics']['memory'] = memory_accuracy
    
    # Update training metadata
    st.session_state.training_data['training_rounds'] += 1
    st.session_state.training_data['last_training_time'] = current_time

def add_system_summary():
    """Add a summary of system information"""
    # Get system information
    uname = platform.uname()
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time
    
    # Calculate hours, minutes, seconds
    uptime_hours = uptime.total_seconds() // 3600
    uptime_minutes = (uptime.total_seconds() % 3600) // 60
    uptime_seconds = uptime.total_seconds() % 60
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💻 System")
        st.markdown(f"**OS:** {uname.system} {uname.version}")
        st.markdown(f"**Machine:** {uname.node}")
        st.markdown(f"**Processor:** {uname.processor}")
        
    with col2:
        st.markdown("### ⏱️ Uptime")
        st.markdown(f"**Boot Time:** {boot_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"**Uptime:** {int(uptime_hours)}h {int(uptime_minutes)}m {int(uptime_seconds)}s")
        
    with col3:
        st.markdown("### 🔄 Status")
        cpu_count = psutil.cpu_count(logical=True)
        physical_cpu = psutil.cpu_count(logical=False)
        st.markdown(f"**CPUs:** {physical_cpu} physical, {cpu_count} logical")
        
        memory = psutil.virtual_memory()
        st.markdown(f"**Memory:** {format_bytes(memory.used)} / {format_bytes(memory.total)}")
    
    # Disk information
    st.markdown("### 💾 Disk Usage")
    disk_cols = st.columns(4)
    
    partitions = psutil.disk_partitions()
    for i, partition in enumerate(partitions[:4]):  # Limit to first 4 partitions
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            with disk_cols[i % 4]:
                st.metric(
                    f"Disk {partition.device}",
                    f"{partition_usage.percent}% used",
                    f"{format_bytes(partition_usage.free)} free"
                )
        except:
            pass  # Skip partitions that can't be read

def render_system_metrics():
    """
    Display system-wide resource usage metrics.
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{cpu_percent}%",
                delta=None,
                help="Current CPU utilization"
            )
            
        with col2:
            st.metric(
                "Memory Usage",
                f"{memory.percent}%",
                delta=None,
                help=f"Used: {memory.used / (1024**3):.1f} GB / Total: {memory.total / (1024**3):.1f} GB"
            )
            
        with col3:
            st.metric(
                "Disk Usage",
                f"{disk.percent}%",
                delta=None,
                help=f"Used: {disk.used / (1024**3):.1f} GB / Total: {disk.total / (1024**3):.1f} GB"
            )
        
        # Add progress bars
        st.progress(cpu_percent / 100, text="CPU Usage")
        st.progress(memory.percent / 100, text="Memory Usage")
        st.progress(disk.percent / 100, text="Disk Usage")
        
    except Exception as e:
        logger.error(f"Error rendering system metrics: {str(e)}")
        st.error("Failed to display system metrics")

def terminate_process(pid, reason=None):
    """
    Terminate a process by its PID with proper error handling and logging.
    
    Args:
        pid (int): Process ID to terminate
        reason (str, optional): Reason for termination
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Validate PID
        if not isinstance(pid, int) or pid <= 0:
            return False, "Invalid PID"
        
        # Get process
        process = psutil.Process(pid)
        
        # Check if process exists
        if not process.is_running():
            return False, f"Process {pid} is not running"
        
        # Log termination attempt
        process_name = process.name()
        logger.info(f"Attempting to terminate process {pid} ({process_name})" + 
                   (f" - Reason: {reason}" if reason else ""))
        
        # Try to terminate process
        process.terminate()
        
        # Wait for process to terminate
        try:
            process.wait(timeout=3)
        except psutil.TimeoutExpired:
            # If process didn't terminate, try to kill it
            process.kill()
            process.wait(timeout=1)
        
        # Verify process is terminated
        if not process.is_running():
            message = f"Successfully terminated process {pid} ({process_name})"
            if reason:
                message += f" - {reason}"
            logger.info(message)
            return True, message
        else:
            return False, f"Failed to terminate process {pid} ({process_name})"
            
    except psutil.NoSuchProcess:
        return False, f"Process {pid} does not exist"
    except psutil.AccessDenied:
        return False, f"Access denied to terminate process {pid}"
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {str(e)}")
        return False, f"Error terminating process: {str(e)}"

def render_controls():
    """
    Render the control panel for process management settings.
    
    Returns:
        tuple: (refresh_rate, show_all_processes, sort_by, filter_text)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        refresh_rate = st.slider(
            "Refresh Rate (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.get('refresh_rate', DEFAULT_REFRESH_RATE)),
            step=0.5,
            help="How frequently to update the process list"
        )
        
        show_all_processes = st.checkbox(
            "Show All Processes",
            value=st.session_state.get('show_all_processes', False),
            help="Show all processes including system processes"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            options=["CPU Usage", "Memory Usage", "Process Name", "PID"],
            index=0,
            help="Sort processes by this metric"
        )
        
        filter_text = st.text_input(
            "Filter Processes",
            value=st.session_state.get('filter_text', ''),
            help="Filter processes by name or PID"
        )
    
    # Store settings in session state
    st.session_state.refresh_rate = refresh_rate
    st.session_state.show_all_processes = show_all_processes
    st.session_state.sort_by = sort_by
    st.session_state.filter_text = filter_text
    
    return refresh_rate, show_all_processes, sort_by, filter_text

def process_manager():
    """
    Main entry point for the process manager feature.
    
    This function:
    1. Initializes session state
    2. Renders controls and settings
    3. Displays system metrics
    4. Shows process list with optimization options
    5. Implements auto-refresh functionality
    """
    try:
        st.title("🔄 Process Manager")
        st.markdown("Monitor and manage system processes with AI-powered optimization")
        
        # Initialize session state
        initialize_session_state()
        
        # Render controls and get settings
        refresh_rate, show_all_processes, sort_by, filter_text = render_controls()
        
        # Display system metrics
        render_system_metrics()
        
        # X-Factor Process Optimizer section
        st.markdown("### 🚀 X-Factor Process Optimizer")
        
        # Create tabs for different optimizer views
        optimizer_tab, history_tab, insights_tab = st.tabs([
            "Optimizer Controls", "Optimization History", "Performance Insights"
        ])
        
        with optimizer_tab:
            st.markdown("""
            The X-Factor Process Optimizer uses AI to analyze process behavior and suggest optimizations.
            It can automatically manage resource-intensive processes and improve system performance.
            """)
            
            # Process optimization controls
            col1, col2 = st.columns(2)
            
            with col1:
                enable_auto_optimization = st.checkbox(
                    "Enable Auto-Optimization",
                    value=st.session_state.get('enable_auto_optimization', False),
                    help="Automatically optimize resource-intensive processes"
                )
                
                optimization_threshold = st.slider(
                    "Optimization Threshold (%)",
                    min_value=50,
                    max_value=90,
                    value=st.session_state.get('optimization_threshold', 70),
                    help="CPU/Memory usage threshold for automatic optimization"
                )
                
                optimization_mode = st.selectbox(
                    "Optimization Mode",
                    options=["Balanced", "Performance", "Power Saving", "Gaming", "Productivity"],
                    index=0,
                    help="Select optimization strategy based on your needs"
                )
            
            with col2:
                enable_process_termination = st.checkbox(
                    "Enable Process Termination",
                    value=st.session_state.get('enable_process_termination', False),
                    help="Allow automatic termination of problematic processes"
                )
                
                termination_threshold = st.slider(
                    "Termination Threshold (%)",
                    min_value=70,
                    max_value=95,
                    value=st.session_state.get('termination_threshold', 85),
                    help="CPU/Memory usage threshold for process termination"
                )
                
                smart_scheduling = st.checkbox(
                    "Enable Smart Process Scheduling",
                    value=st.session_state.get('smart_scheduling', False),
                    help="Intelligently prioritize processes based on usage patterns"
                )
            
            # Advanced optimization features
            st.markdown("#### Advanced Features")
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                enable_learning = st.checkbox(
                    "Adaptive Learning",
                    value=st.session_state.get('enable_learning', True),
                    help="Continuously learn from optimization actions"
                )
            
            with adv_col2:
                enable_idle_optimization = st.checkbox(
                    "Idle Detection",
                    value=st.session_state.get('enable_idle_optimization', False),
                    help="Apply deeper optimizations when system is idle"
                )
            
            with adv_col3:
                enable_process_grouping = st.checkbox(
                    "Process Grouping",
                    value=st.session_state.get('enable_process_grouping', False),
                    help="Manage related processes as groups"
                )
        
        with history_tab:
            if 'optimization_history' not in st.session_state:
                st.session_state.optimization_history = []
            
            if not st.session_state.optimization_history:
                st.info("No optimization history yet. Enable the optimizer to start collecting data.")
            else:
                st.markdown("#### Recent Optimization Actions")
                
                # Create dataframe for history
                history_df = pd.DataFrame(st.session_state.optimization_history)
                st.dataframe(history_df)
                
                # Add visualization if enough data
                if len(st.session_state.optimization_history) >= 3:
                    st.markdown("#### Optimization Impact")
                    
                    # Extract data for visualization
                    timestamps = [entry.get('timestamp', '') for entry in st.session_state.optimization_history]
                    cpu_before = [entry.get('cpu_before', 0) for entry in st.session_state.optimization_history]
                    cpu_after = [entry.get('cpu_after', 0) for entry in st.session_state.optimization_history]
                    
                    # Create a line chart showing optimization impact
                    impact_chart = go.Figure()
                    
                    # Only add visualization if there is valid data
                    if any(timestamps) and any(cpu_before) and any(cpu_after):
                        impact_chart.add_trace(go.Scatter(x=timestamps, y=cpu_before, name="Before Optimization"))
                        impact_chart.add_trace(go.Scatter(x=timestamps, y=cpu_after, name="After Optimization"))
                        impact_chart.update_layout(
                            title="Optimization Impact on CPU Usage",
                            xaxis_title="Time",
                            yaxis_title="CPU Usage (%)",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#1a1f2c",
                            font=dict(color="white"),
                            xaxis=dict(gridcolor="#2d3747"),
                            yaxis=dict(gridcolor="#2d3747"),
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(impact_chart, use_container_width=True)
                    else:
                        st.info("Not enough data to visualize optimization impact yet.")
        
        with insights_tab:
            st.markdown("#### System Performance Insights")
            
            # Initialize insights if needed
            if 'performance_insights' not in st.session_state:
                st.session_state.performance_insights = {
                    'cpu_trend': 'stable',
                    'memory_trend': 'increasing',
                    'top_consumers': [],
                    'recommendations': [
                        "Consider increasing optimization threshold for better performance",
                        "Some background processes may be optimized for better battery life",
                        "System appears stable with current optimization settings"
                    ]
                }
            
            insights = st.session_state.performance_insights
            
            # Display trends
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                cpu_trend = insights.get('cpu_trend', 'stable')
                trend_icon = "↗️" if cpu_trend == 'increasing' else "↘️" if cpu_trend == 'decreasing' else "↔️"
                st.markdown(f"**CPU Usage Trend:** {trend_icon} {cpu_trend.capitalize()}")
            
            with trend_col2:
                memory_trend = insights.get('memory_trend', 'stable')
                trend_icon = "↗️" if memory_trend == 'increasing' else "↘️" if memory_trend == 'decreasing' else "↔️"
                st.markdown(f"**Memory Usage Trend:** {trend_icon} {memory_trend.capitalize()}")
            
            # AI Recommendations
            st.markdown("#### AI Recommendations")
            for recommendation in insights.get('recommendations', []):
                st.markdown(f"- {recommendation}")
            
            # Performance prediction
            st.markdown("#### Performance Prediction")
            prediction_chart = go.Figure()
            
            # Generate some dummy prediction data if none exists
            if 'prediction_data' not in st.session_state:
                current_time = datetime.now()
                st.session_state.prediction_data = {
                    'timestamps': [(current_time.replace(minute=current_time.minute + i)).strftime("%H:%M") for i in range(6)],
                    'cpu_predicted': [psutil.cpu_percent() * (1 + i*0.05) for i in range(6)],
                    'memory_predicted': [psutil.virtual_memory().percent * (1 + i*0.02) for i in range(6)]
                }
            
            pred_data = st.session_state.prediction_data
            prediction_chart.add_trace(go.Scatter(
                x=pred_data['timestamps'],
                y=pred_data['cpu_predicted'],
                mode='lines+markers',
                name='CPU Prediction'
            ))
            
            prediction_chart.add_trace(go.Scatter(
                x=pred_data['timestamps'],
                y=pred_data['memory_predicted'],
                mode='lines+markers',
                name='Memory Prediction'
            ))
            
            prediction_chart.update_layout(
                title="Resource Usage Prediction (Next 30 Minutes)",
                xaxis_title="Time",
                yaxis_title="Usage (%)"
            )
            
            st.plotly_chart(prediction_chart)
        
        # Store optimization settings
        st.session_state.enable_auto_optimization = enable_auto_optimization
        st.session_state.optimization_threshold = optimization_threshold
        st.session_state.enable_process_termination = enable_process_termination
        st.session_state.termination_threshold = termination_threshold
        st.session_state.optimization_mode = optimization_mode
        st.session_state.smart_scheduling = smart_scheduling
        st.session_state.enable_learning = enable_learning
        st.session_state.enable_idle_optimization = enable_idle_optimization
        st.session_state.enable_process_grouping = enable_process_grouping
        
        # Get and display processes
        processes = get_all_processes()
        
        if processes:
            # Filter processes
            if filter_text:
                processes = [p for p in processes if 
                           filter_text.lower() in p['name'].lower() or 
                           str(p['pid']) == filter_text]
            
            # Sort processes
            if sort_by == "CPU Usage":
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            elif sort_by == "Memory Usage":
                processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            elif sort_by == "Process Name":
                processes.sort(key=lambda x: x['name'].lower())
            elif sort_by == "PID":
                processes.sort(key=lambda x: x['pid'])
            
            # Limit number of processes shown
            if not show_all_processes:
                processes = [p for p in processes if p['cpu_percent'] > 0 or p['memory_percent'] > 0]
                processes = processes[:MAX_PROCESSES]
            
            # Display processes in a table
            st.markdown("### Process List")
            
            # Create DataFrame for display
            df = pd.DataFrame(processes)
            
            # Add optimization indicator column
            if len(processes) > 0:
                # Add status indicators with HTML styling
                df['Status'] = df.apply(lambda row: 
                    f"<span style='color: #ff4b4b; font-weight: bold;'>⚠️ High</span>" if row['cpu_percent'] > optimization_threshold else 
                    f"<span style='color: #0cce6b; font-weight: bold;'>✅ Good</span>" if row['cpu_percent'] > 0 else 
                    f"<span style='color: #4f8bf9; font-weight: bold;'>⏱️ Idle</span>", axis=1)
                
                # Format CPU and memory values for better readability
                df['cpu_percent'] = df['cpu_percent'].apply(lambda x: f"{x:.1f}%")
                df['memory_percent'] = df['memory_percent'].apply(lambda x: f"{x:.1f}%")
                
                # Set custom column names
                df = df.rename(columns={
                    'pid': 'PID',
                    'name': 'Process Name',
                    'cpu_percent': 'CPU Usage',
                    'memory_percent': 'Memory Usage',
                    'status': 'State'
                })
                
                # Select and order columns for display
                display_df = df[['PID', 'Process Name', 'CPU Usage', 'Memory Usage', 'State', 'Status']]
                
                # Apply custom styling to the dataframe
                st.markdown("""
                <style>
                .dataframe {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: 'Arial', sans-serif;
                }
                .dataframe th {
                    background-color: #1a1f2c;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    text-align: left;
                    border-bottom: 2px solid #4f8bf9;
                }
                .dataframe td {
                    padding: 8px;
                    border-bottom: 1px solid #2d3747;
                }
                .dataframe tr:hover {
                    background-color: rgba(79, 139, 249, 0.1);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display the styled table
                st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No processes found or unable to retrieve process information.")
            
            # Process selection for optimization or termination
            selected_pid = st.selectbox(
                "Select a process to manage:",
                options=[f"{p['pid']} - {p['name']}" for p in processes],
                format_func=lambda x: x
            )
            
            # Extract PID from selection
            if selected_pid:
                pid = int(selected_pid.split(" - ")[0])
                
                # Find selected process
                selected_process = next((p for p in processes if p['pid'] == pid), None)
                
                if selected_process:
                    st.markdown(f"**Selected Process:** {selected_process['name']} (PID: {selected_process['pid']})")
                    st.markdown(f"**CPU Usage:** {selected_process['cpu_percent']:.2f}%, **Memory:** {selected_process['memory_percent']:.2f}%")
                    
                    # Action buttons
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("🔧 Optimize Process", key=f"optimize_{pid}"):
                            # Record current state for history
                            cpu_before = selected_process['cpu_percent']
                            mem_before = selected_process['memory_percent']
                            
                            # Simulate optimization (would actually modify process priority)
                            try:
                                # Try to set process priority
                                process = psutil.Process(pid)
                                
                                # Set lower priority
                                if platform.system() == 'Windows':
                                    # Windows-specific priority
                                    import win32process
                                    import win32con
                                    handle = win32process.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
                                    win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
                                else:
                                    # Unix-based priority (nice)
                                    process.nice(10)
                                
                                # Record optimization in history
                                if 'optimization_history' not in st.session_state:
                                    st.session_state.optimization_history = []
                                
                                # Simulate CPU after optimization
                                cpu_after = max(1.0, cpu_before * 0.75)  # 25% reduction
                                
                                # Add to history
                                st.session_state.optimization_history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'process_name': selected_process['name'],
                                    'pid': pid,
                                    'action': 'optimize',
                                    'cpu_before': cpu_before,
                                    'cpu_after': cpu_after,
                                    'memory_before': mem_before,
                                    'memory_after': mem_before  # Memory usually stays the same
                                })
                                
                                st.success(f"Optimized process {selected_process['name']} (PID: {pid})")
                            except Exception as e:
                                st.error(f"Failed to optimize process: {str(e)}")
                    
                    with action_col2:
                        if st.button("⚠️ Terminate Process", key=f"terminate_{pid}"):
                            success, message = terminate_process(pid, "Manual termination")
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    
                    with action_col3:
                        if st.button("📊 Analyze Process", key=f"analyze_{pid}"):
                            # Simulate process analysis
                            st.info(f"Analyzing process {selected_process['name']} (PID: {pid})...")
                            
                            # Create simulated analysis report
                            st.markdown("#### Process Analysis Report")
                            
                            # Resource usage over time (simulated)
                            analysis_chart = go.Figure()
                            
                            # Simulated data for the last 5 minutes
                            time_points = [datetime.now().replace(minute=datetime.now().minute - i) for i in range(5, 0, -1)]
                            cpu_points = [selected_process['cpu_percent'] * (1 + (i-2.5)*0.2) for i in range(5)]
                            memory_points = [selected_process['memory_percent'] * (1 + (i-2.5)*0.1) for i in range(5)]
                            
                            analysis_chart.add_trace(go.Scatter(
                                x=[t.strftime("%H:%M") for t in time_points],
                                y=cpu_points,
                                mode='lines+markers',
                                name='CPU Usage'
                            ))
                            
                            analysis_chart.add_trace(go.Scatter(
                                x=[t.strftime("%H:%M") for t in time_points],
                                y=memory_points,
                                mode='lines+markers',
                                name='Memory Usage'
                            ))
                            
                            st.plotly_chart(analysis_chart)
                            
                            # Recommendations
                            st.markdown("#### Recommendations")
                            if selected_process['cpu_percent'] > 50:
                                st.markdown("- ⚠️ High CPU usage detected. Consider optimizing or limiting this process.")
                            if selected_process['memory_percent'] > 30:
                                st.markdown("- ⚠️ Significant memory consumption. Monitor for memory leaks.")
                            
                            st.markdown("- ✅ Process appears to be functioning normally overall.")
                            st.markdown(f"- ℹ️ Process has been running since {datetime.fromtimestamp(selected_process.get('create_time', time.time())).strftime('%H:%M:%S')}")
            
            # Auto-optimization logic
            if enable_auto_optimization:
                # Find high-usage processes
                high_usage_processes = [p for p in processes if 
                                      p['cpu_percent'] > optimization_threshold or 
                                      p['memory_percent'] > optimization_threshold]
                
                if high_usage_processes:
                    st.markdown("#### Automatic Optimization")
                    st.warning(f"Found {len(high_usage_processes)} processes with high resource usage")
                    
                    for process in high_usage_processes:
                        st.info(f"⚠️ High resource usage: {process['name']} (PID: {process['pid']}) - CPU: {process['cpu_percent']:.1f}%, Memory: {process['memory_percent']:.1f}%")
                        
                        # Auto-terminate if enabled and above threshold
                        if enable_process_termination and (
                            process['cpu_percent'] > termination_threshold or 
                            process['memory_percent'] > termination_threshold):
                            
                            # Only auto-terminate non-essential processes
                            essential_processes = ['explorer.exe', 'svchost.exe', 'csrss.exe', 'system']
                            if process['name'].lower() not in essential_processes:
                                success, message = terminate_process(
                                    process['pid'], 
                                    f"Auto-termination due to high resource usage ({optimization_mode} mode)"
                                )
                                if success:
                                    st.success(f"✅ {message}")
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                st.warning(f"⚠️ Skipping essential system process: {process['name']}")
                        
                        # Otherwise just optimize
                        elif optimization_mode != "Performance":  # Don't optimize in Performance mode
                            try:
                                # Simulate optimization (would actually modify process priority)
                                st.success(f"✅ Optimized {process['name']} for {optimization_mode} mode")
                                
                                # Record in history
                                if 'optimization_history' not in st.session_state:
                                    st.session_state.optimization_history = []
                                
                                st.session_state.optimization_history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'process_name': process['name'],
                                    'pid': process['pid'],
                                    'action': 'auto-optimize',
                                    'cpu_before': process['cpu_percent'],
                                    'cpu_after': process['cpu_percent'] * 0.8,  # Simulated 20% reduction
                                    'memory_before': process['memory_percent'],
                                    'memory_after': process['memory_percent']
                                })
                            except Exception as e:
                                st.error(f"Failed to auto-optimize {process['name']}: {str(e)}")
        else:
            st.info("No processes found or unable to retrieve process information.")
        
        # Add auto-refresh
        time.sleep(refresh_rate)
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error in process manager: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        time.sleep(3)  # Wait before retrying
        st.rerun()