"""
Performance Graphs Module for Arthashila

This module provides real-time performance monitoring with interactive graphs.
It displays CPU and memory usage over time with configurable refresh rates.

Features:
- Real-time CPU and memory monitoring
- Anomaly detection using machine learning
- Predictive resource usage forecasting
- Interactive controls for refresh rate and feature toggling
- Detailed anomaly alerts with severity classification and recommendations
"""

import streamlit as st
import psutil
import time
from collections import deque
import sys
import os
import numpy as np
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization function from utils
from utils import create_line_chart

# Constants
MAX_DATA_POINTS = 60  # Store up to 60 data points (1 minute at 1-second refresh)
DEFAULT_REFRESH_RATE = 2.0  # Default refresh rate in seconds
ANOMALY_THRESHOLD = -0.4  # Threshold for anomaly detection

def initialize_session_state():
    """
    Initialize the session state variables for storing time series data.
    Uses deque with a maximum length to automatically discard old data points.
    
    Also initializes the anomaly detection model if not already in session state.
    
    Returns:
        tuple: Tuple containing the CPU and memory data deques
    """
    try:
        # Initialize data storage if not already in session state
        if 'cpu_data' not in st.session_state:
            st.session_state.cpu_data = deque(maxlen=MAX_DATA_POINTS)
        
        if 'memory_data' not in st.session_state:
            st.session_state.memory_data = deque(maxlen=MAX_DATA_POINTS)
        
        # Initialize anomaly detection model if not already in session state
        if 'anomaly_model' not in st.session_state:
            st.session_state.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
            st.session_state.anomaly_model_trained = False
        
        # Initialize anomaly flags
        if 'anomaly_detected' not in st.session_state:
            st.session_state.anomaly_detected = False
            
        # Initialize prediction data
        if 'prediction_data' not in st.session_state:
            st.session_state.prediction_data = {'cpu': [], 'memory': []}
            
        # Initialize anomaly history if not in session state
        if 'anomaly_history' not in st.session_state:
            st.session_state.anomaly_history = []
            
        # Initialize anomaly details if not in session state
        if 'anomaly_details' not in st.session_state:
            st.session_state.anomaly_details = {}
            
        # Initialize anomaly scores if not in session state
        if 'anomaly_scores' not in st.session_state:
            st.session_state.anomaly_scores = deque(maxlen=MAX_DATA_POINTS)
            
        # Ensure compatibility with process_manager.py
        if 'refresh_rate' not in st.session_state:
            st.session_state.refresh_rate = float(DEFAULT_REFRESH_RATE)
        elif isinstance(st.session_state.refresh_rate, (list, tuple)):
            # Convert to float if it's a list or tuple
            st.session_state.refresh_rate = float(DEFAULT_REFRESH_RATE)
        else:
            # Ensure it's a float
            st.session_state.refresh_rate = float(st.session_state.refresh_rate)
            
        # Initialize training data if not exists - compatibility with process_manager
        if 'training_data' not in st.session_state:
            st.session_state.training_data = {
                'cpu_samples': [],
                'memory_samples': [],
                'process_samples': [],
                'training_rounds': 0,
                'last_training_time': datetime.now(),
                'accuracy_metrics': {'cpu': 0.0, 'memory': 0.0, 'process': 0.0},
                'calibration_factor': {'cpu': 1.0, 'memory': 1.0, 'process_cpu': 1.0}
            }
            
        return st.session_state.cpu_data, st.session_state.memory_data
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        # Return empty deques if initialization fails
        return deque(maxlen=MAX_DATA_POINTS), deque(maxlen=MAX_DATA_POINTS)

def update_performance_data():
    """
    Collect current system performance metrics and add them to the data deques.
    
    Also checks for anomalies if enough data is available and the anomaly detection
    feature is enabled.
    
    Returns:
        tuple: Current CPU and memory usage percentages
    """
    try:
        current_time = time.time()
        
        # Apply calibration factors from training data for consistency with process_manager
        cpu_calibration = st.session_state.training_data.get('calibration_factor', {}).get('cpu', 1.0)
        memory_calibration = st.session_state.training_data.get('calibration_factor', {}).get('memory', 1.0)
        
        # Get CPU and memory usage with calibration applied
        cpu_percent = psutil.cpu_percent(interval=0.1) * cpu_calibration  # Quick sampling
        memory_percent = psutil.virtual_memory().percent * memory_calibration
        
        # Add to data collections
        st.session_state.cpu_data.append((current_time, cpu_percent))
        st.session_state.memory_data.append((current_time, memory_percent))
        
        # Check for anomalies if we have enough data and the feature is enabled
        if len(st.session_state.cpu_data) >= 10 and st.session_state.get('enable_anomaly_detection', False):
            st.session_state.anomaly_detected = check_for_anomalies()
        
        # Add to training data (limited to prevent excessive memory usage)
        if len(st.session_state.training_data['cpu_samples']) < 1000:
            st.session_state.training_data['cpu_samples'].append(cpu_percent)
            st.session_state.training_data['memory_samples'].append(memory_percent)
            
        return cpu_percent, memory_percent
    except Exception as e:
        logger.error(f"Error updating performance data: {str(e)}")
        # Return default values if data collection fails
        return 0.0, 0.0

def check_for_anomalies():
    """
    Use Isolation Forest to detect anomalies in performance data.
    Enhanced with more visible indicators when anomalies are detected.
    
    This function:
    1. Prepares recent CPU and memory data for anomaly detection
    2. Trains the model if it hasn't been trained yet
    3. Calculates anomaly scores for the latest data points
    4. Returns True if an anomaly is detected based on the threshold
    
    Returns:
        bool: True if an anomaly is detected, False otherwise
    """
    try:
        # Prepare data for anomaly detection
        if len(st.session_state.cpu_data) < 10:
            return False
            
        cpu_values = np.array([v for _, v in st.session_state.cpu_data]).reshape(-1, 1)
        memory_values = np.array([v for _, v in st.session_state.memory_data]).reshape(-1, 1)
        
        # Combine features for more robust detection
        features = np.hstack((cpu_values, memory_values))
        
        # Standardize the data
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train the model if not already trained and we have enough data
        if not st.session_state.anomaly_model_trained and len(st.session_state.cpu_data) >= 15:
            st.session_state.anomaly_model.fit(features_scaled)
            st.session_state.anomaly_model_trained = True
            st.session_state.last_training_time = time.time()
        
        # Retrain model periodically to adapt to changing system behavior
        elif st.session_state.anomaly_model_trained:
            time_since_training = time.time() - st.session_state.get('last_training_time', 0)
            if time_since_training > 300:  # Retrain every 5 minutes
                st.session_state.anomaly_model.fit(features_scaled)
                st.session_state.last_training_time = time.time()
        
        # Detect anomalies if model is trained
        anomaly_details = {'detected': False, 'score': 0, 'cpu': 0, 'memory': 0}
        
        if st.session_state.anomaly_model_trained:
            # Predict anomaly scores (-1 for anomalies, 1 for normal data)
            scores = st.session_state.anomaly_model.decision_function(features_scaled)
            
            # Store all scores for visualization
            if 'anomaly_scores' not in st.session_state:
                st.session_state.anomaly_scores = deque(maxlen=MAX_DATA_POINTS)
            
            # Add current score with timestamp
            st.session_state.anomaly_scores.append((time.time(), scores[-1]))
            
            # Check if the latest data point is anomalous
            if scores[-1] < ANOMALY_THRESHOLD:
                anomaly_details = {
                    'detected': True,
                    'score': scores[-1],
                    'cpu': cpu_values[-1][0],
                    'memory': memory_values[-1][0],
                    'time': time.time()
                }
                
                # Store anomaly in session state for history
                if 'anomaly_history' not in st.session_state:
                    st.session_state.anomaly_history = []
                
                # Only store last 10 anomalies
                if len(st.session_state.anomaly_history) >= 10:
                    st.session_state.anomaly_history.pop(0)
                
                st.session_state.anomaly_history.append(anomaly_details)
                return True
        
        # Update session state with latest details
        st.session_state.anomaly_details = anomaly_details
        return False
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return False

def predict_future_usage(steps=10):
    """
    Predict future CPU and memory usage based on recent trends.
    
    Uses a simple exponential moving average with trend calculation to forecast
    future resource usage. This is more lightweight than complex ML models.
    
    Args:
        steps (int): Number of time steps to predict into the future
        
    Returns:
        dict: Predicted values for CPU and memory in a dictionary with keys 'cpu' and 'memory'
    """
    try:
        if len(st.session_state.cpu_data) < 10:
            return None
        
        # Extract recent values
        recent_cpu = [v for _, v in list(st.session_state.cpu_data)[-10:]]
        recent_memory = [v for _, v in list(st.session_state.memory_data)[-10:]]
        
        # Simple exponential moving average prediction
        cpu_weight = 0.7
        memory_weight = 0.7
        
        cpu_predictions = []
        memory_predictions = []
        
        # Start with last actual values
        last_cpu = recent_cpu[-1]
        last_memory = recent_memory[-1]
        
        for _ in range(steps):
            # Calculate trend based on last few points
            cpu_trend = np.mean(np.diff(recent_cpu[-5:]) if len(recent_cpu) >= 5 else [0])
            memory_trend = np.mean(np.diff(recent_memory[-5:]) if len(recent_memory) >= 5 else [0])
            
            # Predict next value with dampening
            next_cpu = max(0, min(100, last_cpu + cpu_trend * cpu_weight))
            next_memory = max(0, min(100, last_memory + memory_trend * memory_weight))
            
            cpu_predictions.append(next_cpu)
            memory_predictions.append(next_memory)
            
            # Update for next iteration
            last_cpu = next_cpu
            last_memory = next_memory
            
            # Reduce weight for further predictions
            cpu_weight *= 0.9
            memory_weight *= 0.9
        
        # Store predictions
        st.session_state.prediction_data = {
            'cpu': cpu_predictions,
            'memory': memory_predictions
        }
        
        return st.session_state.prediction_data
    except Exception as e:
        logger.error(f"Error predicting future usage: {str(e)}")
        # Return empty predictions if forecasting fails
        return {'cpu': [], 'memory': []}

def render_performance_controls():
    """
    Render the control panel for performance monitoring settings.
    
    This function creates UI controls for:
    - Adjusting the refresh rate
    - Enabling/disabling AI anomaly detection
    - Enabling/disabling predictive analytics
    
    Returns:
        tuple: Selected refresh rate, anomaly detection flag, and predictions flag
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # Add refresh rate selector - ensuring it's float to avoid errors
        try:
            refresh_rate = float(st.slider(
                "Refresh Rate (seconds)", 
                min_value=1.0, 
                max_value=10.0, 
                value=float(st.session_state.get('refresh_rate', DEFAULT_REFRESH_RATE)),
                step=0.5,
                help="How frequently to update the performance graphs"
            ))
        except (ValueError, TypeError):
            refresh_rate = DEFAULT_REFRESH_RATE
            st.warning("Error with refresh rate, using default value.")
    
    with col2:
        # AI settings
        enable_anomaly_detection = st.checkbox(
            "Enable AI Anomaly Detection", 
            value=True,
            help="Use machine learning to detect unusual system behavior"
        )
        
        enable_predictions = st.checkbox(
            "Show Predictive Trends", 
            value=True,
            help="Forecast future resource usage based on current trends"
        )
    
    # Store refresh rate in session state for use in other modules
    st.session_state.refresh_rate = float(refresh_rate)
    
    return refresh_rate, enable_anomaly_detection, enable_predictions

def render_performance_graphs(cpu_data, memory_data):
    """
    Render the CPU and memory usage graphs side by side with enhanced visuals.
    
    If predictions are enabled, it also overlays the future predicted values
    on the graphs with a dashed line.
    
    Args:
        cpu_data (deque): Collection of CPU usage data points
        memory_data (deque): Collection of memory usage data points
    """
    try:
        # Create two columns for CPU and Memory graphs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CPU Usage")
            
            # Create the base line chart
            cpu_fig = create_line_chart(
                list(cpu_data),
                "CPU Usage Over Time",
                "Time",
                "CPU %"
            )
            
            # Add prediction overlay if enabled and predictions exist
            if st.session_state.get('enable_predictions', False) and st.session_state.prediction_data['cpu']:
                # Create x values for predictions (future time points)
                if cpu_data:
                    last_time = cpu_data[-1][0]
                    time_step = 2  # seconds between predictions
                    future_times = [last_time + (i+1)*time_step for i in range(len(st.session_state.prediction_data['cpu']))]
                    
                    # Add prediction trace
                    cpu_fig.add_trace(
                        go.Scatter(
                            x=future_times,
                            y=st.session_state.prediction_data['cpu'],
                            mode="lines+markers",
                            line=dict(color="rgba(255, 165, 0, 0.8)", dash="dash"),
                            name="Predicted CPU",
                            marker=dict(symbol="circle-open")
                        )
                    )
            
            # Mark anomalies on the graph if any are detected and we have anomaly history
            if hasattr(st.session_state, 'anomaly_history') and st.session_state.anomaly_history:
                # Extract times and values for anomalies that are within our data window
                min_time = cpu_data[0][0] if cpu_data else 0
                max_time = cpu_data[-1][0] if cpu_data else float('inf')
                
                anomaly_times = []
                anomaly_values = []
                
                for anomaly in st.session_state.anomaly_history:
                    if min_time <= anomaly.get('time', 0) <= max_time:
                        anomaly_times.append(anomaly.get('time', 0))
                        anomaly_values.append(anomaly.get('cpu', 0))
                
                if anomaly_times:
                    # Add red markers for anomalies
                    cpu_fig.add_trace(
                        go.Scatter(
                            x=anomaly_times,
                            y=anomaly_values,
                            mode="markers",
                            marker=dict(
                                color="red",
                                size=10,
                                symbol="x",
                                line=dict(width=2, color="red")
                            ),
                            name="Anomalies"
                        )
                    )
            
            # Add hover annotations for better real-time monitoring
            cpu_fig.update_layout(
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
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
            
            # Add prediction overlay if enabled and predictions exist
            if st.session_state.get('enable_predictions', False) and st.session_state.prediction_data['memory']:
                # Create x values for predictions (future time points)
                if memory_data:
                    last_time = memory_data[-1][0]
                    time_step = 2  # seconds between predictions
                    future_times = [last_time + (i+1)*time_step for i in range(len(st.session_state.prediction_data['memory']))]
                    
                    # Add prediction trace
                    memory_fig.add_trace(
                        go.Scatter(
                            x=future_times,
                            y=st.session_state.prediction_data['memory'],
                            mode="lines+markers",
                            line=dict(color="rgba(255, 165, 0, 0.8)", dash="dash"),
                            name="Predicted Memory",
                            marker=dict(symbol="circle-open")
                        )
                    )
            
            # Mark anomalies on the graph if any are detected
            if hasattr(st.session_state, 'anomaly_history') and st.session_state.anomaly_history:
                # Extract times and values for anomalies that are within our data window
                min_time = memory_data[0][0] if memory_data else 0
                max_time = memory_data[-1][0] if memory_data else float('inf')
                
                anomaly_times = []
                anomaly_values = []
                
                for anomaly in st.session_state.anomaly_history:
                    if min_time <= anomaly.get('time', 0) <= max_time:
                        anomaly_times.append(anomaly.get('time', 0))
                        anomaly_values.append(anomaly.get('memory', 0))
                
                if anomaly_times:
                    # Add red markers for anomalies
                    memory_fig.add_trace(
                        go.Scatter(
                            x=anomaly_times,
                            y=anomaly_values,
                            mode="markers",
                            marker=dict(
                                color="red",
                                size=10,
                                symbol="x",
                                line=dict(width=2, color="red")
                            ),
                            name="Anomalies"
                        )
                    )
            
            # Add hover annotations
            memory_fig.update_layout(
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            st.plotly_chart(memory_fig, use_container_width=True)
        
        # If anomalies are detected, show a detailed anomaly section
        if st.session_state.get('anomaly_detected', False):
            with st.expander("🚨 Anomaly Detection Details", expanded=True):
                st.markdown("### System Anomalies Detected")
                
                # Get the anomaly details
                anomaly_details = st.session_state.get('anomaly_details', {})
                
                if anomaly_details:
                    anomaly_score = anomaly_details.get('score', 0)
                    severity = "High" if anomaly_score < -0.6 else "Medium" if anomaly_score < -0.4 else "Low"
                    severity_color = "#ff0000" if severity == "High" else "#ff9900" if severity == "Medium" else "#ffcc00"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: rgba(255, 0, 0, 0.1); border-left: 5px solid {severity_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-weight: bold; font-size: 16px;">Anomaly Detected</div>
                                <div>CPU: {anomaly_details.get('cpu', 0):.1f}%, Memory: {anomaly_details.get('memory', 0):.1f}%</div>
                            </div>
                            <div>
                                <span style="font-size: 14px; padding: 3px 8px; background-color: {severity_color}; color: white; border-radius: 3px;">
                                    {severity} Severity
                                </span>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div style="font-size: 14px;">Anomaly score: {anomaly_score:.4f}</div>
                            <div style="font-size: 14px;">Time: {datetime.fromtimestamp(anomaly_details.get('time', time.time())).strftime('%H:%M:%S')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Potential causes and recommended actions
                    st.markdown("#### Potential Causes")
                    causes = [
                        "Unusual process activity or resource consumption",
                        "Background system operations",
                        "Malware or unwanted applications",
                        "System service issues"
                    ]
                    
                    for cause in causes:
                        st.markdown(f"- {cause}")
                    
                    st.markdown("#### Recommended Actions")
                    actions = [
                        "Check the Process Manager for resource-intensive processes",
                        "Run a system scan for malware",
                        "Review recently installed applications",
                        "Monitor system for recurring patterns"
                    ]
                    
                    for action in actions:
                        st.markdown(f"- {action}")
                
                # Show anomaly history if available
                if hasattr(st.session_state, 'anomaly_history') and st.session_state.anomaly_history:
                    st.markdown("#### Recent Anomaly History")
                    
                    # Create a DataFrame for the anomaly history
                    history_data = []
                    for anomaly in st.session_state.anomaly_history:
                        history_data.append({
                            "Time": datetime.fromtimestamp(anomaly.get('time', 0)).strftime('%H:%M:%S'),
                            "CPU %": f"{anomaly.get('cpu', 0):.1f}%",
                            "Memory %": f"{anomaly.get('memory', 0):.1f}%",
                            "Score": f"{anomaly.get('score', 0):.4f}"
                        })
                    
                    # Display as a table
                    st.table(pd.DataFrame(history_data))
    except Exception as e:
        logger.error(f"Error rendering performance graphs: {str(e)}")
        st.error(f"Error rendering graphs: {str(e)}")

def performance_graphs():
    """
    Main entry point for the performance graphs feature.
    
    This function:
    1. Initializes data storage
    2. Updates with current performance data
    3. Renders control panel and graphs
    4. Enables AI features like anomaly detection and predictions
    5. Refreshes the display at the specified interval
    """
    try:
        st.title("📊 AI-Powered Performance Analytics")
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="flex-grow: 1;">Monitor system performance with AI-enhanced anomaly detection and predictive analytics</div>
            <div class="real-time-badge">🔄 REAL-TIME</div>
        </div>
        
        <style>
        .real-time-badge {
            background-color: #0cce6b;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            display: inline-block;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state for data storage
        cpu_data, memory_data = initialize_session_state()
        
        # Render control panel and get settings
        refresh_rate, enable_anomaly_detection, enable_predictions = render_performance_controls()
        
        # Save settings to session state
        st.session_state.enable_anomaly_detection = enable_anomaly_detection
        st.session_state.enable_predictions = enable_predictions
        
        # Add current performance data
        cpu_current, memory_current = update_performance_data()
        
        # Generate predictions if enabled
        if enable_predictions and len(cpu_data) >= 10:
            predict_future_usage(steps=10)
        
        # Display current metrics above the graphs
        col1, col2 = st.columns(2)
        
        # Check for anomalies and display appropriate indicators
        if st.session_state.anomaly_detected and enable_anomaly_detection:
            col1.metric("Current CPU Usage", f"{cpu_current:.1f}%", delta="⚠️ Unusual", delta_color="inverse", help="Unusual behavior detected")
            col2.metric("Current Memory Usage", f"{memory_current:.1f}%", delta="⚠️ Unusual", delta_color="inverse", help="Unusual behavior detected")
            st.warning("⚠️ Anomalous system behavior detected! This may indicate unusual processes or potential security issues.")
        else:
            col1.metric("Current CPU Usage", f"{cpu_current:.1f}%")
            col2.metric("Current Memory Usage", f"{memory_current:.1f}%")
        
        # Display graphs
        render_performance_graphs(cpu_data, memory_data)
        
        # Add information about the graphs
        with st.expander("About AI-Enhanced Performance Monitoring"):
            st.markdown("""
            ### Understanding Performance Metrics
            
            **CPU Usage**: Shows the percentage of CPU resources currently in use. High sustained 
            CPU usage can indicate processing bottlenecks or resource-intensive applications.
            
            **Memory Usage**: Shows the percentage of RAM currently in use. Memory that stays 
            consistently high may indicate memory leaks or insufficient RAM for your workload.
            
            ### AI-Powered Features
            
            **Anomaly Detection**: Uses Isolation Forest machine learning algorithm to identify unusual 
            system behavior based on historical patterns. This can help detect potential security issues 
            or system problems before they become critical.
            
            **Predictive Analytics**: Analyzes recent performance trends to forecast future resource usage, 
            helping you plan for scaling or optimization.
            
            The graphs show data for the last {0} data points. The refresh rate determines 
            how frequently new data points are collected.
            """.format(MAX_DATA_POINTS))
        
        # Add real-time timestamp display
        st.markdown(f"""
        <div style="text-align: right; color: #888; font-size: 0.8em; margin-top: 20px;">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}
        </div>
        """, unsafe_allow_html=True)
        
        # Add auto-refresh with the specified rate
        if refresh_rate > 0:
            time.sleep(max(0.1, refresh_rate))  # Minimum 0.1 seconds to avoid excessive reloading
            st.rerun()
    except Exception as e:
        logger.error(f"Error in performance_graphs main function: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        
        # Still attempt to refresh to recover from errors
        time.sleep(3)  # Wait longer after an error
        st.rerun()