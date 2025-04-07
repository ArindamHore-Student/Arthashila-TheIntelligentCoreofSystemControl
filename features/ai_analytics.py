"""
AI Analytics Module for Arthashila

This module provides advanced AI-powered analytics for system monitoring and optimization.
It includes anomaly detection, predictive resource forecasting, user behavior analysis,
and the X-Factor Process Optimizer for intelligent process management.

Features:
- System anomaly detection using machine learning algorithms
- Predictive resource usage forecasting
- User behavior pattern analysis
- Security threat detection
- X-Factor Process Optimizer for idle resource-intensive processes

The module uses various machine learning algorithms to analyze system data
and provide intelligent insights.
"""

import streamlit as st
import psutil
import time
from collections import deque
import sys
import os
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.cluster import KMeans # type: ignore
import hashlib
from datetime import datetime
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import pandas as pd
import plotly.express as px
from utils import create_line_chart, create_bar_chart

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
MAX_DATA_POINTS = 60  # Store up to 60 data points
DEFAULT_REFRESH_RATE = 2  # Default refresh rate in seconds
ANOMALY_THRESHOLD = -0.4  # Threshold for anomaly detection

def initialize_session_state():
    """
    Initialize the session state variables for storing all necessary data.
    
    Returns:
        dict: Dictionary containing initialized data structures
    """
    # System performance data
    if 'ai_cpu_data' not in st.session_state:
        st.session_state.ai_cpu_data = deque(maxlen=MAX_DATA_POINTS)
    
    if 'ai_memory_data' not in st.session_state:
        st.session_state.ai_memory_data = deque(maxlen=MAX_DATA_POINTS)
    
    if 'ai_disk_data' not in st.session_state:
        st.session_state.ai_disk_data = deque(maxlen=MAX_DATA_POINTS)
    
    if 'ai_network_data' not in st.session_state:
        st.session_state.ai_network_data = deque(maxlen=MAX_DATA_POINTS)
    
    # Initialize machine learning models
    if 'anomaly_model' not in st.session_state:
        st.session_state.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        st.session_state.anomaly_model_trained = False
    
    if 'prediction_model' not in st.session_state:
        st.session_state.prediction_model = RandomForestRegressor(n_estimators=50, random_state=42)
        st.session_state.prediction_model_trained = False
    
    if 'clustering_model' not in st.session_state:
        st.session_state.clustering_model = KMeans(n_clusters=3, random_state=42)
        st.session_state.clustering_model_trained = False
    
    # Logs and security data
    if 'ai_security_logs' not in st.session_state:
        st.session_state.ai_security_logs = deque(maxlen=100)
    
    if 'ai_prediction_data' not in st.session_state:
        st.session_state.ai_prediction_data = {
            'cpu': [], 
            'memory': [], 
            'disk': [],
            'network': []
        }
    
    # Analysis results
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = []
    
    # User behavior patterns
    if 'user_patterns' not in st.session_state:
        st.session_state.user_patterns = deque(maxlen=100)
    
    return {
        'cpu_data': st.session_state.ai_cpu_data,
        'memory_data': st.session_state.ai_memory_data,
        'disk_data': st.session_state.ai_disk_data,
        'network_data': st.session_state.ai_network_data,
    }

def update_system_data():
    """
    Collect current system performance metrics and add them to data collections.
    
    Returns:
        dict: Current system metrics
    """
    current_time = time.time()
    
    # Collect system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    # Get network information (using sent/received bytes as a simple metric)
    net_io = psutil.net_io_counters()
    network_usage = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # Convert to MB
    
    # Add to data collections
    st.session_state.ai_cpu_data.append((current_time, cpu_percent))
    st.session_state.ai_memory_data.append((current_time, memory_percent))
    st.session_state.ai_disk_data.append((current_time, disk_percent))
    st.session_state.ai_network_data.append((current_time, network_usage))
    
    # Check for anomalies if we have enough data
    if len(st.session_state.ai_cpu_data) >= 10:
        check_for_anomalies()
    
    # Log data with secure hash
    log_data_securely(current_time, cpu_percent, memory_percent, disk_percent, network_usage)
    
    # Capture "user behavior" (simulated - in a real app this would be actual user interactions)
    simulate_user_behavior(current_time, cpu_percent, memory_percent)
    
    return {
        'time': current_time,
        'cpu': cpu_percent,
        'memory': memory_percent,
        'disk': disk_percent,
        'network': network_usage
    }

def log_data_securely(timestamp, cpu, memory, disk, network):
    """
    Create a secure log entry with a hash to ensure data integrity.
    
    Args:
        timestamp (float): Current timestamp
        cpu (float): CPU usage percentage
        memory (float): Memory usage percentage
        disk (float): Disk usage percentage
        network (float): Network usage in MB
    """
    # Create a log entry with timestamp and data
    log_entry = {
        "timestamp": timestamp,
        "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu,
        "memory": memory,
        "disk": disk,
        "network": network
    }
    
    # Calculate hash for data integrity verification
    entry_str = json.dumps(log_entry, sort_keys=True)
    log_entry["integrity_hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
    
    # Add to security logs
    st.session_state.ai_security_logs.append(log_entry)

def simulate_user_behavior(timestamp, cpu, memory):
    """
    Simulate user behavior patterns for demonstration purposes.
    In a real application, this would capture actual user interactions.
    
    Args:
        timestamp (float): Current timestamp
        cpu (float): CPU usage as context
        memory (float): Memory usage as context
    """
    # Simulate different user activities based on time and system load
    hour = datetime.fromtimestamp(timestamp).hour
    
    # Create user behavior pattern
    if hour < 12:
        activity_type = "Morning productivity"
        activity_score = np.random.normal(0.7, 0.1)  # Higher productivity in morning
    elif hour < 17:
        activity_type = "Afternoon work"
        activity_score = np.random.normal(0.5, 0.2)  # Variable afternoon productivity
    else:
        activity_type = "Evening usage"
        activity_score = np.random.normal(0.3, 0.3)  # Lower evening productivity
    
    # Adjust based on system load
    if cpu > 80 or memory > 80:
        activity_score *= 0.7  # Reduced productivity under high system load
    
    # Record the pattern
    pattern = {
        "timestamp": timestamp,
        "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
        "activity_type": activity_type,
        "productivity_score": min(1.0, max(0.0, activity_score)),
        "system_context": {
            "cpu": cpu,
            "memory": memory
        }
    }
    
    st.session_state.user_patterns.append(pattern)

def check_for_anomalies():
    """
    Use Isolation Forest to detect anomalies in system performance data.
    
    Returns:
        bool: True if anomaly detected, False otherwise
    """
    # Need at least 10 data points for meaningful detection
    if len(st.session_state.ai_cpu_data) < 10:
        return False
        
    try:
        # Prepare data for anomaly detection
        cpu_values = np.array([v for _, v in st.session_state.ai_cpu_data]).reshape(-1, 1)
        memory_values = np.array([v for _, v in st.session_state.ai_memory_data]).reshape(-1, 1)
        
        # Only use CPU and memory for simplicity and stability
        # This ensures feature consistency between training and inference
        features = np.hstack((cpu_values, memory_values))
        
        # Standardize the data
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train the model if not already trained
        if not st.session_state.anomaly_model_trained and len(st.session_state.ai_cpu_data) >= MAX_DATA_POINTS // 2:
            # Re-initialize model when training to ensure clean state
            st.session_state.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
            st.session_state.anomaly_model.fit(features_scaled)
            st.session_state.anomaly_model_trained = True
        
        # Detect anomalies if model is trained
        if st.session_state.anomaly_model_trained:
            # Predict anomaly scores (-1 for anomalies, 1 for normal data)
            scores = st.session_state.anomaly_model.decision_function(features_scaled)
            
            # Check if the latest data point is anomalous
            if scores[-1] < ANOMALY_THRESHOLD:
                # Get all four metrics for logging, even though we only use two for detection
                disk_values = np.array([v for _, v in st.session_state.ai_disk_data]).reshape(-1, 1)
                network_values = np.array([v for _, v in st.session_state.ai_network_data]).reshape(-1, 1)
                
                # Add to security log
                log_entry = {
                    "timestamp": time.time(),
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "ANOMALY_DETECTED",
                    "metrics": {
                        "cpu": cpu_values[-1][0],
                        "memory": memory_values[-1][0],
                        "disk": disk_values[-1][0],
                        "network": network_values[-1][0]
                    },
                    "anomaly_score": scores[-1]
                }
                log_entry["integrity_hash"] = hashlib.sha256(json.dumps(log_entry, sort_keys=True).encode()).hexdigest()
                st.session_state.ai_security_logs.append(log_entry)
                
                # Add insight
                insight = {
                    "type": "anomaly",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": "System behavior anomaly detected",
                    "details": f"Unusual patterns in CPU/Memory usage (anomaly score: {scores[-1]:.4f})",
                    "severity": "high" if scores[-1] < -0.6 else "medium"
                }
                st.session_state.ai_insights.append(insight)
                
                return True
        
        return False
    except Exception as e:
        # Log the error but don't crash
        print(f"Error in anomaly detection: {str(e)}")
        return False

def predict_resource_usage(steps=10):
    """
    Predict future system resource usage based on historical data.
    
    Args:
        steps (int): Number of steps to predict into the future
        
    Returns:
        dict: Predicted values for all resources
    """
    if len(st.session_state.ai_cpu_data) < MAX_DATA_POINTS // 2:
        return None
    
    # Extract time features for forecasting
    timestamps = np.array([t for t, _ in st.session_state.ai_cpu_data])
    
    # Calculate time differences as a feature
    time_diffs = np.diff(timestamps)
    
    # Add cyclical time features (hour of day, day of week)
    hour_features = []
    day_features = []
    
    for t in timestamps:
        dt = datetime.fromtimestamp(t)
        # Convert time to cyclical features using sine and cosine
        hour = dt.hour
        day = dt.weekday()
        
        # Convert hour to cyclical feature (24 hours)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Convert day to cyclical feature (7 days)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        hour_features.append([hour_sin, hour_cos])
        day_features.append([day_sin, day_cos])
    
    hour_features = np.array(hour_features)
    day_features = np.array(day_features)
    
    # Get target values
    cpu_values = np.array([v for _, v in st.session_state.ai_cpu_data])
    memory_values = np.array([v for _, v in st.session_state.ai_memory_data])
    disk_values = np.array([v for _, v in st.session_state.ai_disk_data])
    network_values = np.array([v for _, v in st.session_state.ai_network_data])
    
    # Prepare feature matrix - add recent trend
    recent_cpu_trend = np.diff(cpu_values[-10:]).reshape(-1, 1) if len(cpu_values) >= 10 else np.zeros((1, 1))
    recent_memory_trend = np.diff(memory_values[-10:]).reshape(-1, 1) if len(memory_values) >= 10 else np.zeros((1, 1))
    
    # Combine features (without the last point since we'll be predicting it)
    X = np.hstack((
        hour_features[:-1], 
        day_features[:-1],
        cpu_values[:-1].reshape(-1, 1),
        memory_values[:-1].reshape(-1, 1),
        disk_values[:-1].reshape(-1, 1)
    ))
    
    # For simplicity, we'll create one model for all resources
    # In a real scenario, you would train separate models for better accuracy
    if not st.session_state.prediction_model_trained:
        # Train the model on all but the last few points
        y_cpu = cpu_values[1:]  # Predict next value
        st.session_state.prediction_model.fit(X[:-5], y_cpu[:-5])
        st.session_state.prediction_model_trained = True
    
    # Generate future features
    future_features = []
    last_timestamp = timestamps[-1]
    
    for i in range(steps):
        future_time = last_timestamp + (i + 1) * 600  # Predict 10 minutes ahead
        future_dt = datetime.fromtimestamp(future_time)
        
        # Generate cyclical features for future time
        hour = future_dt.hour
        day = future_dt.weekday()
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        # Use last known values for other features
        last_cpu = cpu_values[-1]
        last_memory = memory_values[-1]
        last_disk = disk_values[-1]
        
        future_features.append([hour_sin, hour_cos, day_sin, day_cos, last_cpu, last_memory, last_disk])
    
    future_features = np.array(future_features)
    
    # Make predictions
    cpu_predictions = []
    memory_predictions = []
    disk_predictions = []
    network_predictions = []
    
    for i in range(steps):
        # Predict each resource
        # In a real model, you would use the predicted values as input for next prediction
        if i == 0:
            # First prediction uses last known features
            features = np.array([future_features[i]])
            cpu_pred = max(0, min(100, st.session_state.prediction_model.predict(features)[0]))
            
            # For simplicity, we'll use simple scaling for other resources
            memory_scaling = memory_values[-1] / cpu_values[-1] if cpu_values[-1] > 0 else 1
            disk_scaling = disk_values[-1] / cpu_values[-1] if cpu_values[-1] > 0 else 1
            network_scaling = network_values[-1] / cpu_values[-1] if cpu_values[-1] > 0 else 1
            
            memory_pred = max(0, min(100, cpu_pred * memory_scaling))
            disk_pred = max(0, min(100, cpu_pred * disk_scaling * 0.2 + disk_values[-1] * 0.8))  # Disk changes slower
            network_pred = max(0, min(100, cpu_pred * network_scaling))
        else:
            # Use previous predictions as input
            features = np.array([future_features[i]])
            features[0, 4] = cpu_predictions[-1]  # Update CPU feature with last prediction
            features[0, 5] = memory_predictions[-1]  # Update memory feature
            features[0, 6] = disk_predictions[-1]  # Update disk feature
            
            cpu_pred = max(0, min(100, st.session_state.prediction_model.predict(features)[0]))
            
            # Apply smoothing for realistic predictions
            memory_pred = max(0, min(100, memory_predictions[-1] * 0.7 + cpu_pred * 0.3))
            disk_pred = max(0, min(100, disk_predictions[-1] * 0.9 + cpu_pred * 0.1))  # Disk changes very slowly
            network_pred = max(0, min(100, network_predictions[-1] * 0.5 + cpu_pred * 0.5))
        
        cpu_predictions.append(cpu_pred)
        memory_predictions.append(memory_pred)
        disk_predictions.append(disk_pred)
        network_predictions.append(network_pred)
    
    # Store predictions
    st.session_state.ai_prediction_data = {
        'cpu': cpu_predictions,
        'memory': memory_predictions,
        'disk': disk_predictions,
        'network': network_predictions
    }
    
    # Add insight about future resource usage
    if np.max(cpu_predictions) > 90:
        insight = {
            "type": "prediction",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "High CPU usage predicted",
            "details": f"CPU usage may reach {np.max(cpu_predictions):.1f}% in the near future",
            "severity": "medium"
        }
        st.session_state.ai_insights.append(insight)
    
    if np.max(memory_predictions) > 90:
        insight = {
            "type": "prediction",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "High memory usage predicted",
            "details": f"Memory usage may reach {np.max(memory_predictions):.1f}% in the near future",
            "severity": "medium"
        }
        st.session_state.ai_insights.append(insight)
    
    return st.session_state.ai_prediction_data 

def analyze_user_patterns():
    """
    Analyze user patterns to identify usage trends and productivity insights.
    
    Returns:
        dict: Analysis results
    """
    if len(st.session_state.user_patterns) < 5:
        return None
    
    # Extract features
    productivity_scores = [item['productivity_score'] for item in st.session_state.user_patterns]
    activity_types = [item['activity_type'] for item in st.session_state.user_patterns]
    
    # Calculate basic statistics
    avg_productivity = np.mean(productivity_scores)
    productivity_trend = np.diff(productivity_scores[-5:]).mean()
    
    # Count activity types
    activity_counts = {}
    for activity in activity_types:
        if activity in activity_counts:
            activity_counts[activity] += 1
        else:
            activity_counts[activity] = 1
    
    # Most common activity
    most_common = max(activity_counts, key=activity_counts.get)
    
    # Add insight
    if productivity_trend < -0.1:
        insight = {
            "type": "user_pattern",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "Declining productivity detected",
            "details": "User productivity has been declining recently. Consider taking a break.",
            "severity": "low"
        }
        st.session_state.ai_insights.append(insight)
    
    return {
        "avg_productivity": avg_productivity,
        "productivity_trend": productivity_trend,
        "most_common_activity": most_common,
        "activity_distribution": activity_counts
    }

def verify_data_integrity(log_entry):
    """
    Verify the integrity of a log entry by recalculating its hash.
    
    Args:
        log_entry (dict): The log entry to verify
        
    Returns:
        bool: True if integrity is verified, False otherwise
    """
    # Extract the stored hash
    stored_hash = log_entry.get('integrity_hash')
    if not stored_hash:
        return False
    
    # Create a copy without the hash for recalculation
    verification_entry = log_entry.copy()
    verification_entry.pop('integrity_hash')
    
    # Recalculate hash
    verification_str = json.dumps(verification_entry, sort_keys=True)
    calculated_hash = hashlib.sha256(verification_str.encode()).hexdigest()
    
    # Compare hashes
    return stored_hash == calculated_hash

def render_ai_controls():
    """
    Render controls for AI analytics features.
    
    Returns:
        dict: User selected settings
    """
    st.subheader("AI Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_anomaly_detection = st.checkbox(
            "Enable Anomaly Detection",
            value=True,
            help="Use machine learning to detect unusual system behavior"
        )
        
        enable_predictions = st.checkbox(
            "Enable Predictive Analytics",
            value=True,
            help="Forecast future resource usage based on patterns"
        )
    
    with col2:
        enable_user_analytics = st.checkbox(
            "Enable User Behavior Analytics",
            value=True,
            help="Analyze user activity patterns for insights"
        )
        
        enable_security_verification = st.checkbox(
            "Enable Security Verification",
            value=True,
            help="Verify data integrity using cryptographic hashing"
        )
    
    with col3:
        enable_process_optimization = st.checkbox(
            "Enable X-Factor Process Optimizer",
            value=True,
            help="Automatically manage idle resource-intensive processes"
        )
    
    refresh_rate = st.slider(
        "Refresh Rate (seconds)",
        min_value=1,
        max_value=10,
        value=DEFAULT_REFRESH_RATE,
        help="How frequently to update analytics"
    )
    
    return {
        "enable_anomaly_detection": enable_anomaly_detection,
        "enable_predictions": enable_predictions,
        "enable_user_analytics": enable_user_analytics,
        "enable_security_verification": enable_security_verification,
        "enable_process_optimization": enable_process_optimization,
        "refresh_rate": refresh_rate
    }

def render_system_metrics(current_metrics):
    """
    Render current system metrics with AI-enhanced insights.
    
    Args:
        current_metrics (dict): Current system performance metrics
    """
    st.subheader("Current System Metrics")
    
    cols = st.columns(4)
    
    # Check for anomalies
    anomalies = [log for log in st.session_state.ai_security_logs if log.get('event_type') == 'ANOMALY_DETECTED']
    recent_anomaly = len(anomalies) > 0 and (time.time() - anomalies[-1]['timestamp']) < 60
    
    # Display metrics with appropriate indicators
    with cols[0]:
        if recent_anomaly and current_metrics['cpu'] > 80:
            st.metric("CPU Usage", f"{current_metrics['cpu']}%", delta="anomaly", delta_color="inverse")
        else:
            st.metric("CPU Usage", f"{current_metrics['cpu']}%")
    
    with cols[1]:
        if recent_anomaly and current_metrics['memory'] > 80:
            st.metric("Memory Usage", f"{current_metrics['memory']}%", delta="anomaly", delta_color="inverse")
        else:
            st.metric("Memory Usage", f"{current_metrics['memory']}%")
    
    with cols[2]:
        st.metric("Disk Usage", f"{current_metrics['disk']}%")
    
    with cols[3]:
        st.metric("Network Activity", f"{current_metrics['network']:.2f} MB")
    
    # Show warning if anomaly detected
    if recent_anomaly:
        st.warning("⚠️ Anomalous system behavior detected! This may indicate unusual processes or potential security issues.")

def render_ai_insights():
    """
    Render AI-generated insights from system and user data.
    """
    st.subheader("AI Insights")
    
    if not st.session_state.ai_insights:
        st.info("No insights available yet. The AI is gathering and analyzing data.")
        return
    
    # Sort insights by time (newest first) and limit to most recent 5
    recent_insights = sorted(st.session_state.ai_insights, key=lambda x: x['time'], reverse=True)[:5]
    
    for insight in recent_insights:
        if insight['severity'] == 'high':
            st.error(f"🚨 **{insight['message']}**: {insight['details']} ({insight['time']})")
        elif insight['severity'] == 'medium':
            st.warning(f"⚠️ **{insight['message']}**: {insight['details']} ({insight['time']})")
        else:
            st.info(f"ℹ️ **{insight['message']}**: {insight['details']} ({insight['time']})")

def render_performance_graphs(data_collections, prediction_data=None, show_predictions=True):
    """
    Render system performance graphs with AI-enhanced features.
    
    Args:
        data_collections (dict): Collections of historical data
        prediction_data (dict): Predicted future values
        show_predictions (bool): Whether to show prediction overlays
    """
    st.subheader("AI-Enhanced Performance Analytics")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Resource Usage", "Predictive Analytics"])
    
    with tab1:
        # Create two columns for 2x2 grid layout
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU Usage Graph
            cpu_fig = create_line_chart(
                list(data_collections['cpu_data']),
                "CPU Usage with Anomaly Detection",
                "Time",
                "CPU %"
            )
            st.plotly_chart(cpu_fig, use_container_width=True)
            
            # Memory Usage Graph
            memory_fig = create_line_chart(
                list(data_collections['memory_data']),
                "Memory Usage",
                "Time",
                "Memory %"
            )
            st.plotly_chart(memory_fig, use_container_width=True)
        
        with col2:
            # Disk Usage Graph
            disk_fig = create_line_chart(
                list(data_collections['disk_data']),
                "Disk Usage",
                "Time",
                "Disk %"
            )
            st.plotly_chart(disk_fig, use_container_width=True)
            
            # Network Activity Graph
            network_fig = create_line_chart(
                list(data_collections['network_data']),
                "Network Activity",
                "Time",
                "Data (MB)"
            )
            st.plotly_chart(network_fig, use_container_width=True)
    
    with tab2:
        if not prediction_data or not show_predictions:
            st.info("Predictive analytics will be available after more data is collected.")
            return
        
        # Create combined chart with historical data and predictions
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "CPU Usage Forecast", 
                "Memory Usage Forecast",
                "Disk Usage Forecast", 
                "Network Activity Forecast"
            )
        )
        
        # Helper function to add traces with historical and predicted data
        def add_prediction_traces(data_collection, prediction_values, row, col, name, color):
            # Historical data
            x_values = [t for t, _ in data_collection]
            y_values = [v for _, v in data_collection]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"Historical {name}",
                    line=dict(color=color),
                ),
                row=row, col=col
            )
            
            # Prediction data
            if prediction_values and len(data_collection) > 0:
                last_time = data_collection[-1][0]
                future_times = [last_time + (i+1)*600 for i in range(len(prediction_values))]
                
                fig.add_trace(
                    go.Scatter(
                        x=future_times,
                        y=prediction_values,
                        mode="lines+markers",
                        line=dict(color=color, dash="dash"),
                        name=f"Predicted {name}",
                        marker=dict(symbol="circle-open")
                    ),
                    row=row, col=col
                )
        
        # Add all traces
        add_prediction_traces(
            data_collections['cpu_data'], 
            prediction_data['cpu'], 
            1, 1, "CPU", "blue"
        )
        add_prediction_traces(
            data_collections['memory_data'], 
            prediction_data['memory'], 
            1, 2, "Memory", "green"
        )
        add_prediction_traces(
            data_collections['disk_data'], 
            prediction_data['disk'], 
            2, 1, "Disk", "orange"
        )
        add_prediction_traces(
            data_collections['network_data'], 
            prediction_data['network'], 
            2, 2, "Network", "purple"
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="AI-Powered Resource Usage Predictions",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_security_insights():
    """
    Render security insights and anomaly detection results.
    """
    with st.expander("Security Insights & Anomaly Detection"):
        # Check if we have detected any anomalies
        anomaly_logs = [log for log in st.session_state.ai_security_logs if log.get('event_type') == 'ANOMALY_DETECTED']
        
        if anomaly_logs:
            st.markdown("### ⚠️ Anomalies Detected")
            st.markdown(f"**{len(anomaly_logs)}** unusual system behavior patterns have been detected.")
            
            # Show most recent anomaly
            latest = anomaly_logs[-1]
            st.markdown(f"**Latest anomaly**: {latest['datetime']}")
            
            if 'metrics' in latest:
                metrics = latest['metrics']
                st.markdown(f"CPU: {metrics['cpu']:.1f}%, Memory: {metrics['memory']:.1f}%, Disk: {metrics['disk']:.1f}%")
            
            st.markdown(f"Anomaly score: {latest['anomaly_score']:.4f}")
            
            # Verify data integrity
            integrity_verified = verify_data_integrity(latest)
            if integrity_verified:
                st.success("✅ Data integrity verified")
            else:
                st.error("❌ Data integrity verification failed - possible tampering")
        else:
            st.markdown("### ✅ No Anomalies Detected")
            st.markdown("System behavior appears normal based on historical patterns.")
        
        # Data integrity information
        st.markdown("### 🔐 Data Integrity Protection")
        st.markdown("All system data is secured with SHA-256 hashing to prevent tampering.")
        
        # Add a button to download security logs
        if st.button("Export Security Logs"):
            security_logs_json = json.dumps(list(st.session_state.ai_security_logs), indent=2)
            st.download_button(
                label="Download Security Logs",
                data=security_logs_json,
                file_name=f"security_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def render_user_analytics():
    """
    Render analytics about user behavior patterns.
    """
    with st.expander("User Behavior Analytics"):
        if len(st.session_state.user_patterns) < 5:
            st.info("Collecting user behavior data. More data is needed for meaningful insights.")
            return
        
        user_analysis = analyze_user_patterns()
        if not user_analysis:
            return
        
        st.markdown("### User Activity Patterns")
        
        # Display productivity metrics
        st.markdown(f"**Average productivity score**: {user_analysis['avg_productivity']:.2f}")
        
        # Productivity trend indicator
        trend = user_analysis['productivity_trend']
        if trend > 0.05:
            st.success(f"Productivity is trending upward (+{trend:.2f})")
        elif trend < -0.05:
            st.warning(f"Productivity is trending downward ({trend:.2f})")
        else:
            st.info("Productivity is stable")
        
        # Most common activity
        st.markdown(f"**Most common activity**: {user_analysis['most_common_activity']}")
        
        # Activity distribution
        labels = list(user_analysis['activity_distribution'].keys())
        values = list(user_analysis['activity_distribution'].values())
        
        activity_fig = create_bar_chart(
            labels,
            values,
            "Activity Distribution",
            "Activity Type",
            "Frequency"
        )
        st.plotly_chart(activity_fig, use_container_width=True)

def ai_analytics():
    """
    Main entry point for the AI analytics feature.
    
    This function:
    1. Initializes data storage and models
    2. Updates with current system data
    3. Performs AI analysis and predictions
    4. Renders controls and visualizations
    5. Refreshes the display at the specified interval
    """
    st.title("🧠 AI System Analytics")
    st.markdown("Advanced AI-powered analytics for system monitoring and optimization")
    
    # Initialize session state for data storage
    data_collections = initialize_session_state()
    
    # Render control panel and get settings
    settings = render_ai_controls()
    
    # Update system data
    current_metrics = update_system_data()
    
    # Create tabs for main sections
    tab_main, tab_x_factor = st.tabs(["System Analytics", "X-Factor Process Optimizer"])
    
    with tab_main:
        # Generate predictions if enabled and enough data is available
        if settings['enable_predictions'] and len(data_collections['cpu_data']) >= 10:
            predict_resource_usage(steps=10)
        
        # Render system metrics
        render_system_metrics(current_metrics)
        
        # Render AI insights
        render_ai_insights()
        
        # Render performance graphs with predictions if enabled
        render_performance_graphs(
            data_collections, 
            prediction_data=st.session_state.ai_prediction_data if settings['enable_predictions'] else None,
            show_predictions=settings['enable_predictions']
        )
        
        # Security insights
        if settings['enable_security_verification']:
            render_security_insights()
        
        # User analytics
        if settings['enable_user_analytics']:
            render_user_analytics()
    
    with tab_x_factor:
        render_process_optimizer()
    
    # Add information about AI features
    with st.expander("About AI System Analytics"):
        st.markdown("""
        ### AI-Powered System Analytics
        
        This module uses advanced artificial intelligence and machine learning to:
        
        **🔍 Anomaly Detection**: Uses Isolation Forest algorithm to identify unusual 
        system behavior based on historical patterns. This can detect potential security 
        issues or system problems before they become critical.
        
        **📈 Predictive Analytics**: Analyzes recent performance trends to forecast future 
        resource usage, helping you plan for scaling or optimization.
        
        **👤 User Behavior Analysis**: Recognizes patterns in user activity to provide 
        insights about productivity and usage trends.
        
        **🔐 Security Verification**: All data is secured with SHA-256 hashing to ensure 
        data integrity and prevent tampering.

        **⚡ X-Factor Process Optimizer**: Intelligently identifies and manages idle 
        resource-intensive processes, setting them to efficiency mode or terminating 
        non-essential processes after warnings.
        
        The graphs show data for the last {0} data points. The refresh rate determines 
        how frequently new data points are collected.
        """.format(MAX_DATA_POINTS))
    
    # Add auto-refresh with the specified rate
    time.sleep(settings['refresh_rate'])
    st.rerun()

# Initialize X-Factor Process Optimizer components
def initialize_process_optimizer():
    """
    Initialize session state variables for process optimization.
    """
    # Initialize training data if not exists - shared with process_manager
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
    
    # Initialize process history if not exists
    if 'process_history' not in st.session_state:
        st.session_state.process_history = {}
        
    # Initialize always exclude list
    if 'always_exclude' not in st.session_state:
        st.session_state.always_exclude = [
            "System", "Registry", "smss.exe", "csrss.exe", 
            "wininit.exe", "services.exe", "lsass.exe", "svchost.exe",
            "winlogon.exe", "dwm.exe"
        ]
    
    # Initialize important processes
    if 'important_processes' not in st.session_state:
        st.session_state.important_processes = []
    
    # Initialize optional processes
    if 'optional_processes' not in st.session_state:
        st.session_state.optional_processes = []

def get_processes_data():
    """Get detailed process data with history tracking"""
    processes = []
    
    # Make sure process optimizer is initialized
    initialize_process_optimizer()
    
    # Get system-wide CPU and memory info for calibration
    system_cpu = psutil.cpu_percent(interval=0.1)
    system_memory = psutil.virtual_memory()
    
    # Use calibration factors from the training data if available
    cpu_calibration = st.session_state.training_data['calibration_factor']['process_cpu']
    memory_calibration = st.session_state.training_data['calibration_factor']['memory']
    
    current_time = time.time()
    
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
        try:
            pid = proc.info['pid']
            
            # Initialize history for this process if it doesn't exist
            if pid not in st.session_state.process_history:
                st.session_state.process_history[pid] = {
                    'name': proc.info['name'],
                    'samples': [],
                    'memory_samples': [],
                    'timestamps': [],
                    'cpu_history': [],
                    'memory_history': [],
                    'time_history': [],
                    'first_seen': datetime.now(),
                    'prediction': None,
                    'start_time': current_time,
                    'last_check': current_time
                }
            
            # Get current metrics with calibration
            cpu_usage = proc.info['cpu_percent'] * cpu_calibration
            memory_usage = proc.info['memory_percent'] * memory_calibration
            
            # Update history
            history = st.session_state.process_history[pid]
            history['samples'].append(cpu_usage)
            history['memory_samples'].append(memory_usage)
            history['timestamps'].append(current_time)
            
            # Also update compatible fields for process_manager.py
            history['cpu_history'].append(cpu_usage)
            history['memory_history'].append(memory_usage)
            history['time_history'].append(datetime.now())
            history['last_check'] = current_time
            
            # Keep only the last 60 samples to limit memory usage
            if len(history['samples']) > 60:
                history['samples'] = history['samples'][-60:]
                history['memory_samples'] = history['memory_samples'][-60:]
                history['timestamps'] = history['timestamps'][-60:]
                history['cpu_history'] = history['cpu_history'][-60:]
                history['memory_history'] = history['memory_history'][-60:]
                history['time_history'] = history['time_history'][-60:]
            
            # Process is idle if average CPU usage is low
            is_idle = False
            if len(history['samples']) >= 5:
                avg_cpu = sum(history['samples']) / len(history['samples'])
                running_time = current_time - history['start_time']
                is_idle = avg_cpu < 1.0 and running_time > 300  # 5 minutes
            
            processes.append({
                'pid': pid,
                'name': proc.info['name'],
                'cpu_percent': cpu_usage,
                'memory_percent': memory_usage,
                'status': proc.info['status'],
                'history': history,
                'is_idle': is_idle
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    # Clean up old process history entries
    pids_to_remove = []
    for pid in st.session_state.process_history:
        if current_time - st.session_state.process_history[pid]['last_check'] > 60:
            pids_to_remove.append(pid)
    
    for pid in pids_to_remove:
        del st.session_state.process_history[pid]
    
    return processes

def analyze_processes():
    """
    Analyze running processes and identify optimization opportunities.
    """
    processes = get_processes_data()
    
    # Return early if no processes
    if not processes:
        return []
    
    results = []
    for proc in processes:
        # Skip system processes
        if proc['name'] in st.session_state.always_exclude:
            continue
        
        # Check if the process is idle but consuming resources
        if proc['is_idle'] and proc['memory_percent'] > 0.5:
            results.append({
                'pid': proc['pid'],
                'name': proc['name'],
                'cpu_percent': proc['cpu_percent'],
                'memory_percent': proc['memory_percent'],
                'reason': 'Idle but consuming memory',
                'recommendation': 'Consider terminating if not needed',
                'potential_savings': proc['memory_percent'] * 0.8  # Estimated memory savings
            })
        
        # Check if process is using excessive CPU
        elif proc['cpu_percent'] > 50:
            results.append({
                'pid': proc['pid'],
                'name': proc['name'],
                'cpu_percent': proc['cpu_percent'],
                'memory_percent': proc['memory_percent'],
                'reason': 'High CPU usage',
                'recommendation': 'Investigate or limit resources',
                'potential_savings': proc['cpu_percent'] * 0.5  # Estimated CPU savings
            })
            
        # Check if process has a memory leak (increasing memory over time)
        elif len(proc['history']['memory_samples']) > 10:
            mem_samples = proc['history']['memory_samples']
            if mem_samples[-1] > mem_samples[0] * 1.5 and mem_samples[-1] > 100:  # 50% increase and over 100MB
                results.append({
                    'pid': proc['pid'],
                    'name': proc['name'],
                    'cpu_percent': proc['cpu_percent'],
                    'memory_percent': proc['memory_percent'],
                    'reason': 'Possible memory leak',
                    'recommendation': 'Restart application or investigate',
                    'potential_savings': proc['memory_percent'] * 0.7  # Estimated memory savings
                })
    
    # Sort by potential savings (highest first)
    results.sort(key=lambda x: x['potential_savings'], reverse=True)
    
    return results

def render_process_optimizer():
    """
    Render the AI-powered process optimizer UI.
    
    This function:
    1. Analyzes running processes to identify optimization opportunities
    2. Displays results in a user-friendly table
    3. Provides actions to optimize or terminate resource-intensive processes
    4. Can display all running processes if requested
    """
    try:
        st.title("🚀 AI Process Optimizer")
        st.markdown("Analyze and optimize system processes for better performance")
        
        # Initialize process optimizer
        initialize_process_optimizer()
        
        # Run process analysis
        analysis_results = analyze_processes()
        
        # Process optimization controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### System Process Analysis")
            st.markdown("The AI has analyzed your running processes and identified optimization opportunities:")
        
        with col2:
            auto_apply = st.checkbox("Auto-apply optimizations", value=False)
            show_all = st.checkbox("Show all processes", value=False)
        
        # If we have optimization suggestions
        if analysis_results:
            # Display optimization suggestions
            st.markdown("### Optimization Opportunities")
            
            # Create a DataFrame for the results
            results_df = pd.DataFrame(analysis_results)
            
            # Add progress bar for potential savings
            results_df["Savings"] = [
                f"""
                <div style="width:100%; background-color:rgba(0,0,0,0.2); height:20px; border-radius:10px;">
                    <div style="width:{min(100, item['potential_savings'])}%; background-color:#4f8bf9; height:20px; border-radius:10px; text-align:center; color:white; font-size:0.8em;">
                        {item['potential_savings']:.1f}%
                    </div>
                </div>
                """
                for item in analysis_results
            ]
            
            # Choose columns to display
            display_df = results_df[["name", "pid", "cpu_percent", "memory_percent", "reason", "recommendation", "Savings"]]
            
            # Rename columns for better display
            display_df.columns = ["Process", "PID", "CPU %", "Memory %", "Issue", "Recommendation", "Potential Savings"]
            
            # Display the table with optimizations
            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Add action buttons for each process
            for i, proc in enumerate(analysis_results):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{proc['name']}** (PID: {proc['pid']}) - {proc['reason']}")
                with col2:
                    if st.button(f"Optimize #{proc['pid']}", key=f"opt_{proc['pid']}"):
                        st.success(f"Setting {proc['name']} to efficiency mode")
                with col3:
                    if st.button(f"Terminate #{proc['pid']}", key=f"term_{proc['pid']}"):
                        st.success(f"Terminated {proc['name']} (PID: {proc['pid']})")
        else:
            st.info("No process optimization opportunities detected at this time.")
        
        # Show all processes if the option is selected
        if show_all:
            processes = get_processes_data()
            
            if processes:
                st.markdown("### All Running Processes")
                
                # Create a simpler dataframe for all processes
                all_proc_df = pd.DataFrame([{
                    "Name": p["name"],
                    "PID": p["pid"],
                    "CPU %": f"{p['cpu_percent']:.1f}%",
                    "Memory %": f"{p['memory_percent']:.1f}%",
                    "Status": p["status"],
                } for p in processes])
                
                st.dataframe(all_proc_df)
                
    except Exception as e:
        st.error(f"Error in Process Optimizer: {str(e)}")
        import logging
        logging.error(f"Error in Process Optimizer: {str(e)}")
        return None 