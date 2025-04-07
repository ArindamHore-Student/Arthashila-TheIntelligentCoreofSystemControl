"""
Arthashila - System Monitor

This application provides a comprehensive system monitoring and task management interface
built with Streamlit. It allows users to monitor system resources, manage processes,
track performance metrics, leverage AI-powered analytics, manage battery usage, and plan tasks.

Author: Arthashila Team
Version: 1.2
License: MIT
"""

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import sys
import os
import psutil
import time
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import feature modules
from features.system_overview import system_overview
from features.process_manager import process_manager
from features.performance_graphs import performance_graphs
from features.ai_analytics import ai_analytics
from features.battery_management import battery_management
from features.task_planning import task_planning

def load_custom_css():
    """
    Apply custom CSS styling to enhance the UI appearance.
    This includes theme colors, typography, component styling, and animations.
    """
    st.markdown("""
        <style>
        /* Base Theme */
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stApp {
            background-color: #0e1117;
        }
        .stSidebar {
            background-color: #1a1f2c;
            border-right: 1px solid #2d3747;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #4f8bf9;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-weight: 600;
        }
        .sidebar .sidebar-content {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #4f8bf9;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 8px 16px;
            transition: all 0.3s ease;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            background-color: #3a7be0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Progress Bars */
        .stProgress > div > div > div {
            background-color: #4f8bf9;
        }
        
        /* Select boxes and inputs */
        .stSelectbox > div > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {
            background-color: #1a1f2c;
            color: white;
            border: 1px solid #2d3747;
            border-radius: 4px;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #1a1f2c;
            color: white;
            border-radius: 4px;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1f2c;
            border-radius: 4px 4px 0 0;
            padding: 8px 16px;
            border: 1px solid #2d3747;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4f8bf9;
            color: white;
        }
        
        /* Containers and dividers */
        .stContainer {
            border: 1px solid #2d3747;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            background-color: #1a1f2c;
        }
        
        hr {
            border-top: 1px solid #2d3747;
        }
        
        /* Tables */
        .dataframe {
            border: 1px solid #2d3747;
            border-radius: 4px;
            overflow: hidden;
        }
        .dataframe th {
            background-color: #1a1f2c;
            color: #4f8bf9;
            text-align: left;
            padding: 8px;
        }
        .dataframe td {
            background-color: #0e1117;
            color: white;
            padding: 8px;
        }
        
        /* Logo styling */
        .logo-text {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4f8bf9, #6dd5ed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        
        /* Custom container for cards */
        .card {
            background-color: #1a1f2c;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 16px;
            border: 1px solid #2d3747;
        }
        
        /* Real-time indicators */
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
        
        /* Real-time indicator styling */
        .real-time-indicator {
            display: inline-flex;
            align-items: center;
            background-color: #232730;
            border-radius: 16px;
            padding: 2px 10px;
            margin-left: 10px;
            font-size: 0.8em;
        }
        .indicator-dot {
            height: 8px;
            width: 8px;
            background-color: #0cce6b;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.4; }
            100% { opacity: 1; }
        }
        
        /* Resource usage indicators */
        .resource-indicator {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.85em;
        }
        .resource-label {
            flex: 1;
            color: #a3a8b8;
        }
        .resource-value {
            font-weight: bold;
            margin-right: 5px;
        }
        .usage-bar {
            height: 4px;
            border-radius: 2px;
            flex: 2;
            background-color: #2d3747;
            position: relative;
            overflow: hidden;
        }
        .usage-fill {
            position: absolute;
            height: 100%;
            left: 0;
            top: 0;
        }
        .cpu-usage { background-color: #4f8bf9; }
        .memory-usage { background-color: #0cce6b; }
        .battery-usage { background-color: #f9a825; }
        .low-usage { background-color: #0cce6b; }
        .medium-usage { background-color: #f9a825; }
        .high-usage { background-color: #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """
    Render the application sidebar with navigation and system status.
    
    Returns:
        str: The selected navigation option
    """
    # Check if we need to set the selected option from session state
    default_index = 0
    options = [
        "System Overview", 
        "Process Manager", 
        "Performance Graphs",
        "AI Analytics",
        "Battery & Power", 
        "Task Planning"
    ]
    
    # If navigate_to is in session state, find its index
    if 'navigate_to' in st.session_state:
        try:
            default_index = options.index(st.session_state.navigate_to)
            # Clear it after use
            del st.session_state.navigate_to
        except ValueError:
            default_index = 0
    
    with st.sidebar:
        # Logo and app name 
        st.markdown('<div class="logo-text">⚙️ Arthashila</div>', unsafe_allow_html=True)
        st.markdown("#### System Monitor")
        
        # Navigation menu
        selected = option_menu(
            menu_title="Navigation",
            options=options,
            icons=[
                "cpu", 
                "list-task", 
                "graph-up",
                "robot",
                "battery-charging", 
                "calendar-check"
            ],
            menu_icon="menu-button-wide",
            default_index=default_index,
            styles={
                "container": {"padding": "5px", "background-color": "#1a1f2c"},
                "icon": {"color": "#4f8bf9", "font-size": "14px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#2d3747",
                },
                "nav-link-selected": {"background-color": "#4f8bf9"},
            },
        )
        
        # System stats in sidebar
        st.markdown("#### System Status")
        
        # Get current system information to share between components
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Show current time
        st.markdown(f"**Time**: {datetime.now().strftime('%H:%M:%S')}")
        
        # CPU Usage
        st.markdown(
            f"""
            <div class="resource-indicator">
                <div class="resource-label">CPU Usage</div>
                <div class="resource-value">{cpu_percent}%</div>
                <div class="usage-bar">
                    <div class="usage-fill cpu-usage" style="width: {cpu_percent}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Memory Usage
        memory_percent = memory.percent
        st.markdown(
            f"""
            <div class="resource-indicator">
                <div class="resource-label">Memory Usage</div>
                <div class="resource-value">{memory_percent}%</div>
                <div class="usage-bar">
                    <div class="usage-fill memory-usage" style="width: {memory_percent}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Battery status if available
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                power_plugged = battery.power_plugged
                status = "Charging" if power_plugged else "Discharging"
                
                # Determine battery color class
                color_class = "low-usage"
                if percent < 20:
                    color_class = "high-usage"
                elif percent < 50:
                    color_class = "medium-usage"
                
                st.markdown(
                    f"""
                    <div class="resource-indicator">
                        <div class="resource-label">Battery ({status})</div>
                        <div class="resource-value">{percent}%</div>
                        <div class="usage-bar">
                            <div class="usage-fill {color_class}" style="width: {percent}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Footer
        st.markdown("---")
        st.markdown(
            "Developed with ❤️ by Arthashila Team",
            unsafe_allow_html=True
        )
        
        return selected

def main():
    """Main application entry point"""
    
    # Set page configuration
    st.set_page_config(
        page_title="Arthashila - System Monitor",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    h1, h2, h3, h4 {
        color: #4f8bf9;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2c;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0cce6b;
        color: black;
    }
    
    /* Real-time indicator styling */
    .real-time-indicator {
        display: inline-flex;
        align-items: center;
        background-color: #232730;
        border-radius: 16px;
        padding: 2px 10px;
        margin-left: 10px;
        font-size: 0.8em;
    }
    .indicator-dot {
        height: 8px;
        width: 8px;
        background-color: #0cce6b;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    
    /* Resource usage indicators */
    .resource-indicator {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.85em;
    }
    .resource-label {
        flex: 1;
        color: #a3a8b8;
    }
    .resource-value {
        font-weight: bold;
        margin-right: 5px;
    }
    .usage-bar {
        height: 4px;
        border-radius: 2px;
        flex: 2;
        background-color: #2d3747;
        position: relative;
        overflow: hidden;
    }
    .usage-fill {
        position: absolute;
        height: 100%;
        left: 0;
        top: 0;
    }
    .cpu-usage { background-color: #4f8bf9; }
    .memory-usage { background-color: #0cce6b; }
    .battery-usage { background-color: #f9a825; }
    .low-usage { background-color: #0cce6b; }
    .medium-usage { background-color: #f9a825; }
    .high-usage { background-color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

    # Get current system information to share between components
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Arthashila")
        st.caption("System Monitor")
        
        # Navigation
        st.header("Navigation")
        option = st.radio(
            "Select a feature",
            [
                "System Overview",
                "Process Manager",
                "Performance Graphs",
                "AI Analytics",
                "Battery & Power",
                "Task Planning"
            ]
        )
        
        # System status in sidebar
        st.markdown("---")
        st.subheader("System Status")
        
        # Show current time
        st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
        
        # CPU Usage
        st.markdown(
            f"""
            <div class="resource-indicator">
                <div class="resource-label">CPU Usage</div>
                <div class="resource-value">{cpu_percent}%</div>
                <div class="usage-bar">
                    <div class="usage-fill cpu-usage" style="width: {cpu_percent}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Memory Usage
        memory_percent = memory.percent
        st.markdown(
            f"""
            <div class="resource-indicator">
                <div class="resource-label">Memory Usage</div>
                <div class="resource-value">{memory_percent}%</div>
                <div class="usage-bar">
                    <div class="usage-fill memory-usage" style="width: {memory_percent}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Battery status if available
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                power_plugged = battery.power_plugged
                status = "Charging" if power_plugged else "Discharging"
                
                # Determine battery color class
                color_class = "low-usage"
                if percent < 20:
                    color_class = "high-usage"
                elif percent < 50:
                    color_class = "medium-usage"
                
                st.markdown(
                    f"""
                    <div class="resource-indicator">
                        <div class="resource-label">Battery ({status})</div>
                        <div class="resource-value">{percent}%</div>
                        <div class="usage-bar">
                            <div class="usage-fill {color_class}" style="width: {percent}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Version and developer info
        st.markdown("---")
        st.caption("Version 1.2 | Developed by Arthashila Team")

    # Main content
    if option == "System Overview":
        system_overview()
    elif option == "Process Manager":
        process_manager()
    elif option == "Performance Graphs":
        performance_graphs()
    elif option == "AI Analytics":
        ai_analytics()
    elif option == "Battery & Power":
        battery_management()
    elif option == "Task Planning":
        task_planning()

if __name__ == "__main__":
    main()