"""
SimpleRTOS - Real-Time Operating System Monitor

This application provides a comprehensive system monitoring and task management interface
built with Streamlit. It allows users to monitor system resources, manage processes,
track performance metrics, manage battery usage, plan tasks, and collaborate with team members.

Author: SimpleRTOS Team
Version: 1.0
License: MIT
"""

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import feature modules
from features.system_overview import system_overview
from features.process_manager import process_manager
from features.performance_graphs import performance_graphs
from features.battery_management import battery_management
from features.task_planning import task_planning
from features.collaboration_tools import collaboration_tools

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
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """
    Render the application sidebar with navigation and system status.
    
    Returns:
        str: The selected navigation option
    """
    with st.sidebar:
        # Logo and app name 
        st.markdown('<div class="logo-text">⚙️ SimpleRTOS</div>', unsafe_allow_html=True)
        st.markdown("#### Real-Time Operating System Monitor")
        
        # Navigation menu
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "System Overview", 
                "Process Manager", 
                "Performance Graphs", 
                "Battery & Power", 
                "Task Planning", 
                "Collaboration"
            ],
            icons=[
                "cpu", 
                "list-task", 
                "graph-up", 
                "battery-charging", 
                "calendar-check", 
                "people"
            ],
            menu_icon="menu-button-wide",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#1a1f2c"},
                "icon": {"color": "#4f8bf9", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#2d3747"
                },
                "nav-link-selected": {"background-color": "#4f8bf9"},
            },
        )
        
        st.markdown("---")
        render_system_status()
        
    return selected

def render_system_status():
    """
    Display real-time system status metrics in the sidebar.
    Shows CPU, memory, and disk usage with appropriate color coding.
    """
    import psutil
    
    st.markdown("### System Status")
    
    # Get current system metrics
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Helper function to determine color based on usage percentage
    def get_usage_color(percent):
        if percent > 80:
            return 'red'
        elif percent > 50:
            return 'orange'
        else:
            return '#4f8bf9'
    
    # CPU status with icon
    cpu_color = get_usage_color(cpu_percent)
    st.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div>🔄 CPU Usage</div>"
        f"<div style='margin-left: auto; font-weight: bold; color: {cpu_color};'>{cpu_percent}%</div>"
        f"</div>", 
        unsafe_allow_html=True
    )
    st.progress(cpu_percent / 100)
    
    # Memory status with icon
    memory_color = get_usage_color(memory.percent)
    st.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div>🧠 Memory Usage</div>"
        f"<div style='margin-left: auto; font-weight: bold; color: {memory_color};'>{memory.percent}%</div>"
        f"</div>", 
        unsafe_allow_html=True
    )
    st.progress(memory.percent / 100)
    
    # Disk usage with icon
    disk_color = get_usage_color(disk.percent)
    st.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div>💽 Disk Usage</div>"
        f"<div style='margin-left: auto; font-weight: bold; color: {disk_color};'>{disk.percent}%</div>"
        f"</div>", 
        unsafe_allow_html=True
    )
    st.progress(disk.percent / 100)

def render_footer():
    """
    Render the application footer with version information and copyright.
    """
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888888; font-size: 12px;'>
            <p>SimpleRTOS v1.0 | Built with ❤️ using Streamlit</p>
            <p>© 2023 SimpleRTOS</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    """
    Main application entry point.
    Sets up the page configuration, loads CSS, renders the sidebar,
    and routes to the appropriate feature module based on navigation selection.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="SimpleRTOS",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS for styling
    load_custom_css()
    
    # Render sidebar and get selected option
    selected_option = render_sidebar()
    
    # Route to the appropriate module based on selection
    if selected_option == "System Overview":
        system_overview()
    elif selected_option == "Process Manager":
        process_manager()
    elif selected_option == "Performance Graphs":
        performance_graphs()
    elif selected_option == "Battery & Power":
        battery_management()
    elif selected_option == "Task Planning":
        task_planning()
    elif selected_option == "Collaboration":
        collaboration_tools()
    
    # Render application footer
    render_footer()

if __name__ == "__main__":
    main()