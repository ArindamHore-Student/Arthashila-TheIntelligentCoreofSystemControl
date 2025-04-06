import streamlit as st
import psutil
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for displaying values like CPU usage."""
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
    """Return color based on CPU usage."""
    if cpu_percent < 5:
        return "#0cce6b"  # Green for low usage
    elif cpu_percent < 25:
        return "#4f8bf9"  # Blue for medium usage
    elif cpu_percent < 60:
        return "#f9a825"  # Orange/Yellow for high usage
    else:
        return "#ff4b4b"  # Red for very high usage

def format_bytes(bytes_value):
    """Format bytes into a readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def process_manager():
    st.title("⚙️ Process Manager")
    st.markdown("Monitor and manage system processes in real-time")
    
    # Current time
    st.markdown(f"<div style='color: #888888; margin-bottom: 20px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
    
    # System resource overview
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_fig = create_gauge_chart(cpu_percent, "CPU Usage (%)")
        st.plotly_chart(cpu_fig, use_container_width=True)
    
    with col2:
        mem_fig = create_gauge_chart(memory.percent, "Memory Usage (%)")
        st.plotly_chart(mem_fig, use_container_width=True)
    
    # Process management options
    st.markdown("## 🔍 Process List")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["CPU Usage (%)", "Memory Usage (MB)", "Process ID", "Process Name"],
            index=0
        )
    
    with filter_col2:
        search_term = st.text_input("Search Process", placeholder="Enter process name...")
    
    # Get process list
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_info', 'status', 'create_time']):
        try:
            # Get process information
            proc_info = proc.info
            if search_term and search_term.lower() not in proc_info['name'].lower():
                continue
                
            # Get memory in MB
            memory_mb = proc_info['memory_info'].rss / (1024 * 1024) if proc_info['memory_info'] else 0
            
            # Get process create time
            if proc_info['create_time']:
                create_time = datetime.fromtimestamp(proc_info['create_time']).strftime('%Y-%m-%d %H:%M:%S')
            else:
                create_time = "Unknown"
            
            processes.append({
                'pid': proc_info['pid'],
                'name': proc_info['name'],
                'username': proc_info['username'],
                'cpu_percent': proc_info['cpu_percent'],
                'memory_mb': memory_mb,
                'status': proc_info['status'],
                'create_time': create_time
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    # Sort processes
    if sort_by == "CPU Usage (%)":
        processes = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)
    elif sort_by == "Memory Usage (MB)":
        processes = sorted(processes, key=lambda x: x['memory_mb'], reverse=True)
    elif sort_by == "Process ID":
        processes = sorted(processes, key=lambda x: x['pid'])
    else:  # Process Name
        processes = sorted(processes, key=lambda x: x['name'].lower())
    
    # Display top resource-consuming processes
    top_processes = processes[:5]
    
    # Top processes visualization
    if top_processes:
        st.markdown("### Top Resource-Consuming Processes")
        
        # CPU Usage Bar Chart
        cpu_df = pd.DataFrame([
            {"Process": f"{p['name']} (PID: {p['pid']})", "CPU (%)": p['cpu_percent']} 
            for p in top_processes
        ])
        
        fig = px.bar(
            cpu_df,
            x="CPU (%)",
            y="Process",
            orientation='h',
            color="CPU (%)",
            color_continuous_scale=px.colors.sequential.Blues,
            title="CPU Usage by Process"
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300,
            margin=dict(l=0, r=10, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Memory Usage Bar Chart
        mem_df = pd.DataFrame([
            {"Process": f"{p['name']} (PID: {p['pid']})", "Memory (MB)": p['memory_mb']} 
            for p in top_processes
        ])
        
        fig = px.bar(
            mem_df,
            x="Memory (MB)",
            y="Process",
            orientation='h',
            color="Memory (MB)",
            color_continuous_scale=px.colors.sequential.Blues,
            title="Memory Usage by Process"
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300,
            margin=dict(l=0, r=10, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Process list
    st.markdown("### All Processes")
    
    if not processes:
        st.info("No processes found matching your criteria.")
    else:
        # Convert to DataFrame for table display
        df = pd.DataFrame(processes)
        
        # Display in a scrollable container with a limit
        process_limit = min(100, len(processes))
        limited_df = df.head(process_limit)
        
        # Custom table display with colors based on resource usage
        for i, row in limited_df.iterrows():
            with st.container():
                # Calculate color based on CPU usage
                cpu_color = get_process_color(row['cpu_percent'])
                
                with st.expander(f"{row['name']} (PID: {row['pid']})", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Process Name:** {row['name']}")
                        st.markdown(f"**PID:** {row['pid']}")
                        st.markdown(f"**User:** {row['username']}")
                        st.markdown(f"**Status:** {row['status']}")
                        st.markdown(f"**Started:** {row['create_time']}")
                    
                    with col2:
                        # CPU Usage
                        st.markdown("**CPU Usage**")
                        st.progress(min(row['cpu_percent'] / 100, 1.0))
                        st.markdown(f"<div style='text-align: right; color: {cpu_color};'>{row['cpu_percent']:.2f}%</div>", unsafe_allow_html=True)
                        
                        # Memory Usage
                        st.markdown("**Memory Usage**")
                        memory_percent = min(row['memory_mb'] / (memory.total / (1024 * 1024)) * 100, 100)
                        st.progress(memory_percent / 100)
                        st.markdown(f"<div style='text-align: right;'>{row['memory_mb']:.2f} MB</div>", unsafe_allow_html=True)
                        
                        # Terminate process button
                        if st.button("Terminate Process", key=f"terminate_{row['pid']}"):
                            try:
                                process = psutil.Process(row['pid'])
                                process.terminate()
                                st.success(f"Process {row['name']} (PID: {row['pid']}) terminated successfully!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error terminating process: {str(e)}")
        
        if len(processes) > process_limit:
            st.info(f"Showing {process_limit} of {len(processes)} processes. Use search to find specific processes.")
    
    # Auto refresh option
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (5s)")
    if auto_refresh:
        time.sleep(5)
        st.rerun()