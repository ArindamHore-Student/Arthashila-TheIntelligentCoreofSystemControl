import streamlit as st
import platform
import psutil
import sys
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import get_size

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

def system_overview():
    st.title("🖥️ System Overview")
    st.markdown("View detailed information about your system hardware and software")
    
    # Current time
    st.markdown(f"<div style='color: #888888; margin-bottom: 20px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
    
    # Quick Stats in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_percent = psutil.cpu_percent()
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center; color: #4f8bf9;">CPU Usage</h3>
            <div style="font-size: 2.5rem; text-align: center; font-weight: bold;">{}%</div>
        </div>
        """.format(cpu_percent), unsafe_allow_html=True)
        
    with col2:
        memory = psutil.virtual_memory()
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center; color: #4f8bf9;">Memory Usage</h3>
            <div style="font-size: 2.5rem; text-align: center; font-weight: bold;">{}%</div>
        </div>
        """.format(memory.percent), unsafe_allow_html=True)
        
    with col3:
        disk = psutil.disk_usage('/')
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center; color: #4f8bf9;">Disk Usage</h3>
            <div style="font-size: 2.5rem; text-align: center; font-weight: bold;">{}%</div>
        </div>
        """.format(disk.percent), unsafe_allow_html=True)
    
    # Gauge charts for visual representation
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_fig = create_gauge_chart(cpu_percent, "CPU Usage (%)")
        st.plotly_chart(cpu_fig, use_container_width=True)
    
    with col2:
        mem_fig = create_gauge_chart(memory.percent, "Memory Usage (%)")
        st.plotly_chart(mem_fig, use_container_width=True)
    
    # System Information
    with st.expander("📋 System Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖥️ Hardware")
            uname = platform.uname()
            system_info = [
                ["System", uname.system],
                ["Node Name", uname.node],
                ["Machine", uname.machine],
                ["Processor", uname.processor]
            ]
            
            for prop, value in system_info:
                st.markdown(f"**{prop}:** {value}")
        
        with col2:
            st.markdown("#### 💾 Software")
            software_info = [
                ["OS Release", uname.release],
                ["OS Version", uname.version],
                ["Python Version", platform.python_version()],
                ["Platform", platform.platform()]
            ]
            
            for prop, value in software_info:
                st.markdown(f"**{prop}:** {value}")
    
    # CPU Information
    with st.expander("⚡ CPU Information", expanded=True):
        st.markdown("#### CPU Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_freq = psutil.cpu_freq()
            
            cpu_info = [
                ["Physical Cores", str(psutil.cpu_count(logical=False))],
                ["Total Cores", str(psutil.cpu_count(logical=True))],
            ]
            
            for prop, value in cpu_info:
                st.markdown(f"**{prop}:** {value}")
                
        with col2:
            freq_info = [
                ["Max Frequency", f"{cpu_freq.max:.2f} MHz" if cpu_freq and hasattr(cpu_freq, 'max') and cpu_freq.max else "N/A"],
                ["Min Frequency", f"{cpu_freq.min:.2f} MHz" if cpu_freq and hasattr(cpu_freq, 'min') and cpu_freq.min else "N/A"],
                ["Current Frequency", f"{cpu_freq.current:.2f} MHz" if cpu_freq and hasattr(cpu_freq, 'current') and cpu_freq.current else "N/A"],
            ]
            
            for prop, value in freq_info:
                st.markdown(f"**{prop}:** {value}")
        
        # CPU Usage per core
        st.markdown("#### Per-Core Usage")
        cpu_percents = psutil.cpu_percent(percpu=True)
        
        cols = st.columns(4)
        for i, (col, percent) in enumerate(zip(cols * (len(cpu_percents) // 4 + 1), cpu_percents)):
            with col:
                st.markdown(f"**Core {i}**")
                st.progress(percent / 100)
                st.markdown(f"<div style='text-align: center;'>{percent}%</div>", unsafe_allow_html=True)

    # Memory Information
    with st.expander("🧠 Memory Information", expanded=True):
        svmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RAM Usage")
            memory_info = [
                ["Total", get_size(svmem.total)],
                ["Available", get_size(svmem.available)],
                ["Used", get_size(svmem.used)],
                ["Percentage", f"{svmem.percent}%"]
            ]
            
            for prop, value in memory_info:
                st.markdown(f"**{prop}:** {value}")
                
            # RAM Usage Breakdown
            ram_fig = go.Figure()
            ram_fig.add_trace(go.Pie(
                labels=["Used", "Available"],
                values=[svmem.used, svmem.available],
                hole=.4,
                marker_colors=["#4f8bf9", "#1a1f2c"]
            ))
            ram_fig.update_layout(
                title_text="RAM Usage Breakdown",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font={'color': "white"},
                height=300
            )
            st.plotly_chart(ram_fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Swap Memory")
            swap_info = [
                ["Total", get_size(swap.total)],
                ["Free", get_size(swap.free)],
                ["Used", get_size(swap.used)],
                ["Percentage", f"{swap.percent}%"]
            ]
            
            for prop, value in swap_info:
                st.markdown(f"**{prop}:** {value}")
                
            # Swap Usage Breakdown
            swap_fig = go.Figure()
            swap_fig.add_trace(go.Pie(
                labels=["Used", "Free"],
                values=[swap.used, swap.free],
                hole=.4,
                marker_colors=["#4f8bf9", "#1a1f2c"]
            ))
            swap_fig.update_layout(
                title_text="Swap Usage Breakdown",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font={'color': "white"},
                height=300
            )
            st.plotly_chart(swap_fig, use_container_width=True)
    
    # Disk Information
    with st.expander("💽 Disk Information", expanded=True):
        st.markdown("#### Disk Partitions")
        
        partitions = psutil.disk_partitions()
        for i, partition in enumerate(partitions):
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                
                st.markdown(f"**Partition {i+1}: {partition.mountpoint}**")
                st.markdown(f"**Device:** {partition.device}")
                st.markdown(f"**File System:** {partition.fstype}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Size", get_size(partition_usage.total))
                col2.metric("Used", get_size(partition_usage.used))
                col3.metric("Free", get_size(partition_usage.free))
                
                st.progress(partition_usage.percent / 100)
                st.markdown(f"<div style='text-align: right;'>{partition_usage.percent}% used</div>", unsafe_allow_html=True)
                st.markdown("---")
            except:
                st.markdown(f"**Partition {i+1}: {partition.mountpoint}** (Access Denied)")
                st.markdown("---")