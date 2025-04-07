"""
Battery Management Module for Arthashila

This module provides battery monitoring and power management features.
It displays battery status, remaining time, and power-saving tips
for laptop and mobile device users.
"""

import streamlit as st
import psutil
import time
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_battery_status():
    """
    Get battery information from the system.
    
    Returns:
        dict or None: Dictionary containing battery information if available,
                      None if battery is not detected or an error occurs
    """
    try:
        battery = psutil.sensors_battery()
        
        if battery is None:
            return None
        
        # Extract battery information
        status_info = {
            "percent": battery.percent,
            "power_plugged": battery.power_plugged,
            "status": "Charging" if battery.power_plugged else "Discharging",
            "secsleft": battery.secsleft
        }
        
        return status_info
    except Exception as e:
        st.error(f"Error retrieving battery information: {str(e)}")
        return None

def format_remaining_time(seconds_left):
    """
    Format seconds left into a human-readable time format.
    
    Args:
        seconds_left (int): Seconds of battery life remaining
        
    Returns:
        str: Formatted time string (e.g., "2h 45m remaining")
    """
    if seconds_left == psutil.POWER_TIME_UNLIMITED:
        return "Connected to power"
    elif seconds_left == psutil.POWER_TIME_UNKNOWN:
        return "Time remaining unknown"
    else:
        hours, remainder = divmod(seconds_left, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m remaining"

def get_color_for_battery_level(percent):
    """
    Get appropriate color based on battery percentage.
    
    Args:
        percent (float): Battery percentage
        
    Returns:
        str: Hex color code
    """
    if percent > 50:
        return "#4CAF50"  # Green for good battery level
    elif percent > 20:
        return "#FFA500"  # Orange for medium battery level
    else:
        return "#FF0000"  # Red for low battery level

def render_battery_info(battery_info):
    """
    Render battery information visualization.
    
    Args:
        battery_info (dict): Dictionary with battery status information
    """
    percent = battery_info["percent"]
    power_plugged = battery_info["power_plugged"]
    status = battery_info["status"]
    time_left = format_remaining_time(battery_info["secsleft"])
    
    # Create two columns for battery display
    col1, col2 = st.columns(2)
    
    with col1:
        # Battery percentage with color coding
        color = get_color_for_battery_level(percent)
        st.markdown(f"""
            <div style="text-align: center;">
                <h1 style="color: {color}; font-size: 4rem;">{percent}%</h1>
                <p style="font-size: 1.2rem;">{status}</p>
                <p>{time_left}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Battery level visualization
        st.progress(percent / 100)
        
        # Power status information
        if power_plugged:
            st.success("⚡ Power Adapter is connected")
        else:
            st.warning("🔋 Operating on battery power")
            
            # Show critical warning for very low battery
            if percent < 10:
                st.error("⚠️ CRITICAL: Battery level very low. Connect to power soon.")

def display_power_saving_tips():
    """
    Display power saving tips to help users extend battery life.
    """
    st.subheader("Power Saving Tips")
    
    tips = [
        "Reduce screen brightness to save power",
        "Close unused applications to reduce CPU usage",
        "Disable Wi-Fi and Bluetooth when not in use",
        "Use power-saving mode when battery is low",
        "Avoid running resource-intensive applications on battery"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")
    
    # Additional power management information
    with st.expander("Advanced Power Management"):
        st.markdown("""
        ### Extending Battery Life
        
        Battery life depends on usage patterns and power management settings.
        Here are some additional tips:
        
        1. **Update your operating system** - Newer OS versions often include power optimizations
        2. **Check for battery-draining apps** - Some applications consume excessive power
        3. **Reduce display timeout** - Set your screen to turn off quickly when not in use
        4. **Use battery saver mode** - Enable power-saving modes when needed
        5. **Optimize startup programs** - Disable unnecessary applications from running at startup
        """)

def battery_management():
    """
    Main entry point for the battery management feature.
    
    This function:
    1. Checks if battery is available
    2. Retrieves and displays battery information
    3. Shows power-saving tips
    """
    st.title("🔋 Battery & Power Management")
    st.markdown("Monitor battery status and optimize power usage")

    # Get battery information
    battery_info = get_battery_status()
    
    if battery_info is None:
        st.warning("No battery detected. This feature is only available on laptops and devices with batteries.")
        st.info("Consider using a laptop or mobile device to access battery management features.")
        
        # Show helpful information even without a battery
        with st.expander("Power Management on Desktop Systems"):
            st.markdown("""
            ### Power Management for Desktop Systems
            
            Even without a battery, you can still implement power management:
            
            - Use sleep or hibernate modes when not using your computer
            - Configure power plans to balance performance and energy usage
            - Consider using energy-efficient hardware components
            - Turn off monitors and peripherals when not in use
            """)
        return
    
    # Display battery information
    render_battery_info(battery_info)
    
    # Display power saving tips and battery health information
    display_power_saving_tips()
