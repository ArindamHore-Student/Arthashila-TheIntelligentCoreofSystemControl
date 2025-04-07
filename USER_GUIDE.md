# Arthashila User Guide

This guide will help you navigate and use the Arthashila application effectively.

## Table of Contents
- [Getting Started](#getting-started)
- [Navigation](#navigation)
- [System Overview](#system-overview)
- [Process Manager](#process-manager)
- [Performance Graphs](#performance-graphs)
- [AI Analytics](#ai-analytics)
- [Battery & Power](#battery--power)
- [Task Planning](#task-planning)
- [Collaboration](#collaboration)
- [Troubleshooting](#troubleshooting)
- [Keyboard Shortcuts](#keyboard-shortcuts)

## Getting Started

### Installation

1. Make sure you have Python 3.7 or higher installed on your system
2. Install Arthashila:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```bash
   streamlit run main.py
   ```

### First Launch

When you first launch Arthashila, you'll see the main interface with:
- A navigation sidebar on the left
- The main content area on the right, starting with the System Overview
- Real-time system status metrics in the sidebar

## Navigation

The sidebar contains:
- **Arthashila Logo and Name**: At the top of the sidebar
- **Navigation Menu**: Click on any option to switch to that feature
- **System Status**: Quick view of CPU, memory, and disk usage
- **Footer**: Version information and copyright

## System Overview

The System Overview page provides detailed information about your computer:

### System Information
- **OS Information**: Operating system name, version, and platform
- **System Uptime**: How long your system has been running
- **Hostname**: Your computer's network name
- **User**: Current logged-in user

### CPU Information
- **Processor**: CPU model and manufacturer
- **Architecture**: x86, x64, ARM, etc.
- **Cores**: Number of physical and logical CPU cores
- **Frequency**: Current CPU frequency
- **Usage**: Real-time CPU usage visualized in a gauge chart

### Memory Information
- **Total Memory**: Total installed RAM
- **Available Memory**: Currently available RAM
- **Used Memory**: Memory in use
- **Percentage Used**: Visualized as a gauge chart
- **Swap Memory**: Virtual memory usage

## Process Manager

The Process Manager allows you to view and manage running processes:

### Process List
- View all running processes sorted by various metrics
- See process name, PID, CPU usage, memory usage, and status
- Sort processes by clicking on column headers

### Process Actions
- **Terminate Process**: Select a process and click "Terminate Process" to end it
- **Refresh**: Click "Refresh List" to update the process list
- **Search**: Filter the list by typing in the search box

### Process Details
- Click on any process to view detailed information:
  - Creation time
  - Command line
  - Current working directory
  - Threads count
  - Memory maps

### Process Priority
- Change the priority of selected processes to optimize system performance
- Set processes to higher priorities for important tasks
- Lower priorities for background tasks

## Performance Graphs

The Performance Graphs section provides visual representations of system performance:

### Available Metrics
- **CPU Usage**: Historical CPU usage over time
- **Memory Usage**: RAM and swap memory usage
- **Disk I/O**: Disk read/write operations
- **Network**: Network traffic in/out

### Chart Options
- **Time Range**: Select the time period to display (1 minute to 1 hour)
- **Chart Type**: Choose between line charts and bar charts
- **Refresh Rate**: How often the charts update (5s to 60s)

### Interactivity
- Hover over chart points to see exact values
- Zoom in/out using the chart controls
- Download charts as PNG images

## AI Analytics

The AI Analytics section provides insights into system performance and behavior:

### Available Metrics
- **AI Usage**: Historical AI usage over time
- **AI Efficiency**: AI performance metrics

### Chart Options
- **Time Range**: Select the time period to display (1 minute to 1 hour)
- **Chart Type**: Choose between line charts and bar charts
- **Refresh Rate**: How often the charts update (5s to 60s)

### Interactivity
- Hover over chart points to see exact values
- Zoom in/out using the chart controls
- Download charts as PNG images

## Battery & Power

The Battery & Power section helps monitor and optimize battery usage:

### Battery Status
- **Current Charge**: Battery percentage remaining
- **Power Status**: Whether connected to AC power or running on battery
- **Remaining Time**: Estimated time until battery is depleted
- **Battery Health**: Overall battery condition

### Power Saving Tips
- Recommendations to extend battery life
- Settings you can adjust to save power
- Power profile management

### Power Usage Graph
- Battery drain rate over time
- Impact of applications on battery life
- Charging pattern visualization

## Task Planning

The Task Planning section helps you organize and track your tasks:

### Task Management
- **Create Tasks**: Add new tasks with title, description, priority, and due date
- **Edit Tasks**: Modify existing task details
- **Delete Tasks**: Remove completed or unnecessary tasks
- **Mark as Complete**: Track task completion

### Task Organization
- **Categories**: Organize tasks by project or category
- **Priority Levels**: Set priorities (Low, Medium, High, Critical)
- **Due Dates**: Set and track deadlines
- **Status**: Track whether tasks are pending, in progress, or completed

### Task Filtering
- Filter by status, priority, or category
- Search for specific tasks
- Sort by different criteria (due date, priority, etc.)

### Task Visualization
- Progress charts for task completion
- Timeline view of upcoming deadlines
- Priority distribution charts

## Collaboration

The Collaboration section facilitates teamwork:

### Messaging
- **Send Messages**: Communicate with team members
- **Conversation History**: View past messages
- **Notifications**: Get alerted when you receive new messages

### Shared Tasks
- **Assign Tasks**: Delegate tasks to team members
- **Task Status**: Track who is working on what
- **Updates**: Get notified of task status changes

### Team Management
- **Add Members**: Include new team members
- **Set Roles**: Assign different permissions
- **Activity Log**: See who did what and when

## Troubleshooting

### Common Issues

#### Application Won't Start
- Ensure Python 3.7+ is installed
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check for error messages in the terminal

#### High Resource Usage
- Close the Performance Graphs tab when not needed
- Reduce the refresh rate of monitoring features
- Close unnecessary browser tabs while using Arthashila

#### Tasks Not Saving
- Ensure you have write permissions to the application directory
- Check if disk space is available
- Verify you've clicked the "Save" button after making changes

### Getting Help
- Check the project GitHub repository for issues and solutions
- Submit bug reports with detailed information

## Keyboard Shortcuts

Arthashila supports several keyboard shortcuts to improve your productivity:

- **R**: Refresh current view
- **Ctrl+N**: Create new task (in Task Planning)
- **Esc**: Close modal dialogs
- **F**: Focus on search box
- **1-6**: Quick navigation to the six main features
- **?**: Show keyboard shortcut help dialog

---

We hope this guide helps you make the most of Arthashila. For developer documentation, please refer to the DEVELOPERS.md file. 