import streamlit as st
import json
import os
import time
import pandas as pd
import uuid
from datetime import datetime, date
import plotly.express as px

# File paths
MESSAGES_FILE = "data/messages.json"
TASKS_FILE = "data/tasks.json"

# Colors for different priorities
PRIORITY_COLORS = {
    "Low": "#0cce6b",    # Green
    "Medium": "#f9a825", # Yellow/Orange
    "High": "#ff4b4b"    # Red
}

# Colors for different message types
MESSAGE_COLORS = {
    "General": "#4f8bf9",       # Blue
    "Task Update": "#f9a825",   # Yellow/Orange
    "Announcement": "#ff4b4b",  # Red
    "Question": "#9c27b0",      # Purple
    "Response": "#00bcd4",      # Cyan
}

def load_messages():
    """Load messages from JSON file"""
    os.makedirs(os.path.dirname(MESSAGES_FILE), exist_ok=True)
    try:
        with open(MESSAGES_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_messages(messages):
    """Save messages to JSON file"""
    with open(MESSAGES_FILE, "w") as f:
        json.dump(messages, f, indent=4)

def load_tasks():
    """Load tasks from JSON file"""
    os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
    try:
        with open(TASKS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_tasks(tasks):
    """Save tasks to JSON file"""
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

def collaboration_tools():
    st.title("👥 Collaboration Tools")
    st.markdown("Communicate with your team and manage shared tasks")
    
    # Create tabs for different collaboration features
    tab1, tab2, tab3 = st.tabs(["💬 Team Chat", "🔄 Shared Tasks", "📊 Activity Dashboard"])
    
    with tab1:
        messages = load_messages()
        
        # Message board section
        st.markdown("### Team Chat")
        
        # Message input form
        with st.form("message_form", clear_on_submit=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                message_text = st.text_area("Message", height=100, placeholder="Type your message here...")
                
            with col2:
                username = st.text_input("Your Name", value=st.session_state.get("last_username", ""))
                message_type = st.selectbox(
                    "Message Type", 
                    options=["General", "Task Update", "Announcement", "Question", "Response"]
                )
                
            submit = st.form_submit_button("Send Message", use_container_width=True)
            
            if submit and message_text and username:
                st.session_state["last_username"] = username
                
                # Create new message
                new_message = {
                    "id": str(uuid.uuid4()),
                    "text": message_text,
                    "username": username,
                    "type": message_type,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add to messages and save
                messages.append(new_message)
                save_messages(messages)
                st.success("Message sent!")
                st.rerun()
                
        # Display messages with filtering options
        if messages:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                # Filter options
                filter_username = st.text_input("Filter by User", "")
                filter_type = st.multiselect(
                    "Filter by Type", 
                    options=["General", "Task Update", "Announcement", "Question", "Response"],
                    default=[]
                )
                
                # Sort options
                sort_order = st.radio(
                    "Sort Order",
                    options=["Newest First", "Oldest First"],
                    horizontal=True
                )
                
            # Filter messages
            filtered_messages = messages
            if filter_username:
                filtered_messages = [m for m in filtered_messages if filter_username.lower() in m["username"].lower()]
            if filter_type:
                filtered_messages = [m for m in filtered_messages if m["type"] in filter_type]
                
            # Sort messages
            filtered_messages = sorted(
                filtered_messages, 
                key=lambda m: m["timestamp"],
                reverse=(sort_order == "Newest First")
            )
            
            # Display messages
            if not filtered_messages:
                st.info("No messages match your filter criteria.")
            else:
                st.markdown(f"### {len(filtered_messages)} Messages")
                for message in filtered_messages:
                    with st.container():
                        # Message header
                        message_color = MESSAGE_COLORS.get(message["type"], "#4f8bf9")
                        st.markdown(f"""
                        <div style="
                            background-color: #1e2130; 
                            border-left: 4px solid {message_color}; 
                            border-radius: 4px; 
                            padding: 10px; 
                            margin-bottom: 10px;
                        ">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: bold; color: white;">{message["username"]}</span>
                                <span style="color: #888888;">{message["timestamp"]}</span>
                            </div>
                            <div style="
                                display: inline-block; 
                                background-color: {message_color}; 
                                color: white; 
                                padding: 2px 8px; 
                                border-radius: 10px; 
                                font-size: 0.8em; 
                                margin-bottom: 5px;
                            ">
                                {message["type"]}
                            </div>
                            <div style="margin-top: 5px; white-space: pre-wrap;">{message["text"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No messages found. Be the first to post!")

    with tab2:
        tasks = load_tasks()
        
        st.markdown("### Shared Tasks")
        st.markdown("Collaborate on tasks with your team members")
        
        # Task assignment form
        with st.form("task_assignment_form"):
            st.markdown("#### Assign a New Task")
            
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Task Title", max_chars=100)
                description = st.text_area("Description", max_chars=500)
                
            with col2:
                assignee = st.text_input("Assign To")
                priority = st.selectbox("Priority", options=["Low", "Medium", "High"])
                status = st.selectbox("Status", options=["Not Started", "In Progress", "Testing", "Completed"])
                
            due_date = st.date_input(
                "Due Date (Optional)",
                value=None,
                min_value=date.today(),
                format="YYYY-MM-DD"
            )
            
            submit_button = st.form_submit_button(label="Assign Task")
            
            if submit_button:
                if not title or not assignee:
                    st.error("Title and Assignee are required!")
                else:
                    # Convert date to string if it exists
                    due_date_str = due_date.strftime("%Y-%m-%d") if due_date else None
                    
                    # Create new task with unique ID
                    task_id = str(uuid.uuid4())
                    tasks[task_id] = {
                        "title": title,
                        "description": description,
                        "assignee": assignee,
                        "status": status,
                        "priority": priority,
                        "due_date": due_date_str,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "created_by": st.session_state.get("last_username", "Unknown")
                    }
                    
                    save_tasks(tasks)
                    
                    # Add a message about the new task
                    messages = load_messages()
                    task_message = {
                        "id": str(uuid.uuid4()),
                        "text": f"New task assigned: {title}\nAssigned to: {assignee}\nPriority: {priority}",
                        "username": st.session_state.get("last_username", "System"),
                        "type": "Task Update",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    messages.append(task_message)
                    save_messages(messages)
                    
                    st.success("Task assigned successfully!")
                    st.rerun()
                    
        # Task list
        if tasks:
            st.markdown("#### Current Tasks")
            
            # Filter options
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                filter_assignee = st.text_input("Filter by Assignee", "")
            
            with filter_col2:
                filter_status = st.multiselect(
                    "Filter by Status",
                    options=["Not Started", "In Progress", "Testing", "Completed"],
                    default=[]
                )
                
            with filter_col3:
                filter_priority = st.multiselect(
                    "Filter by Priority",
                    options=["Low", "Medium", "High"],
                    default=[]
                )
                
            # Filter tasks
            filtered_tasks = {}
            for task_id, task in tasks.items():
                if filter_assignee and filter_assignee.lower() not in task.get("assignee", "").lower():
                    continue
                if filter_status and task.get("status") not in filter_status:
                    continue
                if filter_priority and task.get("priority") not in filter_priority:
                    continue
                filtered_tasks[task_id] = task
                
            if not filtered_tasks:
                st.info("No tasks match your filter criteria.")
            else:
                # Sort tasks by priority then due date
                sorted_tasks = sorted(
                    filtered_tasks.items(),
                    key=lambda x: (
                        {"High": 0, "Medium": 1, "Low": 2}.get(x[1].get("priority"), 3),
                        x[1].get("due_date", "9999-12-31")
                    )
                )
                
                # Display tasks
                for task_id, task in sorted_tasks:
                    with st.container():
                        priority_color = PRIORITY_COLORS.get(task.get("priority", "Medium"))
                        
                        with st.expander(
                            f"{task.get('title')} ({task.get('assignee', 'Unassigned')})", 
                            expanded=False
                        ):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Description:** {task.get('description', 'No description')}")
                                
                                # Priority and status badges
                                st.markdown(f"""
                                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                                    <span style="
                                        background-color: {priority_color}; 
                                        padding: 3px 8px; 
                                        border-radius: 20px; 
                                        font-size: 0.8em;
                                    ">
                                        {task.get('priority', 'Medium')} Priority
                                    </span>
                                    <span style="
                                        background-color: #4f8bf9; 
                                        padding: 3px 8px; 
                                        border-radius: 20px; 
                                        font-size: 0.8em;
                                    ">
                                        {task.get('status', 'Not Started')}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Task details
                                due_text = task.get('due_date', 'No due date')
                                st.markdown(f"**Due Date:** {due_text}")
                                st.markdown(f"**Created By:** {task.get('created_by', 'Unknown')}")
                                st.markdown(f"**Created At:** {task.get('created_at', '')}")
                                
                            with col2:
                                # Update task status
                                new_status = st.selectbox(
                                    "Update Status",
                                    options=["Not Started", "In Progress", "Testing", "Completed"],
                                    index=["Not Started", "In Progress", "Testing", "Completed"].index(task.get('status', 'Not Started')),
                                    key=f"status_{task_id}"
                                )
                                
                                if new_status != task.get('status'):
                                    task['status'] = new_status
                                    save_tasks(tasks)
                                    
                                    # Add status update message
                                    messages = load_messages()
                                    status_message = {
                                        "id": str(uuid.uuid4()),
                                        "text": f"Task status updated: {task.get('title')}\nNew status: {new_status}",
                                        "username": st.session_state.get("last_username", "System"),
                                        "type": "Task Update",
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    messages.append(status_message)
                                    save_messages(messages)
                                    
                                    st.rerun()
                                
                                # Delete task button
                                if st.button("Delete Task", key=f"delete_{task_id}"):
                                    # Add deletion message
                                    messages = load_messages()
                                    deletion_message = {
                                        "id": str(uuid.uuid4()),
                                        "text": f"Task deleted: {task.get('title')}\nPreviously assigned to: {task.get('assignee')}",
                                        "username": st.session_state.get("last_username", "System"),
                                        "type": "Task Update",
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    messages.append(deletion_message)
                                    save_messages(messages)
                                    
                                    # Delete the task
                                    del tasks[task_id]
                                    save_tasks(tasks)
                                    st.rerun()
        else:
            st.info("No tasks found. Assign a task to get started!")
    
    with tab3:
        st.markdown("### Activity Dashboard")
        
        # Load data for dashboard
        messages = load_messages()
        tasks = load_tasks()
        
        if not messages and not tasks:
            st.info("No activity data available yet. Start collaborating to see analytics!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Message activity over time
                if messages:
                    st.markdown("#### Message Activity")
                    
                    # Prepare data
                    df_messages = pd.DataFrame(messages)
                    df_messages['date'] = pd.to_datetime(df_messages['timestamp']).dt.date
                    
                    # Count messages by date and type
                    message_counts = df_messages.groupby(['date', 'type']).size().reset_index(name='count')
                    
                    # Create line chart
                    fig = px.line(
                        message_counts, 
                        x='date', 
                        y='count', 
                        color='type',
                        title='Messages Over Time',
                        color_discrete_map=MESSAGE_COLORS
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Number of Messages",
                        legend_title="Message Type",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Message distribution by type
                    type_counts = df_messages['type'].value_counts().reset_index()
                    type_counts.columns = ['type', 'count']
                    
                    fig = px.pie(
                        type_counts,
                        values='count',
                        names='type',
                        title='Message Types Distribution',
                        color='type',
                        color_discrete_map=MESSAGE_COLORS
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Task status distribution
                if tasks:
                    st.markdown("#### Task Analytics")
                    
                    # Prepare data
                    df_tasks = pd.DataFrame(tasks.values())
                    
                    # Task status distribution
                    if 'status' in df_tasks.columns:
                        status_counts = df_tasks['status'].value_counts().reset_index()
                        status_counts.columns = ['status', 'count']
                        
                        fig = px.pie(
                            status_counts,
                            values='count',
                            names='status',
                            title='Task Status Distribution',
                            color='status',
                            color_discrete_map={
                                'Not Started': '#ff4b4b',
                                'In Progress': '#f9a825',
                                'Testing': '#4f8bf9',
                                'Completed': '#0cce6b'
                            }
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Task priority distribution
                    if 'priority' in df_tasks.columns:
                        priority_counts = df_tasks['priority'].value_counts().reset_index()
                        priority_counts.columns = ['priority', 'count']
                        
                        fig = px.bar(
                            priority_counts,
                            x='priority',
                            y='count',
                            title='Tasks by Priority',
                            color='priority',
                            color_discrete_map=PRIORITY_COLORS
                        )
                        fig.update_layout(
                            xaxis_title="Priority",
                            yaxis_title="Number of Tasks",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Top contributors
            if messages:
                st.markdown("#### Top Contributors")
                
                user_counts = df_messages['username'].value_counts().reset_index()
                user_counts.columns = ['username', 'message_count']
                user_counts = user_counts.head(10)  # Top 10 users
                
                fig = px.bar(
                    user_counts,
                    x='username',
                    y='message_count',
                    title='Top Message Contributors',
                    color='message_count',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(
                    xaxis_title="Username",
                    yaxis_title="Number of Messages",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
