import streamlit as st # type: ignore
from datetime import datetime, date, timedelta
import sys
import os
import pandas as pd
import uuid
import json
import plotly.express as px
import plotly.graph_objects as go

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import format_priority

# Task statuses with colors
TASK_STATUSES = {
    "Not Started": {"color": "#ff4b4b", "icon": "🔴"},
    "In Progress": {"color": "#f9a825", "icon": "🟠"},
    "Testing": {"color": "#4f8bf9", "icon": "🔵"},
    "Completed": {"color": "#0cce6b", "icon": "🟢"}
}

# Task priorities with colors
TASK_PRIORITIES = {
    "Low": {"color": "#0cce6b", "icon": "⬇️"},
    "Medium": {"color": "#f9a825", "icon": "➡️"},
    "High": {"color": "#ff4b4b", "icon": "⬆️"}
}

def load_tasks():
    try:
        with open('data/tasks.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        os.makedirs('data', exist_ok=True)
        return {}

def save_tasks(tasks):
    with open('data/tasks.json', 'w') as f:
        json.dump(tasks, f, indent=4)

def task_planning():
    st.title("📝 Task Planning")
    st.markdown("Manage your project tasks and track progress effectively")
    
    # Load tasks
    tasks = load_tasks()
    
    # Set up tabs for different task views
    tab1, tab2, tab3, tab4 = st.tabs(["✨ Task Dashboard", "➕ Add Task", "📊 Task Analysis", "🔧 Configure"])

    with tab1:
        # Dashboard
        if not tasks:
            st.info("No tasks found. Add some tasks to get started!")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Task counts by status
            status_counts = {}
            for task_id, task in tasks.items():
                status = task['status']
                status_counts[status] = status_counts.get(status, 0) + 1
                
            with col1:
                total_tasks = len(tasks)
                completed_tasks = status_counts.get("Completed", 0)
                completion_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
                
                st.markdown(f"""
                <div style='background-color:#1e2130; padding:15px; border-radius:10px; text-align:center; border:1px solid #2d3747;'>
                    <h3 style='margin:0;'>Total Tasks</h3>
                    <div style='font-size:3rem; font-weight:bold;'>{total_tasks}</div>
                    <div style='color:#4f8bf9;'>{completion_percentage:.1f}% Complete</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Upcoming deadlines
                st.markdown(f"""
                <div style='background-color:#1e2130; padding:15px; border-radius:10px; text-align:center; border:1px solid #2d3747;'>
                    <h3 style='margin:0;'>Tasks In Progress</h3>
                    <div style='font-size:3rem; font-weight:bold;'>{status_counts.get("In Progress", 0)}</div>
                    <div style='color:#f9a825;'>Active Tasks</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                # Highest priority tasks
                high_priority_count = sum(1 for task in tasks.values() if task['priority'] == "High")
                st.markdown(f"""
                <div style='background-color:#1e2130; padding:15px; border-radius:10px; text-align:center; border:1px solid #2d3747;'>
                    <h3 style='margin:0;'>High Priority</h3>
                    <div style='font-size:3rem; font-weight:bold;'>{high_priority_count}</div>
                    <div style='color:#ff4b4b;'>Urgent Tasks</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Status distribution chart
            st.markdown("### Task Status Distribution")
            status_data = pd.DataFrame({
                'Status': list(status_counts.keys()),
                'Count': list(status_counts.values()),
                'Color': [TASK_STATUSES[status]['color'] for status in status_counts.keys()]
            })
            
            fig = px.bar(
                status_data,
                x='Status',
                y='Count',
                color='Status',
                color_discrete_map={status: TASK_STATUSES[status]['color'] for status in status_counts.keys()},
                text='Count'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="",
                yaxis_title="Number of Tasks",
                showlegend=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Task filters
            st.markdown("### Task List")
            
            filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
            with filter_col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=list(TASK_STATUSES.keys()),
                    default=list(TASK_STATUSES.keys())
                )
            with filter_col2:
                priority_filter = st.multiselect(
                    "Filter by Priority",
                    options=list(TASK_PRIORITIES.keys()),
                    default=list(TASK_PRIORITIES.keys())
                )
            with filter_col3:
                sort_option = st.selectbox(
                    "Sort by",
                    options=["Priority (High to Low)", "Due Date (Earliest First)", "Status"],
                    index=0
                )
                
            # Sort and filter tasks
            filtered_tasks = {
                task_id: task for task_id, task in tasks.items()
                if task['status'] in status_filter and task['priority'] in priority_filter
            }
            
            sorted_tasks = list(filtered_tasks.items())
            if sort_option == "Priority (High to Low)":
                priority_order = {"High": 0, "Medium": 1, "Low": 2}
                sorted_tasks = sorted(sorted_tasks, key=lambda x: priority_order[x[1]['priority']])
            elif sort_option == "Due Date (Earliest First)":
                sorted_tasks = sorted(sorted_tasks, key=lambda x: x[1]['due_date'] if x[1]['due_date'] else "9999-12-31")
            elif sort_option == "Status":
                status_order = {"Not Started": 0, "In Progress": 1, "Testing": 2, "Completed": 3}
                sorted_tasks = sorted(sorted_tasks, key=lambda x: status_order[x[1]['status']])
            
            # Display tasks
            for task_id, task in sorted_tasks:
                with st.expander(
                    f"{TASK_STATUSES[task['status']]['icon']} {TASK_PRIORITIES[task['priority']]['icon']} {task['title']}",
                    expanded=False
                ):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {task['description']}")
                        st.markdown(f"""
                        <div style='display: flex; gap: 10px; margin-bottom: 10px;'>
                            <span style='background-color:{TASK_STATUSES[task['status']]['color']}; padding:3px 8px; border-radius:20px; font-size:0.8em;'>
                                {task['status']}
                            </span>
                            <span style='background-color:{TASK_PRIORITIES[task['priority']]['color']}; padding:3px 8px; border-radius:20px; font-size:0.8em;'>
                                {task['priority']} Priority
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        due_text = task['due_date'] if task['due_date'] else "No due date"
                        st.markdown(f"**Due Date:** {due_text}")
                        
                    with col2:
                        # Actions
                        new_status = st.selectbox(
                            "Update Status",
                            options=list(TASK_STATUSES.keys()),
                            index=list(TASK_STATUSES.keys()).index(task['status']),
                            key=f"status_{task_id}"
                        )
                        if new_status != task['status']:
                            task['status'] = new_status
                            save_tasks(tasks)
                            st.rerun()
                            
                        if st.button("Delete Task", key=f"delete_{task_id}"):
                            del tasks[task_id]
                            save_tasks(tasks)
                            st.rerun()

    with tab2:
        st.markdown("### Create New Task")
        
        with st.form("new_task_form"):
            title = st.text_input("Task Title", max_chars=100)
            description = st.text_area("Description", max_chars=500)
            
            col1, col2 = st.columns(2)
            with col1:
                priority = st.selectbox("Priority", options=list(TASK_PRIORITIES.keys()))
                
            with col2:
                status = st.selectbox("Initial Status", options=list(TASK_STATUSES.keys()))
                
            due_date = st.date_input(
                "Due Date (Optional)",
                value=None,
                min_value=date.today(),
                format="YYYY-MM-DD"
            )
            
            submit_button = st.form_submit_button(label="Add Task")
            
            if submit_button:
                if not title:
                    st.error("Title is required!")
                else:
                    # Convert date to string if it exists
                    due_date_str = due_date.strftime("%Y-%m-%d") if due_date else None
                    
                    # Create new task with unique ID
                    task_id = str(uuid.uuid4())
                    tasks[task_id] = {
                        "title": title,
                        "description": description,
                        "status": status,
                        "priority": priority,
                        "due_date": due_date_str,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    save_tasks(tasks)
                    st.success("Task added successfully!")
                    st.rerun()

    with tab3:
        if not tasks:
            st.info("No tasks found. Add some tasks to see analytics!")
        else:
            st.markdown("### Task Analytics")
            
            # Priority distribution pie chart
            priority_counts = {}
            for task in tasks.values():
                priority = task['priority']
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

            col1, col2 = st.columns(2)
                
            with col1:
                # Priority Distribution
                priority_df = pd.DataFrame({
                    'Priority': list(priority_counts.keys()),
                    'Count': list(priority_counts.values())
                })
                fig = px.pie(
                    priority_df,
                    values='Count',
                    names='Priority',
                    title="Tasks by Priority",
                    color='Priority',
                    color_discrete_map={
                        'High': TASK_PRIORITIES['High']['color'],
                        'Medium': TASK_PRIORITIES['Medium']['color'],
                        'Low': TASK_PRIORITIES['Low']['color']
                    },
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Status distribution
                status_df = pd.DataFrame({
                    'Status': list(status_counts.keys()),
                    'Count': list(status_counts.values())
                })
                fig = px.pie(
                    status_df,
                    values='Count',
                    names='Status',
                    title="Tasks by Status",
                    color='Status',
                    color_discrete_map={
                        status: TASK_STATUSES[status]['color'] for status in TASK_STATUSES
                    },
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Task Timeline / Burndown
            st.markdown("### Task Timeline")
            
            # Get tasks with due dates
            tasks_with_dates = [
                task for task in tasks.values() 
                if task['due_date'] is not None
            ]
            
            if tasks_with_dates:
                df = pd.DataFrame(tasks_with_dates)
                df['due_date'] = pd.to_datetime(df['due_date'])
                df = df.sort_values('due_date')
                
                # Timeline chart
                fig = go.Figure()
                
                for status in TASK_STATUSES:
                    status_tasks = df[df['status'] == status]
                    if not status_tasks.empty:
                        fig.add_trace(go.Bar(
                            x=status_tasks['due_date'],
                            y=[1] * len(status_tasks),
                            name=status,
                            marker_color=TASK_STATUSES[status]['color'],
                            customdata=status_tasks['title'],
                            hovertemplate='<b>%{customdata}</b><br>Due: %{x}<extra></extra>',
                            orientation='v',
                        ))
                
                fig.update_layout(
                    title="Task Due Dates Timeline",
                    xaxis_title="Due Date",
                    barmode='stack',
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400,
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)',
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        showgrid=False,
                    ),
                    hoverlabel=dict(
                        bgcolor='rgba(0,0,0,0.8)',
                        font_size=12,
                        font_color='white'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No tasks with due dates found. Add due dates to see timeline visualization.")

    with tab4:
        st.markdown("### Task Configuration")
        
        if st.button("Export Tasks (JSON)", type="primary"):
            st.download_button(
                label="Download Tasks Data",
                data=json.dumps(tasks, indent=4),
                file_name="tasks_export.json",
                mime="application/json"
            )
            
        if st.button("Clear All Tasks", type="secondary"):
            confirm = st.checkbox("I understand this will delete all tasks permanently")
            if confirm:
                tasks = {}
                save_tasks(tasks)
                st.success("All tasks have been cleared!")
                st.rerun()
                
        st.markdown("---")
        
        # Sample task templates
        st.markdown("### Task Templates")
        st.markdown("Click to add pre-defined task templates to your task list:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("+ Add Software Development Template"):
                templates = [
                    {
                        "title": "Requirements Analysis",
                        "description": "Gather and analyze requirements from stakeholders",
                        "status": "Not Started",
                        "priority": "High",
                        "due_date": (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")
                    },
                    {
                        "title": "Design Architecture",
                        "description": "Design system architecture and components",
                        "status": "Not Started",
                        "priority": "High",
                        "due_date": (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
                    },
                    {
                        "title": "Implementation",
                        "description": "Implement the designed components and features",
                        "status": "Not Started",
                        "priority": "Medium",
                        "due_date": (date.today() + timedelta(days=14)).strftime("%Y-%m-%d")
                    },
                    {
                        "title": "Testing",
                        "description": "Test the implemented components and features",
                        "status": "Not Started",
                        "priority": "Medium",
                        "due_date": (date.today() + timedelta(days=21)).strftime("%Y-%m-%d")
                    },
                    {
                        "title": "Documentation",
                        "description": "Create documentation for the system",
                        "status": "Not Started",
                        "priority": "Low",
                        "due_date": (date.today() + timedelta(days=25)).strftime("%Y-%m-%d")
                    }
                ]
                
                for template in templates:
                    task_id = str(uuid.uuid4())
                    template["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    tasks[task_id] = template
                
                save_tasks(tasks)
                st.success("Development templates added successfully!")
                st.rerun()
                
        with col2:
            if st.button("+ Add Project Management Template"):
                templates = [
                    {
                        "title": "Project Kickoff",
                        "description": "Initial meeting to start the project",
                        "status": "Not Started",
                        "priority": "High",
                        "due_date": (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
                    },
                    {
                        "title": "Sprint Planning",
                        "description": "Plan the upcoming sprint",
                        "status": "Not Started",
                        "priority": "High",
                        "due_date": (date.today() + timedelta(days=5)).strftime("%Y-%m-%d")
                    },
                    {
                        "title": "Daily Standups",
                        "description": "Regular daily meetings for status updates",
                        "status": "Not Started",
                        "priority": "Medium",
                        "due_date": None
                    },
                    {
                        "title": "Sprint Review",
                        "description": "Review sprint progress and demo",
                        "status": "Not Started",
                        "priority": "Medium",
                        "due_date": (date.today() + timedelta(days=19)).strftime("%Y-%m-%d")
                    },
                    {
                        "title": "Sprint Retrospective",
                        "description": "Reflect on the sprint and identify improvements",
                        "status": "Not Started",
                        "priority": "Low",
                        "due_date": (date.today() + timedelta(days=20)).strftime("%Y-%m-%d")
                    }
                ]
                
                for template in templates:
                    task_id = str(uuid.uuid4())
                    template["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    tasks[task_id] = template
                
                save_tasks(tasks)
                st.success("Project management templates added successfully!")
                st.rerun()
