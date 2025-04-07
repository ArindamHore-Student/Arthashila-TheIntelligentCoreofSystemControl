# Arthashila - System Monitor

A comprehensive system monitoring and task management application built with Streamlit. Arthashila provides real-time system metrics, process management, performance analytics, and task planning capabilities.

![Arthashila Logo](https://img.shields.io/badge/Arthashila-System%20Monitor-blue)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

### 🖥️ System Overview
- Real-time CPU, memory, and disk usage monitoring
- Detailed hardware and software information
- Interactive gauge charts for system metrics
- AI-powered insights integration

### 🔄 Process Manager
- Real-time process monitoring and management
- Process filtering and sorting capabilities
- Resource usage tracking
- Process optimization suggestions

### 📊 Performance Graphs
- Real-time CPU and memory usage graphs
- AI-powered anomaly detection
- Predictive analytics for resource usage
- Customizable refresh rates

### 🤖 AI Analytics
- Intelligent process optimization
- Anomaly detection and alerts
- Performance trend analysis
- Resource usage forecasting

### 🔋 Battery & Power
- Battery status monitoring
- Power management tips
- Energy usage optimization
- Battery health information

### 📝 Task Planning
- Task creation and management
- Priority-based task organization
- Progress tracking
- Task analytics and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arthashila.git
cd arthashila
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python -m streamlit run main.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

## Requirements

- Python 3.8 or higher
- Streamlit
- psutil
- plotly
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib

## Project Structure

```
arthashila/
├── features/
│   ├── system_overview.py
│   ├── process_manager.py
│   ├── performance_graphs.py
│   ├── ai_analytics.py
│   ├── battery_management.py
│   └── task_planning.py
├── utils/
│   └── __init__.py
├── data/
│   └── tasks.json
├── main.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [psutil](https://github.com/giampaolo/psutil) for system monitoring
- Inspired by modern system monitoring tools

## Support

For support, please open an issue in the GitHub repository or contact the development team. 