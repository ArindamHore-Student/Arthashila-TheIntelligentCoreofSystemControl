#!/bin/bash

echo "========================================"
echo "   Arthashila Installation Script"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .

echo "========================================"
echo "Installation completed!"
echo "To start Arthashila, run:"
echo "./run_arthashila.sh"
echo "========================================"

# Create launch script
cat > run_arthashila.sh << 'EOL'
#!/bin/bash
source .venv/bin/activate
python run_app.py
EOL

chmod +x run_arthashila.sh

echo "Launch script created: run_arthashila.sh" 