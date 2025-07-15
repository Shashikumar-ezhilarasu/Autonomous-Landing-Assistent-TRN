#!/bin/bash
# Sample commands to run the application

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Run the Streamlit application
echo "Starting Autonomous Landing Assessment System..."
streamlit run app.py --server.port 8501 --server.address localhost

# Alternative run command with custom configuration
# streamlit run app.py --server.port 8502 --server.headless true
