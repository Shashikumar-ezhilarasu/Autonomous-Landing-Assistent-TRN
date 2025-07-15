#!/bin/bash

echo "🚀 Starting Autonomous Landing Assessment System..."
echo "=========================================="

# Change to app directory
cd /Users/shashikumarezhil/Documents/Autonomous-Lander/unified-landing-app

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, cv2, numpy, matplotlib, plotly, skimage" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All dependencies are installed"
else
    echo "❌ Installing missing dependencies..."
    pip3 install -r requirements.txt
fi

# Run the application
echo "🌐 Starting web application..."
echo "📍 The app will be available at: http://localhost:8501"
echo "🔧 To stop the app, press Ctrl+C"
echo ""

# Start Streamlit
streamlit run app.py --server.port 8501
