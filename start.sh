#!/bin/bash

# Smart CV Filter - Easy Launch Script
# This script automatically runs the application

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║              🎯 Smart CV Filter Launcher                     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "apps/streamlit_app.py" ]; then
    echo "❌ Error: Please run this script from the cv-filtering directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo ""
    echo "Would you like to set it up now? (y/n)"
    read -r response
    
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo ""
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
        
        echo "📥 Installing dependencies..."
        source venv/bin/activate
        pip install -r requirements.txt
        
        echo ""
        echo "✅ Setup complete!"
    else
        echo ""
        echo "Please run ./setup.sh first to set up the environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not installed!"
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "🚀 Launching Smart CV Filter..."
echo ""
echo "   📱 Opening in browser: http://localhost:8501"
echo "   ⌨️  Press Ctrl+C to stop the server"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Launch Streamlit
streamlit run apps/streamlit_app.py

echo ""
echo "👋 Application stopped. Thank you for using Smart CV Filter!"
