#!/bin/bash

# Smart CV Filter - Setup Script
# This script sets up the development environment

echo "🚀 Setting up Smart CV Filter..."

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OpenAI API key"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p chroma_db
mkdir -p uploads

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run apps/streamlit_app.py"
echo ""
echo "To configure OpenAI API key:"
echo "  Edit .env file and add: OPENAI_API_KEY=your_key_here"
echo ""
