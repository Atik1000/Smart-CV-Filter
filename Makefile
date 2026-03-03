# Makefile for Smart CV Filter

.PHONY: help setup install run test clean format lint

help:
	@echo "Smart CV Filter - Available Commands:"
	@echo "  make setup    - Set up virtual environment and install dependencies"
	@echo "  make install  - Install dependencies only"
	@echo "  make run      - Run the Streamlit application"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean up temporary files and caches"
	@echo "  make format   - Format code with black"
	@echo "  make lint     - Lint code with flake8"

setup:
	@echo "Setting up Smart CV Filter..."
	python3 -m venv venv
	@echo "Virtual environment created. Activate it with: source venv/bin/activate"
	@echo "Then run: make install"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

run:
	streamlit run apps/streamlit_app.py

test:
	pytest tests/ -v

test-quick:
	python tests/test_analyze_cv.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf dist build
	@echo "Cleanup complete!"

format:
	black apps/ utils/ tests/
	@echo "Code formatted successfully!"

lint:
	flake8 apps/ utils/ tests/ --max-line-length=100
	@echo "Linting complete!"

dev-setup: setup install
	@echo "Development environment ready!"
	@echo "Activate with: source venv/bin/activate"
	@echo "Run with: make run"
