#!/usr/bin/env python3
"""
Run script for Smart CV Filter application.
This provides a simple way to start the application.
"""

import sys
import subprocess
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def main():
    """Main entry point."""
    print("🚀 Starting Smart CV Filter...")
    print()
    
    # Check if in project directory
    project_root = Path(__file__).parent
    app_path = project_root / "apps" / "streamlit_app.py"
    
    if not app_path.exists():
        print("❌ Error: Cannot find streamlit_app.py")
        print(f"   Expected location: {app_path}")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("⚠️  Warning: Dependencies not installed!")
        print()
        print("Please run:")
        print("  pip install -r requirements.txt")
        print()
        print("or use the setup script:")
        print("  ./setup.sh (macOS/Linux)")
        print("  setup.bat (Windows)")
        print()
        sys.exit(1)
    
    # Run Streamlit
    print("✅ All checks passed!")
    print("📱 Launching Streamlit application...")
    print()
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
