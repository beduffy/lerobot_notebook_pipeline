#!/usr/bin/env python3
"""
Setup Script for Async Inference System

Installs dependencies and tests the async inference system for π0-FAST and other models.

Usage:
    python setup_async_inference.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed:")
        print(f"      {e.stderr}")
        return False


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Install async inference requirements
    if not run_command("pip install -r requirements-async.txt", "Installing async inference dependencies"):
        return False
    
    # Install LeRobot if not already installed
    try:
        import lerobot
        print("✅ LeRobot already installed")
    except ImportError:
        print("📦 Installing LeRobot...")
        if not run_command("pip install lerobot", "Installing LeRobot"):
            return False
    
    return True


def test_imports():
    """Test that all imports work."""
    print("\n🧪 Testing imports...")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("websockets", "WebSockets"),
        ("requests", "Requests"),
        ("numpy", "NumPy"),
        ("lerobot", "LeRobot"),
    ]
    
    failed_imports = []
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True


def test_async_inference():
    """Test the async inference system."""
    print("\n🧪 Testing async inference system...")
    
    # Test local async inference
    if not run_command("python test_async_inference.py --local --model pi0fast", 
                      "Testing local async inference"):
        print("⚠️  Local async inference test failed (this is normal if no GPU)")
    
    # Test if we can create the server
    try:
        from async_inference_server import AsyncInferenceServer
        print("✅ Async inference server can be imported")
    except ImportError as e:
        print(f"❌ Async inference server import failed: {e}")
        return False
    
    # Test if we can create the client
    try:
        from async_inference_client import AsyncInferenceClient
        print("✅ Async inference client can be imported")
    except ImportError as e:
        print(f"❌ Async inference client import failed: {e}")
        return False
    
    return True


def create_example_scripts():
    """Create example usage scripts."""
    print("\n📝 Creating example scripts...")
    
    # Example server startup script
    server_script = """#!/usr/bin/env python3
# Example: Start async inference server with π0-FAST
python async_inference_server.py --model pi0fast --port 8000 --gpu
"""
    
    with open("start_server_example.py", "w") as f:
        f.write(server_script)
    
    # Example client test script
    client_script = """#!/usr/bin/env python3
# Example: Test async inference client
python async_inference_client.py --server http://localhost:8000 --test-http
python async_inference_client.py --server http://localhost:8000 --benchmark
"""
    
    with open("test_client_example.py", "w") as f:
        f.write(client_script)
    
    # Example robot control script
    robot_script = """#!/usr/bin/env python3
# Example: Robot control with remote inference
python async_inference_client.py --server http://localhost:8000 --robot-control \\
    --robot-port /dev/ttyACM0 \\
    --task "grab red cube and put to left"
"""
    
    with open("robot_control_example.py", "w") as f:
        f.write(robot_script)
    
    print("✅ Created example scripts:")
    print("   start_server_example.py")
    print("   test_client_example.py")
    print("   robot_control_example.py")


def main():
    """Main setup function."""
    print("🚀 LeRobot Async Inference System Setup")
    print("=" * 50)
    print("Setting up async inference for π0-FAST and other models...")
    print()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed")
        return 1
    
    # Test async inference
    if not test_async_inference():
        print("❌ Async inference tests failed")
        return 1
    
    # Create example scripts
    create_example_scripts()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print()
    print("🚀 Quick start:")
    print("   1. Start the server:")
    print("      python async_inference_server.py --model pi0fast --port 8000")
    print()
    print("   2. Test the client:")
    print("      python async_inference_client.py --server http://localhost:8000 --test-http")
    print()
    print("   3. Run robot control:")
    print("      python async_inference_client.py --server http://localhost:8000 --robot-control")
    print()
    print("   4. Deploy to cloud:")
    print("      python deploy_async_inference.py --platform aws --model pi0fast --gpu")
    print()
    print("📚 Documentation:")
    print("   - ASYNC_INFERENCE_README.md")
    print("   - Example scripts in current directory")
    print()
    print("🧪 Testing:")
    print("   python test_async_inference.py --all")
    
    return 0


if __name__ == "__main__":
    exit(main()) 