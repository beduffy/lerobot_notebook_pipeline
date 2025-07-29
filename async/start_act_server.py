#!/usr/bin/env python3
"""
Start ACT Async Inference Server

Quick script to start the ACT async inference server for testing.

Usage:
    python start_act_server.py
"""

import sys
import subprocess
from pathlib import Path

# Add async folder to path
sys.path.insert(0, str(Path(__file__).parent / "async"))

def main():
    """Start ACT async inference server."""
    print("🚀 Starting ACT Async Inference Server...")
    print("=" * 50)
    
    # Check if async folder exists
    async_folder = Path(__file__).parent / "async"
    if not async_folder.exists():
        print("❌ Async folder not found!")
        print("   Make sure you're in the correct directory")
        return 1
    
    # Check if server script exists
    server_script = async_folder / "async_inference_server.py"
    if not server_script.exists():
        print("❌ Server script not found!")
        print(f"   Expected: {server_script}")
        return 1
    
    print("✅ Found async inference server")
    print("   Starting ACT model on port 8000...")
    print()
    print("🔗 Server will be available at:")
    print("   HTTP: http://localhost:8000")
    print("   WebSocket: ws://localhost:8765")
    print()
    print("📖 API docs: http://localhost:8000/docs")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Start the server
        cmd = [
            sys.executable, 
            str(server_script),
            "--model", "act",
            "--port", "8000",
            "--host", "0.0.0.0"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print()
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 