#!/usr/bin/env python3
"""
Simple Cloud Deployment for Async Inference Server

Deploy to Lightning Cloud with GPU support.

Usage:
    python3 deploy_cloud.py --model act --port 8000
"""

import argparse
import subprocess
import sys
from pathlib import Path

def create_lightning_app():
    """Create Lightning app configuration."""
    app_code = '''import lightning as L
from async.async_inference_server import AsyncInferenceServer
import uvicorn

class AsyncInferenceApp(L.LightningWork):
    def __init__(self, model_type="act", port=8000):
        super().__init__(port=port)
        self.model_type = model_type
        self.server = None
    
    def run(self):
        # Initialize server
        self.server = AsyncInferenceServer(
            model_type=self.model_type,
            device="cuda"  # Use GPU on cloud
        )
        
        # Start server
        self.server.run_http_server(
            host="0.0.0.0",
            port=self.port
        )

# Create Lightning app
app = L.LightningApp(
    AsyncInferenceApp(
        model_type="act",  # Change this to test different models
        port=8000
    )
)
'''
    
    with open("lightning_app.py", "w") as f:
        f.write(app_code)
    
    print("✅ Created lightning_app.py")

def create_requirements():
    """Create requirements.txt for cloud deployment."""
    requirements = '''# Core dependencies
torch>=2.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
websockets>=11.0.0
requests>=2.28.0
numpy>=1.21.0

# LeRobot
git+https://github.com/huggingface/lerobot.git

# Lightning
lightning>=2.0.0

# Optional: GPU monitoring
psutil>=5.9.0
'''
    
    with open("requirements-cloud.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements-cloud.txt")

def deploy_to_lightning(model_type="act", port=8000):
    """Deploy to Lightning Cloud."""
    print(f"🚀 Deploying {model_type.upper()} async inference server to Lightning Cloud...")
    
    # Create necessary files
    create_lightning_app()
    create_requirements()
    
    # Update model type in app
    with open("lightning_app.py", "r") as f:
        content = f.read()
    
    content = content.replace('model_type="act"', f'model_type="{model_type}"')
    content = content.replace('port=8000', f'port={port}')
    
    with open("lightning_app.py", "w") as f:
        f.write(content)
    
    print(f"   Model: {model_type.upper()}")
    print(f"   Port: {port}")
    print(f"   GPU: Enabled")
    
    # Deploy command
    cmd = f"lightning run app lightning_app.py --cloud"
    print(f"\n📋 Run this command to deploy:")
    print(f"   {cmd}")
    print(f"\n🔗 Server will be available at:")
    print(f"   http://your-app-url:{port}")
    print(f"   ws://your-app-url:8765 (WebSocket)")
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Deploy async inference to Lightning Cloud")
    parser.add_argument("--model", default="act", 
                       choices=["act", "diffusion", "vqbet", "smolvla", "pi0fast"],
                       help="Model to deploy")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    args = parser.parse_args()
    
    print("⚡ Lightning Cloud Deployment")
    print("=" * 40)
    
    deploy_cmd = deploy_to_lightning(args.model, args.port)
    
    print("\n📖 Next steps:")
    print("1. Run the deployment command above")
    print("2. Wait for deployment to complete")
    print("3. Use the client script to test:")
    print(f"   python3 test_client.py --url http://your-app-url:{args.port}")

if __name__ == "__main__":
    main() 