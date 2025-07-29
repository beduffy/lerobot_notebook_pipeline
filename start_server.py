#!/usr/bin/env python3
"""
Simple Async Inference Server

Start async inference server locally.

Usage:
    python3 start_server.py --model act --port 8000
"""

import argparse
import sys
from pathlib import Path

# Add async folder to path
sys.path.insert(0, str(Path(__file__).parent / "async"))

try:
    from async_inference_server import AsyncInferenceServer
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"Async server not available: {e}")
    ASYNC_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Start async inference server")
    parser.add_argument("--model", default="act", 
                       choices=["act", "diffusion", "vqbet", "smolvla", "pi0fast"],
                       help="Model to load")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    if not ASYNC_AVAILABLE:
        print("❌ Async server not available")
        return 1
    
    print("🚀 Starting Async Inference Server")
    print("=" * 40)
    print(f"Model: {args.model.upper()}")
    print(f"Port: {args.port}")
    print(f"Host: {args.host}")
    print(f"Device: {args.device}")
    print()
    
    try:
        # Create and start server
        server = AsyncInferenceServer(
            model_type=args.model,
            device=args.device
        )
        
        print("✅ Server initialized successfully!")
        print(f"🔗 Server will be available at:")
        print(f"   HTTP: http://{args.host}:{args.port}")
        print(f"   WebSocket: ws://{args.host}:8765")
        print(f"   API docs: http://{args.host}:{args.port}/docs")
        print()
        print("🛑 Press Ctrl+C to stop the server")
        print()
        
        # Start server
        server.run_http_server(host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 