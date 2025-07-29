#!/usr/bin/env python3
"""
Simple Async Inference Client Test

Test async inference server (local or cloud) and profile FPS.

Usage:
    # Test local server
    python3 test_client.py --url http://localhost:8000
    
    # Test cloud server
    python3 test_client.py --url http://your-cloud-url:8000
    
    # Profile FPS
    python3 test_client.py --url http://localhost:8000 --profile
"""

import argparse
import time
import numpy as np
import requests
import json
from pathlib import Path

# Add async folder to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "async"))

try:
    from async_inference_client import AsyncInferenceClient
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"Async client not available: {e}")
    ASYNC_AVAILABLE = False

def create_test_observations():
    """Create test observations for inference."""
    return {
        "observation.images.front": np.random.randint(0, 255, (3, 96, 96), dtype=np.uint8).tolist(),
        "observation.state": np.random.randn(6).tolist()
    }

def test_server_health(url):
    """Test if server is running."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False

def test_single_inference(url):
    """Test single inference request."""
    if not ASYNC_AVAILABLE:
        print("❌ Async client not available")
        return False
    
    try:
        client = AsyncInferenceClient(url)
        
        # Get server info
        server_info = client.get_server_info()
        if "error" in server_info:
            print(f"❌ Server error: {server_info['error']}")
            return False
        
        print(f"✅ Server status: {server_info.get('status', 'unknown')}")
        
        # Get model info
        model_info = client.get_model_info()
        print(f"✅ Model: {model_info.get('model_type', 'unknown')}")
        print(f"✅ Parameters: {model_info.get('parameters', 'unknown'):,}")
        
        # Test inference
        observations = create_test_observations()
        result = client.test_http_inference(observations)
        
        if result.status == "success":
            print(f"✅ Inference successful!")
            print(f"   Action shape: {result.actions.shape}")
            print(f"   Inference time: {result.inference_time:.3f}s")
            print(f"   Total latency: {result.latency:.3f}s")
            return True
        else:
            print(f"❌ Inference failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def profile_fps(url, num_requests=100):
    """Profile FPS with multiple requests."""
    if not ASYNC_AVAILABLE:
        print("❌ Async client not available")
        return False
    
    try:
        client = AsyncInferenceClient(url)
        observations = create_test_observations()
        
        print(f"🔄 Profiling FPS with {num_requests} requests...")
        
        times = []
        successful = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            request_start = time.time()
            result = client.test_http_inference(observations)
            request_time = time.time() - request_start
            
            if result.status == "success":
                times.append(request_time)
                successful += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{num_requests}")
        
        total_time = time.time() - start_time
        
        if times:
            times = np.array(times)
            mean_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / mean_time
            
            print(f"\n📊 FPS Profile Results:")
            print(f"   Successful requests: {successful}/{num_requests}")
            print(f"   Mean inference time: {mean_time:.3f}s ± {std_time:.3f}s")
            print(f"   FPS: {fps:.1f}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Requests/second: {successful/total_time:.1f}")
            
            return True
        else:
            print("❌ No successful requests")
            return False
            
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test async inference server")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Server URL (local or cloud)")
    parser.add_argument("--profile", action="store_true", 
                       help="Profile FPS with multiple requests")
    parser.add_argument("--requests", type=int, default=100,
                       help="Number of requests for profiling")
    
    args = parser.parse_args()
    
    print("🧪 Async Inference Client Test")
    print("=" * 40)
    print(f"Server URL: {args.url}")
    print()
    
    # Test server health
    print("1. Testing server health...")
    if not test_server_health(args.url):
        print("❌ Server is not running or not reachable")
        print(f"   Start server with: python3 async/async_inference_server.py --model act --port 8000")
        return 1
    
    # Test single inference
    print("\n2. Testing single inference...")
    if not test_single_inference(args.url):
        print("❌ Single inference test failed")
        return 1
    
    # Profile FPS if requested
    if args.profile:
        print("\n3. Profiling FPS...")
        if not profile_fps(args.url, args.requests):
            print("❌ FPS profiling failed")
            return 1
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    exit(main()) 