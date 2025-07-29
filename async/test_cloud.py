#!/usr/bin/env python3
"""
Test Cloud Async Inference Server

Simple script to test the cloud server from your laptop.

Usage:
    python3 test_cloud.py --url http://your-cloud-url:8000
"""

import argparse
import requests
import time
import json

def test_cloud_server(url):
    """Test cloud server health and inference."""
    print(f"🧪 Testing cloud server: {url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test model info
    try:
        response = requests.get(f"{url}/model_info", timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model: {model_info.get('model_type', 'unknown')}")
            print(f"✅ Parameters: {model_info.get('parameters', 'unknown'):,}")
        else:
            print(f"❌ Model info error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False
    
    # Test inference
    try:
        # Create test observations
        test_data = {
            "observations": {
                "observation.images.front": [[[[0] * 96] * 96] * 3],  # Simple test image
                "observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Test state
            },
            "task_description": None,
            "priority": 1
        }
        
        print("🔄 Testing inference...")
        start_time = time.time()
        
        response = requests.post(
            f"{url}/inference",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            inference_time = time.time() - start_time
            
            print("✅ Inference successful!")
            print(f"   Action shape: {len(result.get('actions', []))}")
            print(f"   Inference time: {result.get('inference_time', 0):.3f}s")
            print(f"   Total latency: {inference_time:.3f}s")
            print(f"   FPS: {1.0/result.get('inference_time', 1):.1f}")
            
            return True
        else:
            print(f"❌ Inference failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def profile_fps(url, num_requests=50):
    """Profile FPS with multiple requests."""
    print(f"🔄 Profiling FPS with {num_requests} requests...")
    
    test_data = {
        "observations": {
            "observation.images.front": [[[[0] * 96] * 96] * 3],
            "observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "task_description": None,
        "priority": 1
    }
    
    times = []
    successful = 0
    
    start_time = time.time()
    
    for i in range(num_requests):
        request_start = time.time()
        
        try:
            response = requests.post(
                f"{url}/inference",
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                request_time = time.time() - request_start
                times.append(request_time)
                successful += 1
            else:
                print(f"   Request {i+1} failed: {response.status_code}")
                
        except Exception as e:
            print(f"   Request {i+1} error: {e}")
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{num_requests}")
    
    total_time = time.time() - start_time
    
    if times:
        import numpy as np
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

def main():
    parser = argparse.ArgumentParser(description="Test cloud async inference server")
    parser.add_argument("--url", required=True, help="Cloud server URL")
    parser.add_argument("--profile", action="store_true", help="Profile FPS")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests for profiling")
    
    args = parser.parse_args()
    
    print("🌐 Cloud Async Inference Test")
    print("=" * 40)
    print(f"Server URL: {args.url}")
    print()
    
    # Test basic functionality
    if not test_cloud_server(args.url):
        print("❌ Basic test failed")
        return 1
    
    # Profile FPS if requested
    if args.profile:
        print("\n" + "=" * 40)
        if not profile_fps(args.url, args.requests):
            print("❌ FPS profiling failed")
            return 1
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    exit(main()) 