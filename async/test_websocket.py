#!/usr/bin/env python3
"""
WebSocket Async Inference Test

Test WebSocket performance vs HTTP for real-time inference.

Usage:
    python3 test_websocket.py --url ws://localhost:8765 --profile
"""

import argparse
import asyncio
import websockets
import json
import time
import numpy as np

def create_test_observations():
    """Create test observations for inference."""
    return {
        "observation.images.front": np.random.randint(0, 255, (3, 96, 96), dtype=np.uint8).tolist(),
        "observation.state": np.random.randn(6).tolist()
    }

async def test_websocket_single(url):
    """Test single WebSocket inference."""
    print(f"🧪 Testing WebSocket: {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            # Test data
            test_data = {
                "observations": create_test_observations(),
                "task_description": None,
                "priority": 1
            }
            
            # Send request
            start_time = time.time()
            await websocket.send(json.dumps(test_data))
            
            # Receive response
            response = await websocket.recv()
            total_time = time.time() - start_time
            
            result = json.loads(response)
            
            print("✅ WebSocket inference successful!")
            print(f"   Action shape: {len(result.get('actions', []))}")
            print(f"   Inference time: {result.get('inference_time', 0):.3f}s")
            print(f"   Total latency: {total_time:.3f}s")
            print(f"   FPS: {1.0/result.get('inference_time', 1):.1f}")
            
            return True
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

async def profile_websocket_fps(url, num_requests=50):
    """Profile WebSocket FPS."""
    print(f"🔄 Profiling WebSocket FPS with {num_requests} requests...")
    
    try:
        async with websockets.connect(url) as websocket:
            times = []
            successful = 0
            
            start_time = time.time()
            
            for i in range(num_requests):
                request_start = time.time()
                
                # Test data
                test_data = {
                    "observations": create_test_observations(),
                    "task_description": None,
                    "priority": 1
                }
                
                try:
                    # Send request
                    await websocket.send(json.dumps(test_data))
                    
                    # Receive response
                    response = await websocket.recv()
                    request_time = time.time() - request_start
                    
                    result = json.loads(response)
                    if result.get('status') == 'success':
                        times.append(request_time)
                        successful += 1
                    
                except Exception as e:
                    print(f"   Request {i+1} error: {e}")
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{num_requests}")
            
            total_time = time.time() - start_time
            
            if times:
                times = np.array(times)
                mean_time = np.mean(times)
                std_time = np.std(times)
                fps = 1.0 / mean_time
                
                print(f"\n📊 WebSocket FPS Profile Results:")
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
        print(f"❌ WebSocket profiling failed: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test WebSocket async inference")
    parser.add_argument("--url", required=True, help="WebSocket URL")
    parser.add_argument("--profile", action="store_true", help="Profile FPS")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests")
    
    args = parser.parse_args()
    
    print("🔌 WebSocket Async Inference Test")
    print("=" * 40)
    print(f"WebSocket URL: {args.url}")
    print()
    
    # Test single inference
    if not await test_websocket_single(args.url):
        print("❌ Single WebSocket test failed")
        return 1
    
    # Profile FPS if requested
    if args.profile:
        print("\n" + "=" * 40)
        if not await profile_websocket_fps(args.url, args.requests):
            print("❌ WebSocket FPS profiling failed")
            return 1
    
    print("\n✅ All WebSocket tests passed!")
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main())) 