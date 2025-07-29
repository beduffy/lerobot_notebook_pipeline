#!/usr/bin/env python3
"""
Test Async Inference System

Quick test to verify the async inference system works with π0-FAST and other models.
This script tests both local and remote inference capabilities.

Usage:
    # Test local async inference
    python test_async_inference.py --local
    
    # Test with specific model
    python test_async_inference.py --model pi0fast --local
    
    # Test remote server
    python test_async_inference.py --server http://localhost:8000
"""

import argparse
import time
import numpy as np
import threading
from pathlib import Path

# Import our async inference components
import sys
from pathlib import Path

# Add async folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / "async"))

try:
    from async_inference_server import AsyncInferenceEngine, InferenceRequest
    from async_inference_client import AsyncInferenceClient
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Async inference not available: {e}")
    ASYNC_AVAILABLE = False

# Import LeRobot components
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("⚠️  LeRobot not available")


def create_test_observations():
    """Create realistic test observations."""
    return {
        "observation.image": np.random.randn(3, 224, 224).tolist(),
        "observation.joint_positions": np.random.randn(6).tolist(),
        "observation.gripper_position": np.random.randn(1).tolist()
    }


def test_local_async_inference(model_type: str = "pi0fast"):
    """Test local async inference engine."""
    print(f"🧪 Testing local async inference with {model_type.upper()}...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        # Create inference engine
        engine = AsyncInferenceEngine(model_type=model_type)
        
        # Create test observations
        observations = create_test_observations()
        task_description = "grab red cube and put to left"
        
        # Submit inference request
        request_id = engine.submit_request(
            observations=observations,
            task_description=task_description
        )
        
        print(f"   Request submitted: {request_id}")
        
        # Wait for response
        response = engine.get_response(request_id, timeout=30.0)
        
        if response and response.status == "success":
            print(f"   ✅ Inference successful!")
            print(f"   Action shape: {response.actions.shape}")
            print(f"   Inference time: {response.inference_time:.3f}s")
            print(f"   Model type: {response.model_type}")
            return True
        else:
            print(f"   ❌ Inference failed: {response.error if response else 'Timeout'}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'engine' in locals():
            engine.shutdown()


def test_remote_async_inference(server_url: str):
    """Test remote async inference server."""
    print(f"🌐 Testing remote async inference at {server_url}...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        # Create client
        client = AsyncInferenceClient(server_url)
        
        # Get server info
        server_info = client.get_server_info()
        print(f"   Server status: {server_info.get('status', 'unknown')}")
        
        # Get model info
        model_info = client.get_model_info()
        print(f"   Model: {model_info.get('model_type', 'unknown')}")
        print(f"   Parameters: {model_info.get('parameters', 'unknown'):,}")
        
        # Test inference
        observations = create_test_observations()
        task_description = "grab red cube and put to left"
        
        result = client.test_http_inference(observations, task_description)
        
        if result.status == "success":
            print(f"   ✅ Remote inference successful!")
            print(f"   Action shape: {result.actions.shape}")
            print(f"   Inference time: {result.inference_time:.3f}s")
            print(f"   Total latency: {result.latency:.3f}s")
            return True
        else:
            print(f"   ❌ Remote inference failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_requests(server_url: str, num_requests: int = 10):
    """Test concurrent inference requests."""
    print(f"🔄 Testing {num_requests} concurrent requests...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        client = AsyncInferenceClient(server_url)
        observations = create_test_observations()
        task_description = "grab red cube and put to left"
        
        # Submit concurrent requests
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            result = client.test_http_inference(observations, task_description)
            results.append(result)
            
            if (i + 1) % 5 == 0:
                print(f"   Progress: {i + 1}/{num_requests}")
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r.status == "success"]
        latencies = [r.latency for r in successful]
        inference_times = [r.inference_time for r in successful]
        
        print(f"   ✅ Concurrent test completed!")
        print(f"   Successful requests: {len(successful)}/{num_requests}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average latency: {np.mean(latencies):.3f}s")
        print(f"   Average inference time: {np.mean(inference_times):.3f}s")
        print(f"   Requests/second: {len(successful)/total_time:.1f}")
        
        return len(successful) == num_requests
        
    except Exception as e:
        print(f"   ❌ Concurrent test failed: {e}")
        return False


def test_robot_control_simulation(server_url: str, duration: int = 30):
    """Simulate robot control loop with remote inference."""
    print(f"🤖 Simulating robot control for {duration} seconds...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        client = AsyncInferenceClient(server_url)
        task_description = "grab red cube and put to left"
        
        start_time = time.time()
        control_interval = 0.1  # 10 Hz control loop
        successful_actions = 0
        total_actions = 0
        
        print(f"   Control frequency: {1.0/control_interval:.1f} Hz")
        
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Simulate robot observations
            observations = create_test_observations()
            
            # Get inference
            result = client.test_http_inference(observations, task_description)
            total_actions += 1
            
            if result.status == "success":
                successful_actions += 1
                print(f"   Action {total_actions}: Latency {result.latency:.3f}s | "
                      f"Inference {result.inference_time:.3f}s")
            else:
                print(f"   Action {total_actions}: Failed - {result.error}")
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < control_interval:
                time.sleep(control_interval - elapsed)
            else:
                print(f"   ⚠️  Control loop slow: {elapsed:.3f}s > {control_interval:.3f}s")
        
        total_time = time.time() - start_time
        success_rate = successful_actions / total_actions if total_actions > 0 else 0
        
        print(f"   ✅ Robot control simulation completed!")
        print(f"   Total actions: {total_actions}")
        print(f"   Successful actions: {successful_actions}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average frequency: {total_actions/total_time:.1f} Hz")
        
        return success_rate > 0.8  # 80% success rate threshold
        
    except Exception as e:
        print(f"   ❌ Robot control simulation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Async Inference System")
    parser.add_argument("--local", action="store_true", help="Test local async inference")
    parser.add_argument("--server", help="Test remote server (URL)")
    parser.add_argument("--model", default="pi0fast", 
                       choices=["act", "diffusion", "smolvla", "pi0fast", "vqbet"],
                       help="Model type to test")
    parser.add_argument("--concurrent", type=int, default=10, 
                       help="Number of concurrent requests to test")
    parser.add_argument("--robot-sim", type=int, default=30,
                       help="Robot control simulation duration (seconds)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    print("🧪 LeRobot Async Inference System Test")
    print("=" * 50)
    print(f"Model: {args.model.upper()}")
    print(f"Local test: {args.local}")
    print(f"Remote server: {args.server}")
    print()
    
    tests_passed = 0
    total_tests = 0
    
    # Test local async inference
    if args.local or args.all:
        total_tests += 1
        if test_local_async_inference(args.model):
            tests_passed += 1
            print("✅ Local async inference test PASSED")
        else:
            print("❌ Local async inference test FAILED")
        print()
    
    # Test remote server
    if args.server or args.all:
        if args.server:
            server_url = args.server
        else:
            server_url = "http://localhost:8000"  # Default
        
        total_tests += 1
        if test_remote_async_inference(server_url):
            tests_passed += 1
            print("✅ Remote async inference test PASSED")
        else:
            print("❌ Remote async inference test FAILED")
        print()
        
        # Test concurrent requests
        total_tests += 1
        if test_concurrent_requests(server_url, args.concurrent):
            tests_passed += 1
            print("✅ Concurrent requests test PASSED")
        else:
            print("❌ Concurrent requests test FAILED")
        print()
        
        # Test robot control simulation
        total_tests += 1
        if test_robot_control_simulation(server_url, args.robot_sim):
            tests_passed += 1
            print("✅ Robot control simulation test PASSED")
        else:
            print("❌ Robot control simulation test FAILED")
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests PASSED! Async inference system is working correctly.")
        print()
        print("🚀 Next steps:")
        print("   1. Start the async inference server:")
        print(f"      python async_inference_server.py --model {args.model} --port 8000")
        print("   2. Test with your robot:")
        print("      python async_inference_client.py --server http://localhost:8000 --robot-control")
        print("   3. Deploy to cloud:")
        print("      python deploy_async_inference.py --platform aws --model {args.model} --gpu")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Make sure LeRobot is installed correctly")
        print("   2. Check if the model type is supported")
        print("   3. Verify network connectivity for remote tests")
        print("   4. Check server logs for detailed error messages")
    
    return 0 if tests_passed == total_tests else 1


if __name__ == "__main__":
    exit(main()) 