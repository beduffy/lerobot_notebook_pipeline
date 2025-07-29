#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Async Inference Test with ACT

Simple test to verify async inference works with ACT model.
This is a minimal test that can run quickly to verify the system.

Usage:
    python quick_async_test.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add async folder to path
sys.path.insert(0, str(Path(__file__).parent / "async"))

try:
    from async_inference_server import AsyncInferenceEngine
    from async_inference_client import AsyncInferenceClient
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"❌ Async inference not available: {e}")
    print("   Make sure you're in the correct directory")
    ASYNC_AVAILABLE = False


def create_test_observations():
    """Create simple test observations for ACT."""
    return {
        "observation.image": np.random.randn(3, 224, 224).tolist(),
        "observation.joint_positions": np.random.randn(6).tolist(),
        "observation.gripper_position": np.random.randn(1).tolist()
    }


def test_act_local():
    """Test ACT model with local async inference."""
    print("🧪 Testing ACT with local async inference...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        # Create inference engine with ACT
        print("   Creating ACT inference engine...")
        engine = AsyncInferenceEngine(model_type="act")
        
        # Create test observations
        observations = create_test_observations()
        
        # Submit inference request
        print("   Submitting inference request...")
        request_id = engine.submit_request(
            observations=observations,
            task_description=None  # ACT doesn't need task description
        )
        
        print(f"   Request ID: {request_id}")
        
        # Wait for response
        print("   Waiting for response...")
        response = engine.get_response(request_id, timeout=30.0)
        
        if response and response.status == "success":
            print(f"   ✅ ACT inference successful!")
            print(f"   Action shape: {response.actions.shape}")
            print(f"   Inference time: {response.inference_time:.3f}s")
            print(f"   Model type: {response.model_type}")
            return True
        else:
            print(f"   ❌ ACT inference failed: {response.error if response else 'Timeout'}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'engine' in locals():
            engine.shutdown()


def test_act_server():
    """Test ACT model with server (if running)."""
    print("\n🌐 Testing ACT with server (if running)...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        # Try to connect to server
        client = AsyncInferenceClient("http://localhost:8000")
        
        # Get server info
        server_info = client.get_server_info()
        if "error" in server_info:
            print("   ⚠️  No server running on localhost:8000")
            print("   Start server with: python async/async_inference_server.py --model act --port 8000")
            return False
        
        print(f"   Server status: {server_info.get('status', 'unknown')}")
        
        # Get model info
        model_info = client.get_model_info()
        print(f"   Model: {model_info.get('model_type', 'unknown')}")
        print(f"   Parameters: {model_info.get('parameters', 'unknown'):,}")
        
        # Test inference
        observations = create_test_observations()
        result = client.test_http_inference(observations)
        
        if result.status == "success":
            print(f"   ✅ Server inference successful!")
            print(f"   Action shape: {result.actions.shape}")
            print(f"   Inference time: {result.inference_time:.3f}s")
            print(f"   Total latency: {result.latency:.3f}s")
            return True
        else:
            print(f"   ❌ Server inference failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Server test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Quick ACT Async Inference Test")
    print("=" * 40)
    print("Testing async inference with ACT model...")
    print()
    
    # Test local async inference
    local_success = test_act_local()
    
    # Test server (if available)
    server_success = test_act_server()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"   Local ACT inference: {'✅ PASSED' if local_success else '❌ FAILED'}")
    print(f"   Server ACT inference: {'✅ PASSED' if server_success else '❌ FAILED'}")
    
    if local_success:
        print("\n🎉 ACT async inference is working!")
        print("\n🚀 Next steps:")
        print("   1. Start server with ACT:")
        print("      python async/async_inference_server.py --model act --port 8000")
        print()
        print("   2. Test with robot:")
        print("      python async/async_inference_client.py --server http://localhost:8000 --robot-control")
        print()
        print("   3. Test other models:")
        print("      python async/async_inference_server.py --model pi0fast --port 8001")
        print("      python async/async_inference_server.py --model smolvla --port 8002")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure LeRobot is installed: pip install lerobot")
        print("   2. Check if ACT model is available")
        print("   3. Verify all dependencies are installed")
    
    return 0 if local_success else 1


if __name__ == "__main__":
    exit(main()) 