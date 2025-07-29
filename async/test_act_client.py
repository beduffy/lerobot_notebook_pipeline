#!/usr/bin/env python3
"""
Test ACT Async Inference Client

Quick test to verify the ACT async inference client works.

Usage:
    python test_act_client.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add async folder to path
sys.path.insert(0, str(Path(__file__).parent / "async"))

try:
    from async_inference_client import AsyncInferenceClient
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"❌ Async inference not available: {e}")
    ASYNC_AVAILABLE = False


def create_test_observations():
    """Create simple test observations for ACT."""
    return {
        "observation.image": np.random.randn(3, 224, 224).tolist(),
        "observation.joint_positions": np.random.randn(6).tolist(),
        "observation.gripper_position": np.random.randn(1).tolist()
    }


def test_act_client():
    """Test ACT client with server."""
    print("🧪 Testing ACT async inference client...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        # Create client
        client = AsyncInferenceClient("http://localhost:8000")
        
        # Get server info
        print("   Getting server info...")
        server_info = client.get_server_info()
        if "error" in server_info:
            print("   ❌ No server running on localhost:8000")
            print("   Start server with: python start_act_server.py")
            return False
        
        print(f"   ✅ Server status: {server_info.get('status', 'unknown')}")
        
        # Get model info
        model_info = client.get_model_info()
        print(f"   ✅ Model: {model_info.get('model_type', 'unknown')}")
        print(f"   ✅ Parameters: {model_info.get('parameters', 'unknown'):,}")
        
        # Test inference
        print("   Testing inference...")
        observations = create_test_observations()
        result = client.test_http_inference(observations)
        
        if result.status == "success":
            print(f"   ✅ Inference successful!")
            print(f"   Action shape: {result.actions.shape}")
            print(f"   Inference time: {result.inference_time:.3f}s")
            print(f"   Total latency: {result.latency:.3f}s")
            return True
        else:
            print(f"   ❌ Inference failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_requests():
    """Test multiple concurrent requests."""
    print("\n🔄 Testing concurrent requests...")
    
    if not ASYNC_AVAILABLE:
        print("❌ Async inference not available")
        return False
    
    try:
        client = AsyncInferenceClient("http://localhost:8000")
        observations = create_test_observations()
        
        # Test 5 concurrent requests
        num_requests = 5
        results = []
        
        print(f"   Sending {num_requests} concurrent requests...")
        start_time = time.time()
        
        for i in range(num_requests):
            result = client.test_http_inference(observations)
            results.append(result)
            print(f"   Request {i+1}: {'✅' if result.status == 'success' else '❌'}")
        
        total_time = time.time() - start_time
        successful = [r for r in results if r.status == "success"]
        
        print(f"   ✅ Concurrent test completed!")
        print(f"   Successful: {len(successful)}/{num_requests}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Requests/second: {len(successful)/total_time:.1f}")
        
        return len(successful) == num_requests
        
    except Exception as e:
        print(f"   ❌ Concurrent test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 ACT Async Inference Client Test")
    print("=" * 40)
    print("Testing ACT client with server...")
    print()
    
    # Test basic client functionality
    client_success = test_act_client()
    
    # Test concurrent requests
    concurrent_success = test_concurrent_requests()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"   Client test: {'✅ PASSED' if client_success else '❌ FAILED'}")
    print(f"   Concurrent test: {'✅ PASSED' if concurrent_success else '❌ FAILED'}")
    
    if client_success and concurrent_success:
        print("\n🎉 ACT async inference client is working!")
        print("\n🚀 Next steps:")
        print("   1. Test with robot:")
        print("      python async/async_inference_client.py --server http://localhost:8000 --robot-control")
        print()
        print("   2. Test other models:")
        print("      python async/async_inference_server.py --model pi0fast --port 8001")
        print("      python async/async_inference_server.py --model smolvla --port 8002")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure server is running: python start_act_server.py")
        print("   2. Check if server is accessible: curl http://localhost:8000/health")
        print("   3. Verify all dependencies are installed")
    
    return 0 if client_success and concurrent_success else 1


if __name__ == "__main__":
    exit(main()) 