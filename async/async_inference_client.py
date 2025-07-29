#!/usr/bin/env python3
"""
Async Inference Client for LeRobot Models

Client for testing async inference server with real-time robot control.
Supports both HTTP and WebSocket connections.

Features:
- Real-time inference over network
- Support for all LeRobot model types
- Robot control integration
- Performance monitoring
- Batch inference testing

Usage:
    # Test HTTP endpoint
    python async_inference_client.py --server http://localhost:8000 --test-http
    
    # Test WebSocket endpoint  
    python async_inference_client.py --server ws://localhost:8765 --test-websocket
    
    # Test with robot control
    python async_inference_client.py --server http://localhost:8000 --robot-control
    
    # Performance benchmark
    python async_inference_client.py --server http://localhost:8000 --benchmark
"""

import argparse
import asyncio
import json
import time
import numpy as np
import requests
import websockets
from typing import Dict, Any, List, Optional
import threading
from dataclasses import dataclass
import statistics

# Optional robot imports
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from lerobot.common.robots import make_robot_from_config, so101_follower
    LEROBOT_ROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_ROBOT_AVAILABLE = False


@dataclass
class InferenceResult:
    """Result from inference request."""
    request_id: str
    actions: np.ndarray
    inference_time: float
    latency: float  # Total round-trip time
    status: str
    error: Optional[str] = None


class AsyncInferenceClient:
    """Client for async inference server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = requests.Session()
        
        # Performance tracking
        self.latency_history = []
        self.inference_time_history = []
        
    def test_http_inference(self, observations: Dict[str, Any], 
                           task_description: Optional[str] = None) -> InferenceResult:
        """Test HTTP inference endpoint."""
        start_time = time.time()
        
        try:
            # Prepare request
            payload = {
                "observations": observations,
                "task_description": task_description,
                "priority": 1
            }
            
            # Send request
            response = self.session.post(
                f"{self.server_url}/inference",
                json=payload,
                timeout=30.0
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return InferenceResult(
                    request_id=data["request_id"],
                    actions=np.array(data["actions"]),
                    inference_time=data["inference_time"],
                    latency=latency,
                    status=data["status"],
                    error=data.get("error")
                )
            else:
                return InferenceResult(
                    request_id="",
                    actions=np.array([]),
                    inference_time=0.0,
                    latency=latency,
                    status="error",
                    error=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            latency = time.time() - start_time
            return InferenceResult(
                request_id="",
                actions=np.array([]),
                inference_time=0.0,
                latency=latency,
                status="error",
                error=str(e)
            )
    
    async def test_websocket_inference(self, observations: Dict[str, Any],
                                      task_description: Optional[str] = None) -> InferenceResult:
        """Test WebSocket inference endpoint."""
        start_time = time.time()
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # Prepare request
                payload = {
                    "observations": observations,
                    "task_description": task_description,
                    "priority": 1
                }
                
                # Send request
                await websocket.send(json.dumps(payload))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                data = json.loads(response)
                
                latency = time.time() - start_time
                
                return InferenceResult(
                    request_id=data.get("request_id", ""),
                    actions=np.array(data.get("actions", [])),
                    inference_time=data.get("inference_time", 0.0),
                    latency=latency,
                    status=data.get("status", "success"),
                    error=data.get("error")
                )
                
        except Exception as e:
            latency = time.time() - start_time
            return InferenceResult(
                request_id="",
                actions=np.array([]),
                inference_time=0.0,
                latency=latency,
                status="error",
                error=str(e)
            )
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = self.session.get(f"{self.server_url}/model_info")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark(self, num_requests: int = 100, 
                 observations: Optional[Dict[str, Any]] = None,
                 task_description: Optional[str] = None) -> Dict[str, Any]:
        """Run performance benchmark."""
        if observations is None:
            # Create dummy observations for testing
            observations = {
                "observation.image": np.random.randn(3, 224, 224).tolist(),
                "observation.joint_positions": np.random.randn(6).tolist(),
                "observation.gripper_position": np.random.randn(1).tolist()
            }
        
        print(f"🚀 Running benchmark with {num_requests} requests...")
        
        results = []
        start_time = time.time()
        
        for i in range(num_requests):
            result = self.test_http_inference(observations, task_description)
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{num_requests}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_results = [r for r in results if r.status == "success"]
        
        if successful_results:
            latencies = [r.latency for r in successful_results]
            inference_times = [r.inference_time for r in successful_results]
            
            stats = {
                "total_requests": num_requests,
                "successful_requests": len(successful_results),
                "success_rate": len(successful_results) / num_requests,
                "total_time": total_time,
                "requests_per_second": num_requests / total_time,
                "latency": {
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "std": statistics.stdev(latencies) if len(latencies) > 1 else 0
                },
                "inference_time": {
                    "mean": statistics.mean(inference_times),
                    "median": statistics.median(inference_times),
                    "min": min(inference_times),
                    "max": max(inference_times),
                    "std": statistics.stdev(inference_times) if len(inference_times) > 1 else 0
                }
            }
        else:
            stats = {
                "total_requests": num_requests,
                "successful_requests": 0,
                "success_rate": 0.0,
                "total_time": total_time,
                "requests_per_second": 0.0,
                "error": "No successful requests"
            }
        
        return stats


class RobotControlClient:
    """Robot control client with async inference."""
    
    def __init__(self, server_url: str, robot_config: Dict[str, Any]):
        self.inference_client = AsyncInferenceClient(server_url)
        self.robot_config = robot_config
        self.robot = None
        self.is_running = False
        
    def setup_robot(self):
        """Setup robot connection."""
        if not LEROBOT_ROBOT_AVAILABLE:
            print("❌ LeRobot robot not available")
            return False
        
        try:
            print("🤖 Setting up robot...")
            self.robot = make_robot_from_config(self.robot_config)
            print("✅ Robot setup complete!")
            return True
        except Exception as e:
            print(f"❌ Robot setup failed: {e}")
            return False
    
    def get_observations(self) -> Dict[str, Any]:
        """Get current robot observations."""
        if self.robot is None:
            # Return dummy observations for testing
            return {
                "observation.image": np.random.randn(3, 224, 224).tolist(),
                "observation.joint_positions": np.random.randn(6).tolist(),
                "observation.gripper_position": np.random.randn(1).tolist()
            }
        
        try:
            # Get robot state
            joint_positions = self.robot.get_joint_positions()
            gripper_position = self.robot.get_gripper_position()
            
            # Get camera image
            if OPENCV_AVAILABLE:
                camera = cv2.VideoCapture(4)  # Adjust camera index as needed
                ret, frame = camera.read()
                camera.release()
                
                if ret:
                    # Resize and normalize image
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.transpose(2, 0, 1) / 255.0  # HWC to CHW, normalize
                    image = frame.tolist()
                else:
                    image = np.random.randn(3, 224, 224).tolist()
            else:
                image = np.random.randn(3, 224, 224).tolist()
            
            return {
                "observation.image": image,
                "observation.joint_positions": joint_positions.tolist(),
                "observation.gripper_position": [gripper_position]
            }
            
        except Exception as e:
            print(f"⚠️  Error getting observations: {e}")
            return {
                "observation.image": np.random.randn(3, 224, 224).tolist(),
                "observation.joint_positions": np.random.randn(6).tolist(),
                "observation.gripper_position": np.random.randn(1).tolist()
            }
    
    def execute_action(self, action: np.ndarray):
        """Execute action on robot."""
        if self.robot is None:
            print(f"🤖 Simulated action: {action}")
            return
        
        try:
            # Extract joint positions and gripper command
            joint_positions = action[:6]  # First 6 values are joint positions
            gripper_command = action[6] if len(action) > 6 else 0.0
            
            # Execute action
            self.robot.set_joint_positions(joint_positions)
            self.robot.set_gripper_position(gripper_command)
            
        except Exception as e:
            print(f"❌ Error executing action: {e}")
    
    def run_control_loop(self, task_description: str = "grab red cube and put to left",
                        control_frequency: float = 10.0):
        """Run robot control loop with async inference."""
        if not self.setup_robot():
            print("⚠️  Running in simulation mode")
        
        print(f"🎮 Starting robot control loop...")
        print(f"   Task: {task_description}")
        print(f"   Frequency: {control_frequency} Hz")
        
        self.is_running = True
        control_interval = 1.0 / control_frequency
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Get observations
                observations = self.get_observations()
                
                # Get inference
                result = self.inference_client.test_http_inference(
                    observations, task_description
                )
                
                if result.status == "success":
                    # Execute action
                    self.execute_action(result.actions)
                    
                    # Print performance info
                    print(f"🤖 Action executed | Latency: {result.latency:.3f}s | "
                          f"Inference: {result.inference_time:.3f}s")
                else:
                    print(f"❌ Inference failed: {result.error}")
                
                # Control timing
                elapsed = time.time() - loop_start
                if elapsed < control_interval:
                    time.sleep(control_interval - elapsed)
                else:
                    print(f"⚠️  Control loop running slow: {elapsed:.3f}s > {control_interval:.3f}s")
                    
        except KeyboardInterrupt:
            print("\n🛑 Control loop stopped by user")
        finally:
            self.is_running = False


def main():
    parser = argparse.ArgumentParser(description="Async Inference Client")
    parser.add_argument("--server", required=True, help="Server URL (http:// or ws://)")
    parser.add_argument("--test-http", action="store_true", help="Test HTTP endpoint")
    parser.add_argument("--test-websocket", action="store_true", help="Test WebSocket endpoint")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--robot-control", action="store_true", help="Run robot control loop")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of benchmark requests")
    parser.add_argument("--robot-port", default="/dev/ttyACM0", help="Robot port")
    parser.add_argument("--robot-type", default="so101_follower", help="Robot type")
    parser.add_argument("--task", default="grab red cube and put to left", help="Task description")
    
    args = parser.parse_args()
    
    print("🔌 LeRobot Async Inference Client")
    print("=" * 40)
    print(f"Server: {args.server}")
    print()
    
    # Create client
    client = AsyncInferenceClient(args.server)
    
    # Get server info
    print("📊 Server Information:")
    server_info = client.get_server_info()
    for key, value in server_info.items():
        print(f"   {key}: {value}")
    
    print()
    
    # Get model info
    print("🤖 Model Information:")
    model_info = client.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    print()
    
    # Run tests
    if args.test_http:
        print("🌐 Testing HTTP endpoint...")
        observations = {
            "observation.image": np.random.randn(3, 224, 224).tolist(),
            "observation.joint_positions": np.random.randn(6).tolist(),
            "observation.gripper_position": np.random.randn(1).tolist()
        }
        
        result = client.test_http_inference(observations, args.task)
        print(f"   Status: {result.status}")
        print(f"   Latency: {result.latency:.3f}s")
        print(f"   Inference time: {result.inference_time:.3f}s")
        if result.actions.size > 0:
            print(f"   Action shape: {result.actions.shape}")
        if result.error:
            print(f"   Error: {result.error}")
    
    elif args.test_websocket:
        print("🔌 Testing WebSocket endpoint...")
        observations = {
            "observation.image": np.random.randn(3, 224, 224).tolist(),
            "observation.joint_positions": np.random.randn(6).tolist(),
            "observation.gripper_position": np.random.randn(1).tolist()
        }
        
        async def test_websocket():
            result = await client.test_websocket_inference(observations, args.task)
            print(f"   Status: {result.status}")
            print(f"   Latency: {result.latency:.3f}s")
            print(f"   Inference time: {result.inference_time:.3f}s")
            if result.actions.size > 0:
                print(f"   Action shape: {result.actions.shape}")
            if result.error:
                print(f"   Error: {result.error}")
        
        asyncio.run(test_websocket())
    
    elif args.benchmark:
        print("📊 Running performance benchmark...")
        stats = client.benchmark(args.num_requests, task_description=args.task)
        
        print(f"\n📈 Benchmark Results:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Successful requests: {stats['successful_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Total time: {stats['total_time']:.2f}s")
        print(f"   Requests/second: {stats['requests_per_second']:.1f}")
        
        if 'latency' in stats:
            print(f"\n⏱️  Latency (seconds):")
            print(f"   Mean: {stats['latency']['mean']:.3f}")
            print(f"   Median: {stats['latency']['median']:.3f}")
            print(f"   Min: {stats['latency']['min']:.3f}")
            print(f"   Max: {stats['latency']['max']:.3f}")
            print(f"   Std: {stats['latency']['std']:.3f}")
        
        if 'inference_time' in stats:
            print(f"\n🤖 Inference Time (seconds):")
            print(f"   Mean: {stats['inference_time']['mean']:.3f}")
            print(f"   Median: {stats['inference_time']['median']:.3f}")
            print(f"   Min: {stats['inference_time']['min']:.3f}")
            print(f"   Max: {stats['inference_time']['max']:.3f}")
            print(f"   Std: {stats['inference_time']['std']:.3f}")
    
    elif args.robot_control:
        print("🤖 Starting robot control...")
        robot_config = {
            "type": args.robot_type,
            "port": args.robot_port,
            "id": "async_inference_robot"
        }
        
        robot_client = RobotControlClient(args.server, robot_config)
        robot_client.run_control_loop(args.task)
    
    else:
        print("ℹ️  No test specified. Use --test-http, --test-websocket, --benchmark, or --robot-control")


if __name__ == "__main__":
    main() 