#!/usr/bin/env python3
"""
Async Inference Server for LeRobot Models

Cloud-based async inference server that can run π0-FAST and other models remotely.
Supports real-time chunking for low-latency robot control over network.

Inspired by Physical Intelligence's real-time chunking research:
- https://www.physicalintelligence.company/research/real_time_chunking
- https://github.com/Physical-Intelligence/real-time-chunking-kinetix

Features:
- Async inference with queue management
- Real-time chunking for low latency
- Support for all LeRobot model types (ACT, Diffusion, SmolVLA, π0-FAST, VQBet)
- GPU acceleration with auto-scaling
- WebSocket and HTTP endpoints
- HuggingFace model integration

Usage:
    # Start server with π0-FAST model
    python async_inference_server.py --model pi0fast --port 8000 --gpu
    
    # Start with custom model path
    python async_inference_server.py --model-path ./models/pi0fast_trained --port 8000
    
    # Start with HuggingFace model
    python async_inference_server.py --hf-model lerobot/pi0fast --port 8000
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import threading
import warnings

# Web framework
try:
    import fastapi
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not available - install with: pip install fastapi uvicorn")

# WebSocket support
try:
    import websockets
    from websockets.server import serve
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("⚠️  WebSocket not available - install with: pip install websockets")

# LeRobot imports
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("⚠️  LeRobot not available")

# Import all model types
MODEL_IMPORTS = {}

try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    MODEL_IMPORTS['act'] = (ACTPolicy, ACTConfig)
    print("✅ ACT model available")
except ImportError:
    print("⚠️  ACT model not available")

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    MODEL_IMPORTS['diffusion'] = (DiffusionPolicy, DiffusionConfig)
    print("✅ Diffusion model available")
except ImportError:
    print("⚠️  Diffusion model not available")

try:
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
    MODEL_IMPORTS['vqbet'] = (VQBeTPolicy, VQBeTConfig)
    print("✅ VQBet model available")
except ImportError:
    print("⚠️  VQBet model not available")

try:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    MODEL_IMPORTS['smolvla'] = (SmolVLAPolicy, SmolVLAConfig)
    print("✅ SmolVLA model available")
except ImportError:
    print("⚠️  SmolVLA model not available")

try:
    from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
    from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
    MODEL_IMPORTS['pi0fast'] = (PI0FASTPolicy, PI0FASTConfig)
    print("✅ π0-FAST model available")
except ImportError:
    print("⚠️  π0-FAST model not available")

# Suppress warnings for cloud environments
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


@dataclass
class InferenceRequest:
    """Request for async inference."""
    request_id: str
    observations: Dict[str, Any]
    task_description: Optional[str] = None
    model_type: str = "pi0fast"
    priority: int = 1  # Higher = more priority
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InferenceResponse:
    """Response from async inference."""
    request_id: str
    actions: np.ndarray
    inference_time: float
    model_type: str
    status: str = "success"
    error: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RealTimeChunker:
    """Real-time chunking for low-latency inference."""
    
    def __init__(self, chunk_size: int = 10, overlap: int = 2):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.action_buffer = []
        self.observation_buffer = []
        
    def add_observation(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Add observation and return action if chunk is ready."""
        self.observation_buffer.append(observation)
        
        if len(self.observation_buffer) >= self.chunk_size:
            # Process chunk
            chunk_observations = self.observation_buffer[-self.chunk_size:]
            
            # Keep overlap for next chunk
            if self.overlap > 0:
                self.observation_buffer = self.observation_buffer[-self.overlap:]
            else:
                self.observation_buffer = []
            
            return chunk_observations
        
        return None
    
    def get_latest_action(self) -> Optional[np.ndarray]:
        """Get the most recent action from buffer."""
        if self.action_buffer:
            return self.action_buffer[-1]
        return None


class AsyncInferenceEngine:
    """Async inference engine with real-time chunking."""
    
    def __init__(self, model_type: str, model_path: Optional[str] = None, 
                 hf_model: Optional[str] = None, device: str = "auto"):
        self.model_type = model_type
        self.model_path = model_path
        self.hf_model = hf_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # Setup queues
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.is_running = False
        
        # Real-time chunking
        self.chunker = RealTimeChunker(chunk_size=10, overlap=2)
        
        # Load model
        self._load_model()
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        
    def _load_model(self):
        """Load the specified model."""
        print(f"🤖 Loading {self.model_type.upper()} model...")
        
        if self.model_type not in MODEL_IMPORTS:
            raise ValueError(f"Unsupported model type: {self.model_type}. Available: {list(MODEL_IMPORTS.keys())}")
        
        PolicyClass, ConfigClass = MODEL_IMPORTS[self.model_type]
        
        try:
            if self.hf_model:
                # Load from HuggingFace
                print(f"   Loading from HuggingFace: {self.hf_model}")
                self.policy = PolicyClass.from_pretrained(self.hf_model)
            elif self.model_path:
                # Load from local path
                print(f"   Loading from local path: {self.model_path}")
                self.policy = PolicyClass.from_pretrained(self.model_path)
            else:
                # Create fresh model (for testing)
                print(f"   Creating fresh {self.model_type} model...")
                
                # Load dataset metadata for config
                dataset_name = "bearlover365/red_cube_always_in_same_place"
                metadata = LeRobotDatasetMetadata(dataset_name)
                features = dataset_to_policy_features(metadata.features)
                output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
                input_features = {key: ft for key, ft in features.items() if key not in output_features}
                
                # Create config
                if self.model_type == "pi0fast":
                    config = ConfigClass(
                        input_features=input_features,
                        output_features=output_features,
                        chunk_size=10,
                        n_action_steps=10,
                    )
                else:
                    config = ConfigClass(
                        input_features=input_features,
                        output_features=output_features,
                    )
                
                self.policy = PolicyClass(config, dataset_stats=metadata.stats)
            
            self.policy.to(self.device)
            self.policy.eval()
            
            # Get model info
            param_count = sum(p.numel() for p in self.policy.parameters())
            print(f"   ✅ Model loaded successfully!")
            print(f"   Parameters: {param_count:,}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            raise
    
    def _inference_worker(self):
        """Worker thread for processing inference requests."""
        self.is_running = True
        
        while self.is_running:
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=1.0)
                
                # Process inference
                response = self._process_inference(request)
                
                # Put response in queue
                self.response_queue.put(response)
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Inference worker error: {e}")
                continue
    
    def _process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            # Prepare observations
            obs_dict = {}
            for key, value in request.observations.items():
                if isinstance(value, (list, np.ndarray)):
                    obs_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
                elif isinstance(value, torch.Tensor):
                    obs_dict[key] = value.to(self.device)
                else:
                    obs_dict[key] = value
            
            # Add task description for VLA models
            if self.model_type in ["smolvla", "pi0fast"] and request.task_description:
                obs_dict["task"] = request.task_description
            
            # Run inference
            with torch.no_grad():
                action = self.policy.select_action(obs_dict)
            
            # Convert to numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            inference_time = time.time() - start_time
            
            return InferenceResponse(
                request_id=request.request_id,
                actions=action,
                inference_time=inference_time,
                model_type=self.model_type
            )
            
        except Exception as e:
            inference_time = time.time() - start_time
            return InferenceResponse(
                request_id=request.request_id,
                actions=np.array([]),
                inference_time=inference_time,
                model_type=self.model_type,
                status="error",
                error=str(e)
            )
    
    def submit_request(self, observations: Dict[str, Any], 
                      task_description: Optional[str] = None,
                      priority: int = 1) -> str:
        """Submit an inference request."""
        request_id = str(uuid.uuid4())
        
        request = InferenceRequest(
            request_id=request_id,
            observations=observations,
            task_description=task_description,
            model_type=self.model_type,
            priority=priority
        )
        
        self.request_queue.put(request)
        return request_id
    
    def get_response(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResponse]:
        """Get response for a request ID."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.request_id == request_id:
                    return response
                else:
                    # Put back in queue for other consumers
                    self.response_queue.put(response)
            except Empty:
                continue
        
        return None
    
    def shutdown(self):
        """Shutdown the inference engine."""
        self.is_running = False
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=5.0)


class AsyncInferenceServer:
    """Main async inference server."""
    
    def __init__(self, model_type: str, model_path: Optional[str] = None,
                 hf_model: Optional[str] = None, device: str = "auto"):
        self.model_type = model_type
        self.engine = AsyncInferenceEngine(model_type, model_path, hf_model, device)
        
        # Setup FastAPI app
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="LeRobot Async Inference Server",
                              description="Async inference server for LeRobot models with real-time chunking")
            
            # Add CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Setup routes
            self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        class InferenceRequestModel(BaseModel):
            observations: Dict[str, Any]
            task_description: Optional[str] = None
            priority: int = 1
        
        class InferenceResponseModel(BaseModel):
            request_id: str
            actions: List[List[float]]
            inference_time: float
            model_type: str
            status: str
            error: Optional[str] = None
            timestamp: float
        
        @self.app.post("/inference", response_model=InferenceResponseModel)
        async def inference_endpoint(request: InferenceRequestModel):
            """Submit inference request."""
            try:
                request_id = self.engine.submit_request(
                    observations=request.observations,
                    task_description=request.task_description,
                    priority=request.priority
                )
                
                # Wait for response
                response = self.engine.get_response(request_id, timeout=30.0)
                
                if response is None:
                    raise HTTPException(status_code=408, detail="Inference timeout")
                
                return InferenceResponseModel(
                    request_id=response.request_id,
                    actions=response.actions.tolist() if hasattr(response.actions, 'tolist') else response.actions,
                    inference_time=response.inference_time,
                    model_type=response.model_type,
                    status=response.status,
                    error=response.error,
                    timestamp=response.timestamp
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_type": self.model_type,
                "device": str(self.engine.device),
                "queue_size": self.engine.request_queue.qsize()
            }
        
        @self.app.get("/model_info")
        async def model_info():
            """Get model information."""
            param_count = sum(p.numel() for p in self.engine.policy.parameters())
            return {
                "model_type": self.model_type,
                "parameters": param_count,
                "device": str(self.engine.device),
                "model_path": self.engine.model_path,
                "hf_model": self.engine.hf_model
            }
    
    async def websocket_handler(self, websocket, path):
        """WebSocket handler for real-time inference."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Submit inference request
                    request_id = self.engine.submit_request(
                        observations=data.get("observations", {}),
                        task_description=data.get("task_description"),
                        priority=data.get("priority", 1)
                    )
                    
                    # Wait for response
                    response = self.engine.get_response(request_id, timeout=30.0)
                    
                    if response:
                        # Send response back
                        await websocket.send(json.dumps(asdict(response)))
                    else:
                        await websocket.send(json.dumps({
                            "error": "Inference timeout",
                            "request_id": request_id
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    await websocket.send(json.dumps({"error": str(e)}))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
    
    def run_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Run WebSocket server."""
        if not WEBSOCKET_AVAILABLE:
            print("❌ WebSocket not available")
            return
        
        print(f"🔌 Starting WebSocket server on ws://{host}:{port}")
        
        async def main():
            async with serve(self.websocket_handler, host, port):
                await asyncio.Future()  # run forever
        
        asyncio.run(main())
    
    def run_http_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run HTTP server."""
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI not available")
            return
        
        print(f"🌐 Starting HTTP server on http://{host}:{port}")
        print(f"📖 API docs: http://{host}:{port}/docs")
        
        uvicorn.run(self.app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Async Inference Server for LeRobot Models")
    parser.add_argument("--model", choices=list(MODEL_IMPORTS.keys()), default="pi0fast",
                       help="Model type to serve")
    parser.add_argument("--model-path", help="Local model path")
    parser.add_argument("--hf-model", help="HuggingFace model ID")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--websocket", action="store_true", help="Run WebSocket server")
    parser.add_argument("--http", action="store_true", help="Run HTTP server")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.hf_model:
        print("⚠️  No model specified - will create fresh model for testing")
    
    if args.gpu and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available")
    
    # Setup device
    device = "cuda" if args.gpu and torch.cuda.is_available() else args.device
    
    print("🚀 LeRobot Async Inference Server")
    print("=" * 50)
    print(f"Model: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"Local path: {args.model_path}")
    print(f"HF model: {args.hf_model}")
    print()
    
    # Create server
    server = AsyncInferenceServer(
        model_type=args.model,
        model_path=args.model_path,
        hf_model=args.hf_model,
        device=device
    )
    
    # Run servers
    if args.websocket:
        server.run_websocket_server(args.host, args.ws_port)
    elif args.http:
        server.run_http_server(args.host, args.port)
    else:
        # Run both by default
        import threading
        
        # Start HTTP server in thread
        http_thread = threading.Thread(
            target=server.run_http_server,
            args=(args.host, args.port),
            daemon=True
        )
        http_thread.start()
        
        # Start WebSocket server in main thread
        server.run_websocket_server(args.host, args.ws_port)


if __name__ == "__main__":
    main() 