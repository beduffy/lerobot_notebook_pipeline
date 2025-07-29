#!/usr/bin/env python3
"""
Lightning WebSocket Server

A WebSocket server that works with Lightning Cloud's HTTPS proxy.
This server handles the WebSocket upgrade and provides async inference.

Usage:
    python lightning_websocket_server.py --model act --port 8765
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional
import numpy as np
import torch
import argparse

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
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    ACT_AVAILABLE = True
except ImportError:
    ACT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightningWebSocketServer:
    def __init__(self, model_type: str = "act", device: str = "auto"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.clients = set()
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the specified model."""
        print(f"🤖 Loading {self.model_type.upper()} model...")
        
        if self.model_type == "act" and ACT_AVAILABLE:
            # Create ACT model
            config = ACTConfig(
                input_features={
                    "observation.images.front": {"type": "image", "shape": (3, 96, 96)},
                    "observation.state": {"type": "state", "shape": (6,)}
                },
                output_features={
                    "action": {"type": "action", "shape": (6,)}
                },
                chunk_size=100,
                n_action_steps=100,
                dim_model=512,
                n_heads=8,
                dim_feedforward=3200,
                n_encoder_layers=4,
                n_decoder_layers=1,
                vision_backbone="resnet18",
                pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
                use_vae=True,
                latent_dim=32,
                n_vae_encoder_layers=4,
                dropout=0.1,
                kl_weight=10.0,
                optimizer_lr=1e-5,
                optimizer_weight_decay=1e-4,
                optimizer_lr_backbone=1e-5,
            )
            
            # Create dummy dataset stats
            class DummyStats:
                def __init__(self):
                    self.mean = {"observation.images.front": 0.0, "observation.state": 0.0, "action": 0.0}
                    self.std = {"observation.images.front": 1.0, "observation.state": 1.0, "action": 1.0}
            
            self.model = ACTPolicy(config, dataset_stats=DummyStats())
            self.model.to(self.device)
            self.model.eval()
            
            print(f"   ✅ {self.model_type.upper()} model loaded")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Device: {self.device}")
        else:
            raise RuntimeError(f"Model {self.model_type} not available")
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        client_id = id(websocket)
        self.clients.add(websocket)
        logger.info(f"Client {client_id} connected")
        
        try:
            async for message in websocket:
                try:
                    # Parse request
                    request = json.loads(message)
                    
                    # Process inference
                    response = await self._process_inference(request)
                    
                    # Send response
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "status": "error",
                        "error": f"Invalid JSON: {e}",
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    error_response = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def _process_inference(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request."""
        start_time = time.time()
        
        try:
            # Extract observations
            observations = request.get("observations", {})
            
            # Convert to tensors
            batch = {}
            for key, value in observations.items():
                if isinstance(value, list):
                    tensor = torch.tensor(value, dtype=torch.float32)
                    if key == "observation.images.front":
                        # Ensure correct shape and type for images
                        if tensor.dim() == 3:
                            tensor = tensor.unsqueeze(0)  # Add batch dimension
                        tensor = tensor.to(self.device)
                    batch[key] = tensor
                else:
                    batch[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
            
            # Add task description if provided
            task_description = request.get("task_description")
            if task_description:
                batch["task"] = task_description
            
            # Run inference
            with torch.no_grad():
                loss, output_dict = self.model.forward(batch)
                
                # Extract actions
                actions = output_dict.get("action", torch.zeros(1, 6))
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy().tolist()
            
            inference_time = time.time() - start_time
            
            return {
                "status": "success",
                "actions": actions,
                "inference_time": inference_time,
                "model_type": self.model_type,
                "timestamp": time.time()
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            return {
                "status": "error",
                "error": str(e),
                "inference_time": inference_time,
                "model_type": self.model_type,
                "timestamp": time.time()
            }

async def main():
    parser = argparse.ArgumentParser(description="Lightning WebSocket Server")
    parser.add_argument("--model", default="act", help="Model type")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket host")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    if not WEBSOCKET_AVAILABLE:
        print("❌ WebSocket not available")
        return 1
    
    print(f"🔌 Lightning WebSocket Server")
    print(f"   Model: {args.model.upper()}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Device: {args.device}")
    print()
    
    try:
        server = LightningWebSocketServer(args.model, args.device)
        
        print(f"🚀 Starting WebSocket server...")
        print(f"   WebSocket: ws://{args.host}:{args.port}")
        print(f"   Press Ctrl+C to stop")
        
        async with websockets.serve(server.handle_client, args.host, args.port):
            await asyncio.Future()  # Run forever
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main())) 