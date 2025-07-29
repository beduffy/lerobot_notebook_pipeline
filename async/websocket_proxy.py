#!/usr/bin/env python3
"""
WebSocket Proxy for Lightning Cloud

This script creates a WebSocket proxy that can work with Lightning's HTTPS proxy.
It handles the WebSocket upgrade and forwards messages to the actual WebSocket server.

Usage:
    python websocket_proxy.py --port 8765
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Any
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketProxy:
    def __init__(self, target_host="localhost", target_port=8765):
        self.target_host = target_host
        self.target_port = target_port
        self.clients = set()
        
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connection."""
        client_id = id(websocket)
        self.clients.add(websocket)
        logger.info(f"Client {client_id} connected")
        
        try:
            # Connect to the actual WebSocket server
            target_uri = f"ws://{self.target_host}:{self.target_port}"
            async with websockets.connect(target_uri) as target_websocket:
                logger.info(f"Connected to target server: {target_uri}")
                
                # Create tasks for bidirectional communication
                client_to_target = asyncio.create_task(
                    self._forward_messages(websocket, target_websocket, "client→target")
                )
                target_to_client = asyncio.create_task(
                    self._forward_messages(target_websocket, websocket, "target→client")
                )
                
                # Wait for either direction to close
                done, pending = await asyncio.wait(
                    [client_to_target, target_to_client],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected")
    
    async def _forward_messages(self, source, destination, direction):
        """Forward messages from source to destination."""
        try:
            async for message in source:
                await destination.send(message)
                logger.debug(f"{direction}: {len(message)} bytes")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"{direction}: Connection closed")
        except Exception as e:
            logger.error(f"{direction}: Error forwarding messages: {e}")

async def main():
    parser = argparse.ArgumentParser(description="WebSocket Proxy for Lightning Cloud")
    parser.add_argument("--port", type=int, default=8765, help="Proxy port")
    parser.add_argument("--target-host", default="localhost", help="Target WebSocket host")
    parser.add_argument("--target-port", type=int, default=8765, help="Target WebSocket port")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy host")
    
    args = parser.parse_args()
    
    proxy = WebSocketProxy(args.target_host, args.target_port)
    
    print(f"🔌 WebSocket Proxy")
    print(f"   Listening: {args.host}:{args.port}")
    print(f"   Target: ws://{args.target_host}:{args.target_port}")
    print(f"   Press Ctrl+C to stop")
    
    async with websockets.serve(proxy.handle_client, args.host, args.port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Proxy stopped") 