#!/usr/bin/env python3
"""
Cloud Deployment Script for Async Inference Server

Deploys the async inference server to various cloud platforms with automatic scaling.
Supports AWS SageMaker, Google Cloud Run, Azure Container Instances, and more.

Features:
- Multi-cloud deployment
- Auto-scaling configuration
- GPU acceleration
- Monitoring and logging
- Cost optimization

Usage:
    # Deploy to AWS SageMaker
    python deploy_async_inference.py --platform aws --model pi0fast --gpu
    
    # Deploy to Google Cloud Run
    python deploy_async_inference.py --platform gcp --model pi0fast
    
    # Deploy to Azure Container Instances
    python deploy_async_inference.py --platform azure --model pi0fast --gpu
    
    # Deploy with custom model
    python deploy_async_inference.py --platform aws --model-path ./models/pi0fast_trained
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Cloud platform imports
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import run_v2
    from google.cloud import aiplatform
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class CloudDeployer:
    """Cloud deployment manager for async inference server."""
    
    def __init__(self, platform: str, model_type: str, model_path: Optional[str] = None,
                 hf_model: Optional[str] = None, gpu: bool = False):
        self.platform = platform
        self.model_type = model_type
        self.model_path = model_path
        self.hf_model = hf_model
        self.gpu = gpu
        
        # Deployment configuration
        self.deployment_name = f"lerobot-async-{model_type}-{int(time.time())}"
        self.port = 8000
        self.websocket_port = 8765
        
        # Platform-specific settings
        self.platform_configs = {
            "aws": {
                "instance_type": "ml.g4dn.xlarge" if gpu else "ml.m5.large",
                "region": "us-east-1",
                "framework": "pytorch",
                "python_version": "py38"
            },
            "gcp": {
                "machine_type": "n1-standard-4",
                "region": "us-central1",
                "framework": "pytorch",
                "python_version": "3.8"
            },
            "azure": {
                "vm_size": "Standard_NC6s_v3" if gpu else "Standard_D2s_v3",
                "region": "eastus",
                "framework": "pytorch",
                "python_version": "3.8"
            }
        }
    
    def create_dockerfile(self) -> str:
        """Create Dockerfile for the async inference server."""
        dockerfile_content = f"""# LeRobot Async Inference Server
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-cloud.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Install additional dependencies for async inference
RUN pip install --no-cache-dir \\
    fastapi \\
    uvicorn \\
    websockets \\
    requests \\
    numpy \\
    torch \\
    torchvision

# Copy application files
COPY async_inference_server.py .
COPY async_inference_client.py .

# Create model directory
RUN mkdir -p /app/models

# Copy model if provided
{f"COPY {self.model_path} /app/models/" if self.model_path else "# No local model"}

# Expose ports
EXPOSE {self.port}
EXPOSE {self.websocket_port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{self.port}/health || exit 1

# Start server
CMD ["python", "async_inference_server.py", "--model", "{self.model_type}", "--port", "{self.port}", "--host", "0.0.0.0"]
"""
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        return "Dockerfile"
    
    def create_docker_compose(self) -> str:
        """Create docker-compose.yml for local testing."""
        compose_content = f"""version: '3.8'

services:
  async-inference-server:
    build: .
    ports:
      - "{self.port}:{self.port}"
      - "{self.websocket_port}:{self.websocket_port}"
    environment:
      - MODEL_TYPE={self.model_type}
      - DEVICE={'cuda' if self.gpu else 'cpu'}
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
"""
        
        with open("docker-compose.yml", "w") as f:
            f.write(compose_content)
        
        return "docker-compose.yml"
    
    def create_kubernetes_manifest(self) -> str:
        """Create Kubernetes manifest for deployment."""
        manifest_content = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.deployment_name}
  labels:
    app: lerobot-async-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: lerobot-async-inference
  template:
    metadata:
      labels:
        app: lerobot-async-inference
    spec:
      containers:
      - name: async-inference-server
        image: {self.deployment_name}:latest
        ports:
        - containerPort: {self.port}
          name: http
        - containerPort: {self.websocket_port}
          name: websocket
        env:
        - name: MODEL_TYPE
          value: "{self.model_type}"
        - name: DEVICE
          value: "{'cuda' if self.gpu else 'cpu'}"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: {self.port}
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: {self.port}
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: {self.deployment_name}-service
spec:
  selector:
    app: lerobot-async-inference
  ports:
  - name: http
    port: {self.port}
    targetPort: {self.port}
  - name: websocket
    port: {self.websocket_port}
    targetPort: {self.websocket_port}
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {self.deployment_name}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {self.deployment_name}
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        
        with open(f"{self.deployment_name}-k8s.yaml", "w") as f:
            f.write(manifest_content)
        
        return f"{self.deployment_name}-k8s.yaml"
    
    def deploy_to_aws(self) -> Dict[str, Any]:
        """Deploy to AWS SageMaker or ECS."""
        if not AWS_AVAILABLE:
            raise RuntimeError("AWS SDK not available. Install with: pip install boto3")
        
        config = self.platform_configs["aws"]
        
        print(f"🚀 Deploying to AWS...")
        print(f"   Model: {self.model_type}")
        print(f"   Instance: {config['instance_type']}")
        print(f"   Region: {config['region']}")
        
        # Create SageMaker endpoint configuration
        sagemaker_config = {
            "ModelName": self.deployment_name,
            "PrimaryContainer": {
                "Image": f"{self.deployment_name}:latest",
                "Environment": {
                    "MODEL_TYPE": self.model_type,
                    "DEVICE": "cuda" if self.gpu else "cpu",
                    "PORT": str(self.port)
                }
            },
            "ExecutionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
        }
        
        # Create endpoint configuration
        endpoint_config = {
            "EndpointConfigName": f"{self.deployment_name}-config",
            "ProductionVariants": [
                {
                    "VariantName": "default",
                    "ModelName": self.deployment_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": config["instance_type"],
                    "InitialVariantWeight": 1.0
                }
            ]
        }
        
        # Create endpoint
        endpoint = {
            "EndpointName": self.deployment_name,
            "EndpointConfigName": f"{self.deployment_name}-config"
        }
        
        return {
            "platform": "aws",
            "deployment_name": self.deployment_name,
            "endpoint_url": f"https://{self.deployment_name}.sagemaker.{config['region']}.amazonaws.com",
            "config": sagemaker_config,
            "endpoint_config": endpoint_config,
            "endpoint": endpoint
        }
    
    def deploy_to_gcp(self) -> Dict[str, Any]:
        """Deploy to Google Cloud Run or Vertex AI."""
        if not GCP_AVAILABLE:
            raise RuntimeError("Google Cloud SDK not available. Install with: pip install google-cloud-run google-cloud-aiplatform")
        
        config = self.platform_configs["gcp"]
        
        print(f"🚀 Deploying to Google Cloud...")
        print(f"   Model: {self.model_type}")
        print(f"   Machine: {config['machine_type']}")
        print(f"   Region: {config['region']}")
        
        # Create Cloud Run service configuration
        service_config = {
            "name": f"projects/YOUR_PROJECT/locations/{config['region']}/services/{self.deployment_name}",
            "template": {
                "metadata": {
                    "annotations": {
                        "autoscaling.knative.dev/minScale": "1",
                        "autoscaling.knative.dev/maxScale": "10"
                    }
                },
                "spec": {
                    "containerConcurrency": 100,
                    "timeoutSeconds": 300,
                    "containers": [
                        {
                            "image": f"gcr.io/YOUR_PROJECT/{self.deployment_name}:latest",
                            "ports": [
                                {"containerPort": self.port},
                                {"containerPort": self.websocket_port}
                            ],
                            "env": [
                                {"name": "MODEL_TYPE", "value": self.model_type},
                                {"name": "DEVICE", "value": "cuda" if self.gpu else "cpu"}
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "4",
                                    "memory": "8Gi"
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        return {
            "platform": "gcp",
            "deployment_name": self.deployment_name,
            "service_url": f"https://{self.deployment_name}-YOUR_PROJECT-{config['region']}.run.app",
            "config": service_config
        }
    
    def deploy_to_azure(self) -> Dict[str, Any]:
        """Deploy to Azure Container Instances or AKS."""
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure SDK not available. Install with: pip install azure-mgmt-containerinstance azure-identity")
        
        config = self.platform_configs["azure"]
        
        print(f"🚀 Deploying to Azure...")
        print(f"   Model: {self.model_type}")
        print(f"   VM Size: {config['vm_size']}")
        print(f"   Region: {config['region']}")
        
        # Create Container Instance configuration
        container_config = {
            "location": config["region"],
            "os_type": "Linux",
            "containers": [
                {
                    "name": self.deployment_name,
                    "image": f"{self.deployment_name}:latest",
                    "ports": [
                        {"port": self.port},
                        {"port": self.websocket_port}
                    ],
                    "environment_variables": [
                        {"name": "MODEL_TYPE", "value": self.model_type},
                        {"name": "DEVICE", "value": "cuda" if self.gpu else "cpu"}
                    ],
                    "resources": {
                        "requests": {
                            "memory_in_gb": 4.0,
                            "cpu": 2.0
                        }
                    }
                }
            ],
            "ip_address": {
                "type": "Public",
                "ports": [
                    {"port": self.port, "protocol": "TCP"},
                    {"port": self.websocket_port, "protocol": "TCP"}
                ]
            }
        }
        
        return {
            "platform": "azure",
            "deployment_name": self.deployment_name,
            "container_url": f"http://{self.deployment_name}.{config['region']}.azurecontainer.io",
            "config": container_config
        }
    
    def deploy(self) -> Dict[str, Any]:
        """Deploy to the specified platform."""
        print(f"🚀 Starting deployment to {self.platform.upper()}...")
        print(f"   Model: {self.model_type}")
        print(f"   GPU: {self.gpu}")
        print(f"   Local model: {self.model_path}")
        print(f"   HF model: {self.hf_model}")
        print()
        
        # Create deployment files
        dockerfile = self.create_dockerfile()
        compose_file = self.create_docker_compose()
        k8s_manifest = self.create_kubernetes_manifest()
        
        print(f"📁 Created deployment files:")
        print(f"   Dockerfile: {dockerfile}")
        print(f"   docker-compose.yml: {compose_file}")
        print(f"   Kubernetes manifest: {k8s_manifest}")
        print()
        
        # Deploy to platform
        if self.platform == "aws":
            return self.deploy_to_aws()
        elif self.platform == "gcp":
            return self.deploy_to_gcp()
        elif self.platform == "azure":
            return self.deploy_to_azure()
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")
    
    def create_monitoring_config(self):
        """Create monitoring configuration."""
        # Prometheus configuration
        prometheus_config = f"""global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'lerobot-async-inference'
    static_configs:
      - targets: ['{self.deployment_name}:{self.port}']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""
        
        os.makedirs("monitoring", exist_ok=True)
        with open("monitoring/prometheus.yml", "w") as f:
            f.write(prometheus_config)
        
        # Grafana dashboard
        dashboard_config = {
            "dashboard": {
                "title": "LeRobot Async Inference Metrics",
                "panels": [
                    {
                        "title": "Inference Latency",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            }
                        ]
                    },
                    {
                        "title": "Requests per Second",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(inference_requests_total[5m])",
                                "legendFormat": "RPS"
                            }
                        ]
                    }
                ]
            }
        }
        
        with open("monitoring/grafana-dashboard.json", "w") as f:
            json.dump(dashboard_config, f, indent=2)
        
        print(f"📊 Created monitoring configuration:")
        print(f"   monitoring/prometheus.yml")
        print(f"   monitoring/grafana-dashboard.json")


def main():
    parser = argparse.ArgumentParser(description="Cloud Deployment for Async Inference Server")
    parser.add_argument("--platform", choices=["aws", "gcp", "azure", "local"], required=True,
                       help="Cloud platform to deploy to")
    parser.add_argument("--model", choices=["act", "diffusion", "smolvla", "pi0fast", "vqbet"], 
                       default="pi0fast", help="Model type to deploy")
    parser.add_argument("--model-path", help="Local model path")
    parser.add_argument("--hf-model", help="HuggingFace model ID")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--monitoring", action="store_true", help="Setup monitoring")
    parser.add_argument("--local-test", action="store_true", help="Test locally with docker-compose")
    
    args = parser.parse_args()
    
    print("☁️  LeRobot Async Inference Cloud Deployment")
    print("=" * 50)
    print(f"Platform: {args.platform.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"GPU: {args.gpu}")
    print(f"Local model: {args.model_path}")
    print(f"HF model: {args.hf_model}")
    print()
    
    # Create deployer
    deployer = CloudDeployer(
        platform=args.platform,
        model_type=args.model,
        model_path=args.model_path,
        hf_model=args.hf_model,
        gpu=args.gpu
    )
    
    try:
        if args.local_test:
            print("🐳 Testing locally with docker-compose...")
            deployer.create_dockerfile()
            deployer.create_docker_compose()
            
            print("📋 Run these commands to test locally:")
            print(f"   docker-compose up --build")
            print(f"   python async_inference_client.py --server http://localhost:{deployer.port} --test-http")
            print(f"   python async_inference_client.py --server ws://localhost:{deployer.websocket_port} --test-websocket")
            
        else:
            # Deploy to cloud
            result = deployer.deploy()
            
            print(f"\n✅ Deployment configuration created!")
            print(f"   Platform: {result['platform']}")
            print(f"   Deployment name: {result['deployment_name']}")
            
            if args.monitoring:
                deployer.create_monitoring_config()
                print(f"   Monitoring: Enabled")
            
            # Platform-specific next steps
            if args.platform == "aws":
                print(f"\n📋 Next steps for AWS:")
                print(f"   1. Build and push Docker image:")
                print(f"      docker build -t {result['deployment_name']} .")
                print(f"      aws ecr get-login-password --region {result['config']['region']} | docker login --username AWS --password-stdin")
                print(f"      docker tag {result['deployment_name']}:latest {result['endpoint_url']}:latest")
                print(f"      docker push {result['endpoint_url']}:latest")
                print(f"   2. Create SageMaker model and endpoint")
                print(f"   3. Test with: python async_inference_client.py --server {result['endpoint_url']} --test-http")
            
            elif args.platform == "gcp":
                print(f"\n📋 Next steps for Google Cloud:")
                print(f"   1. Build and push Docker image:")
                print(f"      docker build -t {result['deployment_name']} .")
                print(f"      docker tag {result['deployment_name']}:latest gcr.io/YOUR_PROJECT/{result['deployment_name']}:latest")
                print(f"      docker push gcr.io/YOUR_PROJECT/{result['deployment_name']}:latest")
                print(f"   2. Deploy to Cloud Run")
                print(f"   3. Test with: python async_inference_client.py --server {result['service_url']} --test-http")
            
            elif args.platform == "azure":
                print(f"\n📋 Next steps for Azure:")
                print(f"   1. Build and push Docker image:")
                print(f"      docker build -t {result['deployment_name']} .")
                print(f"      az acr build --registry YOUR_REGISTRY --image {result['deployment_name']}:latest .")
                print(f"   2. Deploy to Container Instances")
                print(f"   3. Test with: python async_inference_client.py --server {result['container_url']} --test-http")
        
        print(f"\n🎉 Deployment ready!")
        print(f"   Model: {args.model.upper()}")
        print(f"   GPU: {args.gpu}")
        print(f"   Platform: {args.platform.upper()}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 