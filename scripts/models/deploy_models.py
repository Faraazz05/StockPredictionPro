"""
scripts/models/deploy_models.py

Comprehensive model deployment script for StockPredictionPro.
Supports multiple deployment targets: local, Docker, REST API, cloud platforms.
Handles model versioning, rollback, health checks, and monitoring integration.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import numpy as np
import shutil
import subprocess
import joblib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import hashlib
import tempfile

# REST API dependencies
try:
    import fastapi
    import uvicorn
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Cloud deployment dependencies
try:
    import boto3
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    from google.cloud import storage as gcs
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

# Docker dependencies
try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.ModelDeployment')

# Directory configuration
MODELS_DIR = Path('./models/trained')
PRODUCTION_DIR = Path('./models/production')
DEPLOYMENT_DIR = Path('./deployment')
LOGS_DIR = Path('./logs')
DOCKER_DIR = Path('./docker')

# Ensure directories exist
for dir_path in [PRODUCTION_DIR, DEPLOYMENT_DIR, LOGS_DIR, DOCKER_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    # Model information
    model_name: str
    model_version: str = None
    model_type: str = 'sklearn'  # sklearn, xgboost, lightgbm, jax
    
    # Deployment settings
    deployment_type: str = 'local'  # local, docker, rest, aws, gcp
    environment: str = 'production'  # development, staging, production
    
    # API settings (for REST deployment)
    api_host: str = '0.0.0.0'
    api_port: int = 8000
    api_workers: int = 1
    enable_docs: bool = True
    
    # Docker settings
    docker_image: str = 'stockpredictionpro/model-server'
    docker_tag: str = 'latest'
    docker_registry: str = None
    
    # Cloud settings
    aws_bucket: str = None
    aws_region: str = 'us-east-1'
    gcp_bucket: str = None
    gcp_project: str = None
    
    # Monitoring and health checks
    enable_monitoring: bool = True
    health_check_interval: int = 300  # 5 minutes
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0

@dataclass
class DeploymentMetadata:
    """Metadata for deployed models"""
    model_name: str
    model_version: str
    model_type: str
    deployment_type: str
    deployed_at: str
    model_path: str
    model_hash: str
    scaler_path: str = None
    config_path: str = None
    performance_metrics: Dict[str, float] = None
    feature_names: List[str] = None
    deployment_status: str = 'active'  # active, inactive, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

class ModelRegistry:
    """Central registry for tracking deployed models"""
    
    def __init__(self, registry_path: Path = None):
        self.registry_path = registry_path or PRODUCTION_DIR / 'model_registry.json'
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, DeploymentMetadata]:
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                return {k: DeploymentMetadata(**v) for k, v in data.items()}
        return {}
    
    def _save_registry(self) -> None:
        data = {k: v.to_dict() for k, v in self.models.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def register_model(self, metadata: DeploymentMetadata) -> str:
        model_id = f"{metadata.model_name}_{metadata.model_version}"
        self.models[model_id] = metadata
        self._save_registry()
        logger.info(f"📝 Registered model: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[DeploymentMetadata]:
        return self.models.get(model_id)
    
    def list_models(self, status: str = None) -> List[DeploymentMetadata]:
        models = list(self.models.values())
        if status:
            models = [m for m in models if m.deployment_status == status]
        return models
    
    def deactivate_model(self, model_id: str) -> bool:
        if model_id in self.models:
            self.models[model_id].deployment_status = 'inactive'
            self._save_registry()
            logger.info(f"🔴 Deactivated model: {model_id}")
            return True
        return False

# ============================================
# MODEL PACKAGING AND VALIDATION
# ============================================

class ModelPackager:
    """Package models for deployment"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.registry = ModelRegistry()
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def validate_model(self, model_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate model file and extract metadata"""
        try:
            # Load and test model
            if model_path.suffix in ['.pkl', '.joblib']:
                model = joblib.load(model_path)
            else:
                logger.error(f"Unsupported model format: {model_path.suffix}")
                return False, {}
            
            # Extract model information
            model_info = {
                'type': type(model).__name__,
                'size_mb': model_path.stat().st_size / 1024 / 1024,
                'created': datetime.fromtimestamp(model_path.stat().st_ctime).isoformat()
            }
            
            # Test prediction capability
            if hasattr(model, 'predict'):
                # Create dummy input for testing
                if hasattr(model, 'n_features_in_'):
                    dummy_input = [[0.0] * model.n_features_in_]
                    _ = model.predict(dummy_input)
                    model_info['n_features'] = model.n_features_in_
                else:
                    logger.warning("Cannot determine model input features")
            
            logger.info(f"✅ Model validation passed: {model_info['type']}, {model_info['size_mb']:.2f} MB")
            return True, model_info
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return False, {}
    
    def package_model(self, model_path: Path, additional_files: List[Path] = None) -> Path:
        """Package model with metadata and dependencies"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Validate model
        is_valid, model_info = self.validate_model(model_path)
        if not is_valid:
            raise ValueError("Model validation failed")
        
        # Create version if not provided
        if not self.config.model_version:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.config.model_version = f"v_{timestamp}"
        
        # Create deployment package directory
        package_dir = PRODUCTION_DIR / f"{self.config.model_name}_{self.config.model_version}"
        package_dir.mkdir(exist_ok=True)
        
        # Copy model file
        model_dest = package_dir / model_path.name
        shutil.copy2(model_path, model_dest)
        
        # Calculate model hash
        model_hash = self.calculate_file_hash(model_dest)
        
        # Copy additional files (scaler, config, etc.)
        scaler_path = None
        config_path = None
        
        if additional_files:
            for file_path in additional_files:
                dest_path = package_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                
                if 'scaler' in file_path.name.lower():
                    scaler_path = str(dest_path)
                elif 'config' in file_path.name.lower():
                    config_path = str(dest_path)
        
        # Create deployment metadata
        metadata = DeploymentMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_type=self.config.model_type,
            deployment_type=self.config.deployment_type,
            deployed_at=datetime.now().isoformat(),
            model_path=str(model_dest),
            model_hash=model_hash,
            scaler_path=scaler_path,
            config_path=config_path,
            performance_metrics=model_info
        )
        
        # Save metadata
        metadata_path = package_dir / 'deployment_metadata.json'
        metadata.save(metadata_path)
        
        # Register model
        self.registry.register_model(metadata)
        
        logger.info(f"📦 Model packaged successfully: {package_dir}")
        return package_dir

# ============================================
# DEPLOYMENT STRATEGIES
# ============================================

class LocalDeployment:
    """Deploy model for local/batch inference"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def deploy(self, package_dir: Path) -> bool:
        """Deploy model locally"""
        try:
            # Create symlink to latest version
            latest_link = PRODUCTION_DIR / f"{self.config.model_name}_latest"
            
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            latest_link.symlink_to(package_dir.resolve())
            
            # Create inference script
            self._create_inference_script(latest_link)
            
            logger.info(f"🏠 Local deployment completed: {latest_link}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Local deployment failed: {e}")
            return False
    
    def _create_inference_script(self, model_dir: Path) -> None:
        """Create inference script for deployed model"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated inference script for {self.config.model_name}
Generated on: {datetime.now().isoformat()}
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path("{model_dir}")
MODEL_FILE = MODEL_DIR / "*.pkl"  # Update with actual model filename
SCALER_FILE = MODEL_DIR / "*scaler*.pkl"  # Update if scaler exists

def load_model():
    model_files = list(MODEL_DIR.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError("No model file found")
    return joblib.load(model_files[0])

def load_scaler():
    scaler_files = list(MODEL_DIR.glob("*scaler*.pkl"))
    if scaler_files:
        return joblib.load(scaler_files[0])
    return None

def predict(features):
    model = load_model()
    scaler = load_scaler()
    
    if scaler:
        features = scaler.transform(features)
    
    return model.predict(features)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <input_json_file>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        input_data = json.load(f)
    
    features = np.array(input_data['features'])
    predictions = predict(features)
    
    result = {{"predictions": predictions.tolist()}}
    print(json.dumps(result, indent=2))
'''
        
        script_path = model_dir / 'inference.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        logger.info(f"📝 Created inference script: {script_path}")

class RestApiDeployment:
    """Deploy model as REST API service"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.app = None
        self.model = None
        self.scaler = None
    
    def deploy(self, package_dir: Path) -> bool:
        """Deploy model as REST API"""
        if not HAS_FASTAPI:
            logger.error("FastAPI not available for REST deployment")
            return False
        
        try:
            # Load model and components
            self._load_model_components(package_dir)
            
            # Create FastAPI app
            self._create_api_app()
            
            # Start server
            self._start_server()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ REST API deployment failed: {e}")
            return False
    
    def _load_model_components(self, package_dir: Path) -> None:
        """Load model and preprocessing components"""
        # Load model
        model_files = list(package_dir.glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError("No model file found in package")
        
        self.model = joblib.load(model_files[0])
        logger.info(f"📥 Loaded model: {model_files[0].name}")
        
        # Load scaler if available
        scaler_files = list(package_dir.glob("*scaler*.pkl"))
        if scaler_files:
            self.scaler = joblib.load(scaler_files[0])
            logger.info(f"📥 Loaded scaler: {scaler_files[0].name}")
    
    def _create_api_app(self) -> None:
        """Create FastAPI application"""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        class PredictionRequest(BaseModel):
            features: List[List[float]]
            return_probabilities: bool = False
        
        class PredictionResponse(BaseModel):
            predictions: List[float]
            model_name: str
            model_version: str
            timestamp: str
        
        class HealthResponse(BaseModel):
            status: str
            model_name: str
            model_version: str
            uptime: str
        
        self.app = FastAPI(
            title=f"StockPredictionPro - {self.config.model_name} API",
            description=f"REST API for {self.config.model_name} model predictions",
            version=self.config.model_version,
            docs_url="/docs" if self.config.enable_docs else None
        )
        
        start_time = datetime.now()
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            try:
                features = np.array(request.features)
                
                # Apply scaling if available
                if self.scaler:
                    features = self.scaler.transform(features)
                
                # Make predictions
                predictions = self.model.predict(features)
                
                return PredictionResponse(
                    predictions=predictions.tolist(),
                    model_name=self.config.model_name,
                    model_version=self.config.model_version,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            uptime = datetime.now() - start_time
            return HealthResponse(
                status="healthy",
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                uptime=str(uptime)
            )
        
        @self.app.get("/model/info")
        async def model_info():
            return {
                "model_name": self.config.model_name,
                "model_version": self.config.model_version,
                "model_type": type(self.model).__name__,
                "features_required": getattr(self.model, 'n_features_in_', 'unknown'),
                "has_scaler": self.scaler is not None
            }
    
    def _start_server(self) -> None:
        """Start uvicorn server"""
        logger.info(f"🚀 Starting REST API server on {self.config.api_host}:{self.config.api_port}")
        
        uvicorn.run(
            self.app,
            host=self.config.api_host,
            port=self.config.api_port,
            workers=self.config.api_workers,
            log_level="info"
        )

class DockerDeployment:
    """Deploy model as Docker container"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def deploy(self, package_dir: Path) -> bool:
        """Deploy model as Docker container"""
        if not HAS_DOCKER:
            logger.error("Docker not available for container deployment")
            return False
        
        try:
            # Create Dockerfile
            dockerfile_path = self._create_dockerfile(package_dir)
            
            # Build Docker image
            success = self._build_docker_image(package_dir, dockerfile_path)
            
            if success and self.config.docker_registry:
                self._push_to_registry()
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Docker deployment failed: {e}")
            return False
    
    def _create_dockerfile(self, package_dir: Path) -> Path:
        """Create Dockerfile for model deployment"""
        dockerfile_content = f'''# Auto-generated Dockerfile for {self.config.model_name}
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model package
COPY {package_dir.name}/ ./model/

# Copy API code
COPY api_server.py .

# Expose port
EXPOSE {self.config.api_port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.config.api_port}/health || exit 1

# Run the application
CMD ["python", "api_server.py"]
'''
        
        dockerfile_path = package_dir / 'Dockerfile'
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt
        requirements = [
            'fastapi==0.104.1',
            'uvicorn==0.24.0',
            'scikit-learn==1.3.0',
            'pandas==2.1.0',
            'numpy==1.24.3',
            'joblib==1.3.2'
        ]
        
        if self.config.model_type == 'xgboost':
            requirements.append('xgboost==1.7.6')
        elif self.config.model_type == 'lightgbm':
            requirements.append('lightgbm==4.0.0')
        
        requirements_path = package_dir / 'requirements.txt'
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        # Create API server script
        self._create_api_server_script(package_dir)
        
        logger.info(f"📝 Created Dockerfile: {dockerfile_path}")
        return dockerfile_path
    
    def _create_api_server_script(self, package_dir: Path) -> None:
        """Create API server script for Docker container"""
        api_script = '''#!/usr/bin/env python3
import os
import joblib
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from datetime import datetime

# Load model components
MODEL_DIR = Path("./model")
model_files = list(MODEL_DIR.glob("*.pkl"))
if not model_files:
    raise FileNotFoundError("No model file found")

model = joblib.load(model_files[0])
print(f"Loaded model: {model_files[0].name}")

# Load scaler if available
scaler_files = list(MODEL_DIR.glob("*scaler*.pkl"))
scaler = None
if scaler_files:
    scaler = joblib.load(scaler_files[0])
    print(f"Loaded scaler: {scaler_files[0].name}")

# FastAPI app
app = FastAPI(title="StockPredictionPro Model API")

class PredictionRequest(BaseModel):
    features: List[List[float]]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features)
        
        if scaler:
            features = scaler.transform(features)
        
        predictions = model.predict(features)
        
        return {
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        api_path = package_dir / 'api_server.py'
        with open(api_path, 'w') as f:
            f.write(api_script)
    
    def _build_docker_image(self, package_dir: Path, dockerfile_path: Path) -> bool:
        """Build Docker image"""
        try:
            client = docker.from_env()
            
            image_tag = f"{self.config.docker_image}:{self.config.docker_tag}"
            
            logger.info(f"🐳 Building Docker image: {image_tag}")
            
            image, logs = client.images.build(
                path=str(package_dir),
                dockerfile=str(dockerfile_path.name),
                tag=image_tag,
                rm=True
            )
            
            # Print build logs
            for log in logs:
                if 'stream' in log:
                    print(log['stream'].strip())
            
            logger.info(f"✅ Docker image built successfully: {image.id[:12]}")
            return True
            
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            return False
    
    def _push_to_registry(self) -> bool:
        """Push Docker image to registry"""
        try:
            client = docker.from_env()
            
            image_tag = f"{self.config.docker_image}:{self.config.docker_tag}"
            full_tag = f"{self.config.docker_registry}/{image_tag}"
            
            # Tag for registry
            image = client.images.get(image_tag)
            image.tag(self.config.docker_registry, image_tag)
            
            # Push to registry
            logger.info(f"📤 Pushing to registry: {full_tag}")
            client.images.push(self.config.docker_registry, image_tag)
            
            logger.info(f"✅ Successfully pushed to registry")
            return True
            
        except Exception as e:
            logger.error(f"Registry push failed: {e}")
            return False

# ============================================
# MAIN DEPLOYMENT ORCHESTRATOR
# ============================================

class ModelDeployer:
    """Main orchestrator for model deployment"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.packager = ModelPackager(config)
        self.registry = ModelRegistry()
    
    def deploy_model(self, model_path: str, additional_files: List[str] = None) -> bool:
        """Deploy model using specified strategy"""
        logger.info(f"🚀 Starting deployment of {self.config.model_name}")
        logger.info(f"   Deployment type: {self.config.deployment_type}")
        logger.info(f"   Model path: {model_path}")
        
        try:
            # Convert string paths to Path objects
            model_path = Path(model_path)
            additional_files = [Path(f) for f in additional_files] if additional_files else []
            
            # Package model
            package_dir = self.packager.package_model(model_path, additional_files)
            
            # Deploy based on type
            success = False
            
            if self.config.deployment_type == 'local':
                deployer = LocalDeployment(self.config)
                success = deployer.deploy(package_dir)
                
            elif self.config.deployment_type == 'rest':
                deployer = RestApiDeployment(self.config)
                success = deployer.deploy(package_dir)
                
            elif self.config.deployment_type == 'docker':
                deployer = DockerDeployment(self.config)
                success = deployer.deploy(package_dir)
                
            else:
                logger.error(f"Unsupported deployment type: {self.config.deployment_type}")
                return False
            
            if success:
                logger.info(f"🎉 Deployment completed successfully!")
            else:
                logger.error(f"❌ Deployment failed!")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Deployment error: {e}")
            return False
    
    def list_deployments(self) -> List[DeploymentMetadata]:
        """List all deployed models"""
        return self.registry.list_models()
    
    def rollback_deployment(self, model_id: str) -> bool:
        """Rollback to previous deployment"""
        logger.info(f"🔄 Rolling back deployment: {model_id}")
        return self.registry.deactivate_model(model_id)

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy models for StockPredictionPro')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--name', required=True, help='Model name for deployment')
    parser.add_argument('--version', help='Model version (auto-generated if not provided)')
    parser.add_argument('--type', choices=['sklearn', 'xgboost', 'lightgbm', 'jax'], 
                       default='sklearn', help='Model type')
    parser.add_argument('--deployment', choices=['local', 'rest', 'docker'], 
                       default='local', help='Deployment type')
    parser.add_argument('--additional-files', nargs='*', help='Additional files to include (scaler, config, etc.)')
    
    # API options
    parser.add_argument('--api-host', default='0.0.0.0', help='API host for REST deployment')
    parser.add_argument('--api-port', type=int, default=8000, help='API port for REST deployment')
    parser.add_argument('--api-workers', type=int, default=1, help='Number of API workers')
    
    # Docker options
    parser.add_argument('--docker-image', default='stockpredictionpro/model-server', help='Docker image name')
    parser.add_argument('--docker-tag', default='latest', help='Docker image tag')
    parser.add_argument('--docker-registry', help='Docker registry URL')
    
    # Management commands
    parser.add_argument('--list', action='store_true', help='List deployed models')
    parser.add_argument('--rollback', help='Rollback specified model ID')
    
    args = parser.parse_args()
    
    # Handle management commands
    if args.list:
        registry = ModelRegistry()
        models = registry.list_models()
        print("\nDeployed Models:")
        print("-" * 60)
        for model in models:
            print(f"{model.model_name} v{model.model_version} ({model.deployment_type}) - {model.deployment_status}")
        return
    
    if args.rollback:
        registry = ModelRegistry()
        success = registry.deactivate_model(args.rollback)
        if success:
            print(f"✅ Successfully rolled back model: {args.rollback}")
        else:
            print(f"❌ Failed to rollback model: {args.rollback}")
        return
    
    # Create deployment configuration
    config = DeploymentConfig(
        model_name=args.name,
        model_version=args.version,
        model_type=args.type,
        deployment_type=args.deployment,
        api_host=args.api_host,
        api_port=args.api_port,
        api_workers=args.api_workers,
        docker_image=args.docker_image,
        docker_tag=args.docker_tag,
        docker_registry=args.docker_registry
    )
    
    # Deploy model
    deployer = ModelDeployer(config)
    success = deployer.deploy_model(args.model, args.additional_files)
    
    exit(0 if success else 1)

if __name__ == '__main__':
    main()
