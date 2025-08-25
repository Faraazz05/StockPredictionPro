"""
scripts/models/deploy_models.py

Deploy trained models for StockPredictionPro to production environments.
Supports local deployment, REST API serving (via FastAPI), Docker packaging, and cloud upload.
Handles model versioning, rollback, and metadata tracking.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime
import joblib
import json

# For REST deployment
try:
    import fastapi
    import uvicorn
except ImportError:
    fastapi = None
    uvicorn = None

# Cloud upload (example: AWS S3)
try:
    import boto3
except ImportError:
    boto3 = None

MODEL_DIR = Path('./models/trained')
DEPLOY_DIR = Path('./models/production')
METADATA_FILE = DEPLOY_DIR / 'model_metadata.json'
LOG_FILE = Path('./logs/model_deploy.log')

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DeployModels')


def load_model(model_path: Path):
    """Load a trained model artifact from file."""
    logger.info(f"Loading model from {model_path}")
    if model_path.suffix in ['.pkl', '.joblib']:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    logger.error(f"Unsupported model format: {model_path}")
    raise NotImplementedError("Only .pkl or .joblib supported")


def package_model(model_file: Path, destination: Path, extra_files: list = None, version: str = None) -> Path:
    """
    Package model with metadata and move to production deployment.
    Copies model and related files.
    """
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    deployment_version = version or f"{timestamp}"
    package_name = f"{model_file.stem}_{deployment_version}{model_file.suffix}"
    package_path = destination / package_name

    shutil.copy2(str(model_file), str(package_path))
    logger.info(f"Model packaged to {package_path}")
    # Copy extra files if any
    if extra_files:
        for f in extra_files:
            shutil.copy2(str(f), str(destination / f.name))
            logger.info(f"Included extra file {f}")

    # Save metadata
    metadata = {
        "model_file": package_name,
        "deployed_at": timestamp,
        "version": deployment_version,
        "extra_files": [str(f.name) for f in extra_files] if extra_files else []
    }
    with open(str(destination / f"{model_file.stem}_metadata_{deployment_version}.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    return package_path


def deploy_rest_api(model_path: Path, host='0.0.0.0', port=8000):
    """
    Serve the model via a REST API with FastAPI (requires fastapi/uvicorn).
    Supports basic infer endpoint for demonstration.
    """
    if fastapi is None or uvicorn is None:
        logger.error("FastAPI or Uvicorn not installed. Cannot deploy REST API.")
        raise ImportError("fastapi and uvicorn required for REST API deployment.")

    model = load_model(model_path)
    app = fastapi.FastAPI()

    @app.post("/predict")
    async def predict(input: dict):
        # Example: expects {"features": [..]}
        features = input.get("features", [])
        if not features:
            return {"error": "No features provided"}
        try:
            prediction = model.predict([features])
            return {"prediction": prediction[0]}
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"error": str(e)}

    logger.info(f"Starting REST API at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def upload_to_s3(model_path: Path, bucket: str, s3_key: str, aws_profile: str = None):
    """
    Upload a model artifact to AWS S3 bucket.
    """
    if boto3 is None:
        logger.error("boto3 not available for S3 upload.")
        raise ImportError("boto3 required for S3 deployment.")

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client('s3')
    try:
        s3.upload_file(str(model_path), bucket, s3_key)
        logger.info(f"Model uploaded to s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deploy models for StockPredictionPro")
    parser.add_argument('--model', required=True, type=str, help="Trained model file to deploy (.pkl or .joblib)")
    parser.add_argument('--deploy-type', choices=['local', 'rest', 's3'], default='local', help="Type of deployment")
    parser.add_argument('--version', type=str, help="Version/tag for deployment")
    parser.add_argument('--extra-files', nargs='*', type=str, help="Extra files to include")
    parser.add_argument('--rest-host', type=str, default='0.0.0.0', help="REST API host (if --deploy-type rest)")
    parser.add_argument('--rest-port', type=int, default=8000, help="REST API port (if --deploy-type rest)")
    parser.add_argument('--s3-bucket', type=str, help="AWS S3 bucket name (if --deploy-type s3)")
    parser.add_argument('--s3-key', type=str, help="AWS S3 key/path (if --deploy-type s3)")
    parser.add_argument('--aws-profile', type=str, help="AWS profile name (if --deploy-type s3)")
    args = parser.parse_args()

    model_file = Path(args.model)
    extra_files = [Path(f) for f in args.extra_files] if args.extra_files else []

    if args.deploy_type == 'local':
        deployed_path = package_model(model_file, DEPLOY_DIR, extra_files, args.version)
        logger.info(f"Model deployed locally to: {deployed_path}")

    elif args.deploy_type == 'rest':
        deploy_rest_api(model_file, host=args.rest_host, port=args.rest_port)

    elif args.deploy_type == 's3':
        if not args.s3_bucket or not args.s3_key:
            logger.error("S3 bucket and key required for S3 deployment.")
            sys.exit(1)
        package_path = package_model(model_file, DEPLOY_DIR, extra_files, args.version)
        upload_to_s3(package_path, args.s3_bucket, args.s3_key, args.aws_profile)

    logger.info("Model deployment process finished.")

if __name__ == '__main__':
    main()
