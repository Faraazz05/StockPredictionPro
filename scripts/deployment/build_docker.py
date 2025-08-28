"""
scripts/deployment/build_docker.py

Automated Docker container building and management for StockPredictionPro.
Creates optimized multi-stage Docker images, handles versioning, pushes to registries,
and manages deployment-ready containers with comprehensive configuration.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import subprocess
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil

# Docker SDK (optional)
try:
    import docker
    HAS_DOCKER_SDK = True
except ImportError:
    HAS_DOCKER_SDK = False

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'docker_build_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.DockerBuild')

# Directory configuration
PROJECT_ROOT = Path('.')
DOCKER_DIR = PROJECT_ROOT / 'docker'
DEPLOYMENT_DIR = PROJECT_ROOT / 'deployment'
CONFIG_DIR = PROJECT_ROOT / 'config'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

# Ensure directories exist
for dir_path in [DOCKER_DIR, DEPLOYMENT_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class DockerConfig:
    """Configuration for Docker build process"""
    # Image settings
    image_name: str = 'stockpredictionpro'
    base_image: str = 'python:3.13-slim'
    registry_url: str = None  # e.g., 'registry.hub.docker.com'
    organization: str = 'stockpredictionpro'
    
    # Build settings
    use_multi_stage: bool = True
    enable_cache: bool = True
    optimize_size: bool = True
    include_dev_tools: bool = False
    
    # Versioning
    auto_version: bool = True
    version_prefix: str = 'v'
    tag_latest: bool = True
    include_git_hash: bool = True
    
    # Registry settings
    push_to_registry: bool = False
    registry_username: str = None
    registry_password: str = None
    
    # Security settings
    scan_vulnerabilities: bool = True
    use_distroless: bool = False
    non_root_user: bool = True
    
    # Components to include
    include_api: bool = True
    include_training: bool = True
    include_automation: bool = True
    include_notebooks: bool = False
    
    # Build context
    dockerignore_patterns: List[str] = None
    
    def __post_init__(self):
        if self.dockerignore_patterns is None:
            self.dockerignore_patterns = [
                '__pycache__',
                '*.pyc',
                '.git',
                '.pytest_cache',
                'logs/*.log',
                'data/cache',
                'data/temp',
                '*.md',
                '.vscode',
                '.idea'
            ]

@dataclass
class BuildResult:
    """Results from Docker build operation"""
    image_id: str
    image_tags: List[str]
    build_time: float
    image_size_mb: float
    status: str  # success, failed
    build_logs: List[str]
    vulnerabilities: Optional[Dict[str, Any]] = None
    pushed_to_registry: bool = False
    registry_urls: List[str] = None
    
    def __post_init__(self):
        if self.registry_urls is None:
            self.registry_urls = []

@dataclass
class DockerBuildReport:
    """Comprehensive Docker build report"""
    build_timestamp: str
    config_used: DockerConfig
    build_results: List[BuildResult]
    total_build_time: float
    overall_status: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# DOCKERFILE GENERATOR
# ============================================

class DockerfileGenerator:
    """Generate optimized Dockerfiles for different deployment scenarios"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
    
    def generate_dockerfile(self, target: str = 'production') -> str:
        """Generate Dockerfile content based on target environment"""
        if self.config.use_multi_stage:
            return self._generate_multistage_dockerfile(target)
        else:
            return self._generate_simple_dockerfile(target)
    
    def _generate_multistage_dockerfile(self, target: str) -> str:
        """Generate multi-stage Dockerfile for optimized builds"""
        dockerfile_content = f'''# Multi-stage Dockerfile for StockPredictionPro
# Generated on: {datetime.now().isoformat()}

# ============================================
# Stage 1: Base Dependencies
# ============================================
FROM {self.config.base_image} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# ============================================
# Stage 2: Development Dependencies (optional)
# ============================================
FROM base as development

RUN pip install --no-cache-dir -r requirements.txt

# Install development tools
RUN apt-get update && apt-get install -y \\
    vim \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 3: Application Code
# ============================================
FROM {"development" if self.config.include_dev_tools else "base"} as application

# Copy application code
COPY scripts/ ./scripts/
COPY config/ ./config/
'''

        # Add conditional components
        if self.config.include_api:
            dockerfile_content += '''COPY api/ ./api/
'''

        if self.config.include_notebooks:
            dockerfile_content += '''COPY notebooks/ ./notebooks/
'''

        # Add models and data structure
        dockerfile_content += '''
# Create necessary directories
RUN mkdir -p logs data/processed data/cache models/production outputs

# Set permissions
RUN chmod +x scripts/automation/*.py
RUN chmod +x scripts/models/*.py
'''

        # Add user creation for security
        if self.config.non_root_user:
            dockerfile_content += '''
# Create non-root user
RUN groupadd -r stockpro && useradd -r -g stockpro stockpro
RUN chown -R stockpro:stockpro /app
USER stockpro
'''

        # Add final stage
        dockerfile_content += f'''
# ============================================
# Stage 4: Production Image
# ============================================
FROM application as production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Default command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 5: Training Image
# ============================================
FROM application as training

# Install additional ML dependencies
RUN pip install --no-cache-dir \\
    optuna \\
    mlflow \\
    tensorboard

# Set training-specific environment
ENV MODEL_TRAINING=true

# Default command for training
CMD ["python", "scripts/models/train_all_models.py"]

# ============================================
# Stage 6: Automation Image
# ============================================
FROM application as automation

# Install cron for scheduling
USER root
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Setup cron jobs
COPY deployment/crontab /etc/cron.d/stockpro-cron
RUN chmod 0644 /etc/cron.d/stockpro-cron
RUN crontab /etc/cron.d/stockpro-cron

{"USER stockpro" if self.config.non_root_user else ""}

# Default command for automation
CMD ["cron", "-f"]
'''

        return dockerfile_content
    
    def _generate_simple_dockerfile(self, target: str) -> str:
        """Generate simple single-stage Dockerfile"""
        dockerfile_content = f'''# Simple Dockerfile for StockPredictionPro
# Generated on: {datetime.now().isoformat()}

FROM {self.config.base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data models outputs

'''

        if self.config.non_root_user:
            dockerfile_content += '''
# Create non-root user
RUN groupadd -r stockpro && useradd -r -g stockpro stockpro
RUN chown -R stockpro:stockpro /app
USER stockpro
'''

        dockerfile_content += '''
# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

        return dockerfile_content
    
    def generate_dockerignore(self) -> str:
        """Generate .dockerignore file content"""
        dockerignore_content = '''# Docker ignore file for StockPredictionPro
# Generated automatically

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Development
.git/
.gitignore
.pytest_cache/
.coverage
.vscode/
.idea/
*.md
README*

# Data and logs
data/cache/
data/temp/
logs/*.log
*.log

# Models (use volume mounts instead)
models/experiments/
models/backup/

# Outputs
outputs/visualizations/
outputs/temp/

# System
.DS_Store
Thumbs.db
*.swp
*.swo

# Docker
Dockerfile*
docker-compose*
.dockerignore

'''
        
        # Add custom patterns
        for pattern in self.config.dockerignore_patterns:
            dockerignore_content += f"{pattern}\n"
        
        return dockerignore_content
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for development"""
        compose_content = f'''version: '3.8'

services:
  # Main API service
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://stockpro:password@db:5432/stockpro
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Training service
  training:
    build:
      context: .
      dockerfile: Dockerfile
      target: training
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://stockpro:password@db:5432/stockpro
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
    restart: "no"
    profiles:
      - training

  # Automation service
  automation:
    build:
      context: .
      dockerfile: Dockerfile
      target: automation
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://stockpro:password@db:5432/stockpro
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped
    profiles:
      - automation

  # Database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=stockpro
      - POSTGRES_USER=stockpro
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./deployment/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U stockpro"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Monitoring (optional)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    profiles:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./deployment/grafana:/etc/grafana/provisioning
    restart: unless-stopped
    profiles:
      - monitoring

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: stockpro-network
'''
        
        return compose_content

# ============================================
# DOCKER BUILDER
# ============================================

class DockerBuilder:
    """Main Docker building and management class"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
        self.dockerfile_generator = DockerfileGenerator(config)
        self.docker_client = None
        
        if HAS_DOCKER_SDK:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Could not connect to Docker daemon via SDK: {e}")
    
    def build_images(self, targets: List[str] = None) -> List[BuildResult]:
        """Build Docker images for specified targets"""
        if targets is None:
            targets = ['production']
            if self.config.include_training:
                targets.append('training')
            if self.config.include_automation:
                targets.append('automation')
        
        logger.info(f"🐳 Building Docker images for targets: {targets}")
        
        # Prepare build context
        self._prepare_build_context()
        
        build_results = []
        
        for target in targets:
            try:
                result = self._build_single_image(target)
                build_results.append(result)
                
                if result.status == 'success':
                    logger.info(f"✅ Successfully built {target} image: {result.image_id[:12]}")
                else:
                    logger.error(f"❌ Failed to build {target} image")
                    
            except Exception as e:
                logger.error(f"❌ Build failed for {target}: {e}")
                build_results.append(BuildResult(
                    image_id='',
                    image_tags=[],
                    build_time=0.0,
                    image_size_mb=0.0,
                    status='failed',
                    build_logs=[str(e)]
                ))
        
        return build_results
    
    def _prepare_build_context(self) -> None:
        """Prepare Docker build context"""
        logger.info("📝 Preparing Docker build context...")
        
        # Generate Dockerfile
        dockerfile_content = self.dockerfile_generator.generate_dockerfile()
        dockerfile_path = DOCKER_DIR / 'Dockerfile'
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Generate .dockerignore
        dockerignore_content = self.dockerfile_generator.generate_dockerignore()
        dockerignore_path = PROJECT_ROOT / '.dockerignore'
        
        with open(dockerignore_path, 'w') as f:
            f.write(dockerignore_content)
        
        # Generate docker-compose.yml
        compose_content = self.dockerfile_generator.generate_docker_compose()
        compose_path = DOCKER_DIR / 'docker-compose.yml'
        
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        # Generate requirements files
        self._generate_requirements_files()
        
        # Generate additional config files
        self._generate_config_files()
        
        logger.info("✅ Build context prepared")
    
    def _generate_requirements_files(self) -> None:
        """Generate requirements files for different environments"""
        
        # Base production requirements
        prod_requirements = '''# Production requirements for StockPredictionPro
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pandas==2.1.3
numpy==1.24.4
scikit-learn==1.3.2
xgboost==1.7.6
lightgbm==4.0.0
joblib==1.3.2
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
requests==2.31.0
aiofiles==23.2.1
'''
        
        # Development requirements (includes production + dev tools)
        dev_requirements = prod_requirements + '''
# Development dependencies
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1
jupyter==1.0.0
ipykernel==6.26.0
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
optuna==3.4.0
mlflow==2.8.1
'''
        
        # Write requirements files
        with open(PROJECT_ROOT / 'requirements-prod.txt', 'w') as f:
            f.write(prod_requirements)
        
        with open(PROJECT_ROOT / 'requirements.txt', 'w') as f:
            f.write(dev_requirements)
    
    def _generate_config_files(self) -> None:
        """Generate additional configuration files"""
        
        # Generate crontab for automation container
        crontab_content = '''# Crontab for StockPredictionPro automation
# Run daily update at 6 AM
0 6 * * * cd /app && python scripts/automation/daily_update.py >> logs/cron.log 2>&1

# Run health check every 15 minutes
*/15 * * * * cd /app && python scripts/automation/health_check.py >> logs/health_cron.log 2>&1

# Run weekly retrain on Sundays at 2 AM
0 2 * * 0 cd /app && python scripts/automation/weekly_retrain.py >> logs/retrain_cron.log 2>&1

# Run monthly cleanup on 1st of each month at 3 AM
0 3 1 * * cd /app && python scripts/automation/monthly_cleanup.py >> logs/cleanup_cron.log 2>&1
'''
        
        crontab_dir = DEPLOYMENT_DIR
        crontab_dir.mkdir(exist_ok=True)
        
        with open(crontab_dir / 'crontab', 'w') as f:
            f.write(crontab_content)
    
    def _build_single_image(self, target: str) -> BuildResult:
        """Build a single Docker image"""
        start_time = time.time()
        build_logs = []
        
        # Generate image tags
        tags = self._generate_image_tags(target)
        
        try:
            if self.docker_client and HAS_DOCKER_SDK:
                # Use Docker SDK
                result = self._build_with_sdk(target, tags, build_logs)
            else:
                # Use Docker CLI
                result = self._build_with_cli(target, tags, build_logs)
            
            build_time = time.time() - start_time
            
            # Get image size
            image_size_mb = self._get_image_size(result['image_id']) if result['image_id'] else 0.0
            
            # Scan for vulnerabilities if enabled
            vulnerabilities = None
            if self.config.scan_vulnerabilities and result['image_id']:
                vulnerabilities = self._scan_vulnerabilities(result['image_id'])
            
            # Push to registry if configured
            pushed_to_registry = False
            registry_urls = []
            
            if self.config.push_to_registry and result['image_id']:
                pushed_to_registry, registry_urls = self._push_to_registry(tags)
            
            return BuildResult(
                image_id=result['image_id'],
                image_tags=tags,
                build_time=build_time,
                image_size_mb=image_size_mb,
                status='success' if result['image_id'] else 'failed',
                build_logs=build_logs,
                vulnerabilities=vulnerabilities,
                pushed_to_registry=pushed_to_registry,
                registry_urls=registry_urls
            )
            
        except Exception as e:
            build_time = time.time() - start_time
            build_logs.append(f"Build failed: {e}")
            
            return BuildResult(
                image_id='',
                image_tags=tags,
                build_time=build_time,
                image_size_mb=0.0,
                status='failed',
                build_logs=build_logs
            )
    
    def _build_with_sdk(self, target: str, tags: List[str], build_logs: List[str]) -> Dict[str, Any]:
        """Build image using Docker SDK"""
        try:
            dockerfile_path = DOCKER_DIR / 'Dockerfile'
            
            # Build image
            image, logs = self.docker_client.images.build(
                path=str(PROJECT_ROOT),
                dockerfile=str(dockerfile_path),
                target=target,
                tag=tags[0],
                rm=True,
                nocache=not self.config.enable_cache,
                forcerm=True
            )
            
            # Collect build logs
            for log in logs:
                if 'stream' in log:
                    build_logs.append(log['stream'].strip())
            
            # Tag additional tags
            for tag in tags[1:]:
                image.tag(tag)
            
            return {'image_id': image.id}
            
        except Exception as e:
            build_logs.append(f"SDK build failed: {e}")
            return {'image_id': ''}
    
    def _build_with_cli(self, target: str, tags: List[str], build_logs: List[str]) -> Dict[str, Any]:
        """Build image using Docker CLI"""
        try:
            dockerfile_path = DOCKER_DIR / 'Dockerfile'
            
            # Build command
            cmd = [
                'docker', 'build',
                '-f', str(dockerfile_path),
                '--target', target,
                '-t', tags[0]
            ]
            
            # Add additional tags
            for tag in tags[1:]:
                cmd.extend(['-t', tag])
            
            # Add cache options
            if not self.config.enable_cache:
                cmd.append('--no-cache')
            
            # Add context
            cmd.append(str(PROJECT_ROOT))
            
            # Execute build
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Collect output
            for line in process.stdout:
                build_logs.append(line.strip())
                logger.debug(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                # Get image ID
                image_id = self._get_image_id(tags[0])
                return {'image_id': image_id}
            else:
                build_logs.append(f"Build failed with return code {process.returncode}")
                return {'image_id': ''}
                
        except Exception as e:
            build_logs.append(f"CLI build failed: {e}")
            return {'image_id': ''}
    
    def _generate_image_tags(self, target: str) -> List[str]:
        """Generate image tags for the build"""
        tags = []
        
        # Base tag
        base_tag = f"{self.config.image_name}-{target}"
        
        if self.config.registry_url:
            base_tag = f"{self.config.registry_url}/{self.config.organization}/{base_tag}"
        elif self.config.organization:
            base_tag = f"{self.config.organization}/{base_tag}"
        
        # Version tag
        if self.config.auto_version:
            version = self._generate_version()
            tags.append(f"{base_tag}:{self.config.version_prefix}{version}")
        
        # Latest tag
        if self.config.tag_latest:
            tags.append(f"{base_tag}:latest")
        
        # Git hash tag
        if self.config.include_git_hash:
            git_hash = self._get_git_hash()
            if git_hash:
                tags.append(f"{base_tag}:{git_hash[:8]}")
        
        return tags if tags else [f"{base_tag}:latest"]
    
    def _generate_version(self) -> str:
        """Generate version string"""
        # Use timestamp-based versioning
        return datetime.now().strftime('%Y%m%d.%H%M%S')
    
    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            
        except Exception as e:
            logger.debug(f"Could not get git hash: {e}")
        
        return None
    
    def _get_image_id(self, tag: str) -> str:
        """Get image ID from tag"""
        try:
            result = subprocess.run(
                ['docker', 'images', '-q', tag],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
                
        except Exception as e:
            logger.debug(f"Could not get image ID: {e}")
        
        return ''
    
    def _get_image_size(self, image_id: str) -> float:
        """Get image size in MB"""
        try:
            if self.docker_client and HAS_DOCKER_SDK:
                image = self.docker_client.images.get(image_id)
                return image.attrs['Size'] / (1024 * 1024)
            else:
                result = subprocess.run(
                    ['docker', 'images', '--format', '{{.Size}}', image_id],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    size_str = result.stdout.strip()
                    # Parse size string (e.g., "1.2GB", "500MB")
                    if 'GB' in size_str:
                        return float(size_str.replace('GB', '')) * 1024
                    elif 'MB' in size_str:
                        return float(size_str.replace('MB', ''))
                    
        except Exception as e:
            logger.debug(f"Could not get image size: {e}")
        
        return 0.0
    
    def _scan_vulnerabilities(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Scan image for vulnerabilities"""
        try:
            # Try using docker scan or trivy
            scan_tools = ['trivy', 'docker scan']
            
            for tool in scan_tools:
                try:
                    if tool == 'trivy':
                        result = subprocess.run(
                            ['trivy', 'image', '--format', 'json', image_id],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                    else:
                        result = subprocess.run(
                            ['docker', 'scan', '--json', image_id],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                    
                    if result.returncode == 0:
                        return json.loads(result.stdout)
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            logger.warning("No vulnerability scanner available")
            return None
            
        except Exception as e:
            logger.warning(f"Vulnerability scan failed: {e}")
            return None
    
    def _push_to_registry(self, tags: List[str]) -> Tuple[bool, List[str]]:
        """Push images to registry"""
        if not self.config.registry_url:
            logger.warning("No registry URL configured")
            return False, []
        
        try:
            # Login to registry if credentials provided
            if self.config.registry_username and self.config.registry_password:
                self._login_to_registry()
            
            pushed_urls = []
            
            for tag in tags:
                try:
                    result = subprocess.run(
                        ['docker', 'push', tag],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    if result.returncode == 0:
                        pushed_urls.append(tag)
                        logger.info(f"✅ Pushed to registry: {tag}")
                    else:
                        logger.error(f"❌ Failed to push {tag}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ Push timeout for {tag}")
                    
            return len(pushed_urls) > 0, pushed_urls
            
        except Exception as e:
            logger.error(f"Registry push failed: {e}")
            return False, []
    
    def _login_to_registry(self) -> None:
        """Login to Docker registry"""
        try:
            process = subprocess.Popen(
                ['docker', 'login', self.config.registry_url, 
                 '--username', self.config.registry_username, '--password-stdin'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=self.config.registry_password)
            
            if process.returncode == 0:
                logger.info("✅ Successfully logged into registry")
            else:
                logger.error(f"❌ Registry login failed: {stderr}")
                
        except Exception as e:
            logger.error(f"Registry login error: {e}")

# ============================================
# MAIN ORCHESTRATOR
# ============================================

class DockerBuildOrchestrator:
    """Main orchestrator for Docker build process"""
    
    def __init__(self, config: DockerConfig = None):
        self.config = config or DockerConfig()
        self.builder = DockerBuilder(self.config)
    
    def build_and_deploy(self, targets: List[str] = None) -> DockerBuildReport:
        """Complete build and deployment process"""
        logger.info("🚀 Starting Docker build and deployment process...")
        start_time = time.time()
        
        try:
            # Validate Docker environment
            self._validate_docker_environment()
            
            # Build images
            build_results = self.builder.build_images(targets)
            
            # Generate report
            total_time = time.time() - start_time
            report = self._generate_report(build_results, total_time)
            
            # Save report
            self._save_report(report)
            
            # Print summary
            self._print_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Docker build process failed: {e}")
            
            # Create failure report
            total_time = time.time() - start_time
            return DockerBuildReport(
                build_timestamp=datetime.now().isoformat(),
                config_used=self.config,
                build_results=[],
                total_build_time=total_time,
                overall_status='failed',
                recommendations=[f"Build process failed: {e}"]
            )
    
    def _validate_docker_environment(self) -> None:
        """Validate Docker environment"""
        try:
            # Check if Docker is installed and running
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError("Docker is not installed or not in PATH")
            
            # Check if Docker daemon is running
            result = subprocess.run(
                ['docker', 'info'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError("Docker daemon is not running")
            
            logger.info("✅ Docker environment validated")
            
        except Exception as e:
            logger.error(f"❌ Docker environment validation failed: {e}")
            raise
    
    def _generate_report(self, build_results: List[BuildResult], total_time: float) -> DockerBuildReport:
        """Generate comprehensive build report"""
        
        # Determine overall status
        successful_builds = [r for r in build_results if r.status == 'success']
        failed_builds = [r for r in build_results if r.status == 'failed']
        
        if len(failed_builds) == 0:
            overall_status = 'success'
        elif len(successful_builds) > 0:
            overall_status = 'partial_success'
        else:
            overall_status = 'failed'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(build_results)
        
        return DockerBuildReport(
            build_timestamp=datetime.now().isoformat(),
            config_used=self.config,
            build_results=build_results,
            total_build_time=total_time,
            overall_status=overall_status,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, build_results: List[BuildResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for large images
        large_images = [r for r in build_results if r.image_size_mb > 1000]  # >1GB
        if large_images:
            recommendations.append("Consider optimizing large images (>1GB) using multi-stage builds")
        
        # Check for vulnerabilities
        vulnerable_images = [r for r in build_results if r.vulnerabilities and 
                           r.vulnerabilities.get('high_severity', 0) > 0]
        if vulnerable_images:
            recommendations.append("Address high-severity vulnerabilities in images")
        
        # Check for failed builds
        failed_builds = [r for r in build_results if r.status == 'failed']
        if failed_builds:
            recommendations.append("Review and fix failed builds before deployment")
        
        # Registry push recommendations
        unpushed_images = [r for r in build_results if r.status == 'success' and not r.pushed_to_registry]
        if unpushed_images and self.config.push_to_registry:
            recommendations.append("Some images were not pushed to registry - check credentials and connectivity")
        
        if not recommendations:
            recommendations.append("All builds completed successfully - ready for deployment")
        
        return recommendations
    
    def _save_report(self, report: DockerBuildReport) -> None:
        """Save build report"""
        try:
            report_path = DEPLOYMENT_DIR / f"docker_build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report.save(report_path)
            
            # Save latest report
            latest_path = DEPLOYMENT_DIR / "docker_build_latest.json"
            report.save(latest_path)
            
            logger.info(f"💾 Build report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save build report: {e}")
    
    def _print_summary(self, report: DockerBuildReport) -> None:
        """Print build summary"""
        print("\n" + "="*60)
        print("DOCKER BUILD SUMMARY")
        print("="*60)
        print(f"Build Time: {report.total_build_time/60:.1f} minutes")
        print(f"Overall Status: {report.overall_status.upper()}")
        
        print(f"\nBuild Results:")
        print("-" * 40)
        
        for result in report.build_results:
            status_emoji = "✅" if result.status == 'success' else "❌"
            registry_status = "📤 Pushed" if result.pushed_to_registry else "📥 Local"
            
            print(f"{status_emoji} {result.image_tags[0] if result.image_tags else 'N/A'}")
            print(f"   Size: {result.image_size_mb:.1f} MB")
            print(f"   Build Time: {result.build_time:.1f}s")
            print(f"   Registry: {registry_status}")
            
            if result.vulnerabilities:
                high_vuln = result.vulnerabilities.get('high_severity', 0)
                if high_vuln > 0:
                    print(f"   ⚠️ Vulnerabilities: {high_vuln} high severity")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

def load_config_from_file(config_path: str) -> DockerConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return DockerConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return DockerConfig()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Docker images for StockPredictionPro')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--targets', nargs='+', 
                       choices=['production', 'training', 'automation', 'development'],
                       help='Build targets')
    parser.add_argument('--push', action='store_true', help='Push images to registry')
    parser.add_argument('--no-cache', action='store_true', help='Disable build cache')
    parser.add_argument('--registry', help='Docker registry URL')
    parser.add_argument('--org', help='Organization/namespace')
    parser.add_argument('--tag', help='Additional tag for images')
    parser.add_argument('--scan', action='store_true', help='Enable vulnerability scanning')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = DockerConfig()
    
    # Override config with command line arguments
    if args.push:
        config.push_to_registry = True
    if args.no_cache:
        config.enable_cache = False
    if args.registry:
        config.registry_url = args.registry
    if args.org:
        config.organization = args.org
    if args.scan:
        config.scan_vulnerabilities = True
    
    # Build images
    orchestrator = DockerBuildOrchestrator(config)
    report = orchestrator.build_and_deploy(args.targets)
    
    # Exit with appropriate code
    if report.overall_status == 'success':
        sys.exit(0)
    elif report.overall_status == 'partial_success':
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == '__main__':
    main()
