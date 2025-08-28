"""
scripts/deployment/setup_ci_cd.py

CI/CD pipeline setup and configuration for StockPredictionPro.
Supports GitHub Actions, GitLab CI, Jenkins, and Azure DevOps with automated
testing, building, security scanning, and deployment workflows.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import secrets
import string

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'ci_cd_setup_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.CICDSetup')

# Directory configuration
PROJECT_ROOT = Path('.')
GITHUB_DIR = PROJECT_ROOT / '.github'
WORKFLOWS_DIR = GITHUB_DIR / 'workflows'
GITLAB_CI_FILE = PROJECT_ROOT / '.gitlab-ci.yml'
JENKINS_FILE = PROJECT_ROOT / 'Jenkinsfile'
AZURE_FILE = PROJECT_ROOT / 'azure-pipelines.yml'

# Ensure directories exist
for dir_path in [GITHUB_DIR, WORKFLOWS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class CICDConfig:
    """Configuration for CI/CD pipeline setup"""
    platform: str = 'github'
    repository_url: str = 'https://github.com/username/stockpredictionpro'
    main_branch: str = 'main'
    develop_branch: str = 'develop'
    registry_url: str = 'docker.io'
    image_name: str = 'stockpredictionpro'
    organization: str = 'stockpredictionpro'
    python_version: str = '3.13'
    enable_security_scanning: bool = True
    auto_deploy_develop: bool = True
    test_coverage_threshold: int = 80
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.platform not in ['github', 'gitlab', 'jenkins', 'azure']:
            raise ValueError(f"Unsupported platform: {self.platform}")

@dataclass
class CICDSetupResult:
    """Results from CI/CD setup"""
    platform: str
    files_created: List[str]
    secrets_required: List[str]
    manual_steps: List[str]
    setup_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class GitHubActionsGenerator:
    """Generate GitHub Actions workflows"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
    
    def generate_main_workflow(self) -> str:
        """Generate main CI/CD workflow"""
        return f"""name: CI/CD Pipeline

on:
  push:
    branches: [ {self.config.main_branch}, {self.config.develop_branch} ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ {self.config.main_branch} ]

env:
  PYTHON_VERSION: {self.config.python_version}
  REGISTRY: {self.config.registry_url}
  IMAGE_NAME: {self.config.organization}/{self.config.image_name}

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_USER: postgres
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ env.PYTHON_VERSION }}}}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{{{ runner.os }}}}-pip-${{{{ hashFiles('**/requirements*.txt') }}}}
        restore-keys: |
          ${{{{ runner.os }}}}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check .
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=scripts --cov=api --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb
        REDIS_URL: redis://localhost:6379/0
    
    - name: Check coverage
      run: |
        coverage report --fail-under={self.config.test_coverage_threshold}

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
    
    - name: Run security checks
      run: |
        pip install bandit safety
        bandit -r scripts/ api/
        safety check

  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test]
    if: github.event_name != 'pull_request'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ secrets.REGISTRY_USERNAME }}}}
        password: ${{{{ secrets.REGISTRY_PASSWORD }}}}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:${{{{ github.sha }}}}
          ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-dev:
    name: Deploy to Development
    runs-on: ubuntu-latest
    needs: [build]
    if: github.ref == 'refs/heads/{self.config.develop_branch}'
    environment: development
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Deploy to Kubernetes
      run: |
        echo "Deploying to development environment"
        kubectl apply -f k8s/
      env:
        KUBECONFIG: ${{{{ secrets.KUBECONFIG }}}}
    
    - name: Notify deployment
      if: always()
      run: |
        echo "Development deployment completed: ${{{{ job.status }}}}"

  deploy-prod:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build]
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Deploy to Kubernetes
      run: |
        echo "Deploying to production environment"
        kubectl apply -f k8s/
      env:
        KUBECONFIG: ${{{{ secrets.KUBECONFIG }}}}
    
    - name: Create release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{{{ secrets.GITHUB_TOKEN }}}}
      with:
        tag_name: ${{{{ github.ref }}}}
        release_name: Release ${{{{ github.ref }}}}
        draft: false
        prerelease: false
"""

    def generate_pr_workflow(self) -> str:
        """Generate pull request workflow"""
        return f"""name: Pull Request Check

on:
  pull_request:
    branches: [ {self.config.main_branch} ]

jobs:
  pr-check:
    name: PR Quality Check
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: {self.config.python_version}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run code checks
      run: |
        black --check --diff .
        flake8 . --statistics
        pytest tests/ -v --cov=scripts --cov=api
    
    - name: Security scan
      run: |
        pip install bandit safety
        bandit -r scripts/ api/
        safety check
    
    - name: Validate Docker build
      run: |
        docker build -t test-build .
"""

    def generate_security_workflow(self) -> str:
        """Generate security scanning workflow"""
        return """name: Security Scan

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM
  workflow_dispatch:

jobs:
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run Trivy scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run dependency check
      run: |
        pip install safety
        safety check --json --output safety-results.json
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: security-results
        path: '*-results.*'
"""

class GitLabCIGenerator:
    """Generate GitLab CI configuration"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
    
    def generate_gitlab_ci(self) -> str:
        """Generate .gitlab-ci.yml"""
        return f"""# GitLab CI/CD Pipeline for StockPredictionPro
stages:
  - test
  - security
  - build
  - deploy

variables:
  PYTHON_VERSION: "{self.config.python_version}"
  DOCKER_DRIVER: overlay2
  IMAGE_NAME: {self.config.organization}/{self.config.image_name}

cache:
  paths:
    - .cache/pip/

# Test stage
test:
  stage: test
  image: python:{self.config.python_version}
  services:
    - postgres:15-alpine
    - redis:7-alpine
  variables:
    POSTGRES_DB: testdb
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
  before_script:
    - python -m pip install --upgrade pip
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
  script:
    - flake8 . --statistics
    - black --check .
    - pytest tests/ -v --cov=scripts --cov=api --cov-report=xml
    - coverage report --fail-under={self.config.test_coverage_threshold}
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
  only:
    - {self.config.main_branch}
    - {self.config.develop_branch}
    - merge_requests

# Security stage
security:
  stage: security
  image: python:{self.config.python_version}
  before_script:
    - pip install bandit safety
  script:
    - bandit -r scripts/ api/ -f json -o bandit-results.json
    - safety check --json --output safety-results.json
  artifacts:
    paths:
      - "*-results.json"
    expire_in: 1 week
  allow_failure: true
  only:
    - {self.config.main_branch}
    - {self.config.develop_branch}

# Build stage
build:
  stage: build
  image: docker:stable
  services:
    - docker:stable-dind
  before_script:
    - docker login -u $REGISTRY_USER -p $REGISTRY_PASSWORD
  script:
    - docker build -t $IMAGE_NAME:$CI_COMMIT_SHA .
    - docker push $IMAGE_NAME:$CI_COMMIT_SHA
    - docker tag $IMAGE_NAME:$CI_COMMIT_SHA $IMAGE_NAME:latest
    - docker push $IMAGE_NAME:latest
  only:
    - {self.config.main_branch}
    - {self.config.develop_branch}
    - tags

# Deploy to development
deploy-dev:
  stage: deploy
  image: alpine/k8s:latest
  script:
    - kubectl apply -f k8s/
    - kubectl set image deployment/stockpredictionpro stockpredictionpro=$IMAGE_NAME:$CI_COMMIT_SHA
  environment:
    name: development
    url: https://dev.stockpredictionpro.com
  only:
    - {self.config.develop_branch}

# Deploy to production
deploy-prod:
  stage: deploy
  image: alpine/k8s:latest
  script:
    - kubectl apply -f k8s/
    - kubectl set image deployment/stockpredictionpro stockpredictionpro=$IMAGE_NAME:$CI_COMMIT_SHA
  environment:
    name: production
    url: https://stockpredictionpro.com
  when: manual
  only:
    - tags
    - {self.config.main_branch}
"""

class JenkinsGenerator:
    """Generate Jenkins pipeline"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
    
    def generate_jenkinsfile(self) -> str:
        """Generate Jenkinsfile"""
        return f"""// Jenkins Pipeline for StockPredictionPro
pipeline {{
    agent any
    
    environment {{
        PYTHON_VERSION = '{self.config.python_version}'
        IMAGE_NAME = '{self.config.organization}/{self.config.image_name}'
    }}
    
    options {{
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 60, unit: 'MINUTES')
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Setup') {{
            steps {{
                sh '''
                    python${{PYTHON_VERSION}} -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pip install -r requirements-dev.txt
                '''
            }}
        }}
        
        stage('Test') {{
            parallel {{
                stage('Unit Tests') {{
                    steps {{
                        sh '''
                            . venv/bin/activate
                            pytest tests/ -v --junitxml=test-results.xml --cov=scripts --cov=api --cov-report=xml
                        '''
                    }}
                    post {{
                        always {{
                            junit 'test-results.xml'
                        }}
                    }}
                }}
                
                stage('Code Quality') {{
                    steps {{
                        sh '''
                            . venv/bin/activate
                            flake8 . --output-file=flake8-results.txt --exit-zero
                            black --check .
                        '''
                    }}
                }}
            }}
        }}
        
        stage('Security') {{
            when {{
                anyOf {{
                    branch '{self.config.main_branch}'
                    branch '{self.config.develop_branch}'
                }}
            }}
            steps {{
                sh '''
                    . venv/bin/activate
                    pip install bandit safety
                    bandit -r scripts/ api/ -f json -o bandit-results.json
                    safety check
                '''
            }}
        }}
        
        stage('Build') {{
            when {{
                anyOf {{
                    branch '{self.config.main_branch}'
                    branch '{self.config.develop_branch}'
                    buildingTag()
                }}
            }}
            steps {{
                script {{
                    def image = docker.build("${{IMAGE_NAME}}:${{BUILD_NUMBER}}")
                    docker.withRegistry('https://index.docker.io/v1/', 'docker-hub-credentials') {{
                        image.push()
                        image.push('latest')
                    }}
                }}
            }}
        }}
        
        stage('Deploy Dev') {{
            when {{
                branch '{self.config.develop_branch}'
            }}
            steps {{
                sh '''
                    kubectl apply -f k8s/
                    kubectl set image deployment/stockpredictionpro stockpredictionpro=${{IMAGE_NAME}}:${{BUILD_NUMBER}}
                '''
            }}
        }}
        
        stage('Deploy Prod') {{
            when {{
                buildingTag()
            }}
            input {{
                message "Deploy to production?"
                ok "Deploy"
            }}
            steps {{
                sh '''
                    kubectl apply -f k8s/
                    kubectl set image deployment/stockpredictionpro stockpredictionpro=${{IMAGE_NAME}}:${{BUILD_NUMBER}}
                '''
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        success {{
            echo "Pipeline completed successfully"
        }}
        failure {{
            echo "Pipeline failed"
        }}
    }}
}}
"""

class AzureDevOpsGenerator:
    """Generate Azure DevOps pipeline"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
    
    def generate_azure_pipeline(self) -> str:
        """Generate azure-pipelines.yml"""
        return f"""# Azure DevOps Pipeline for StockPredictionPro
trigger:
  branches:
    include:
      - {self.config.main_branch}
      - {self.config.develop_branch}
  tags:
    include:
      - v*

variables:
  pythonVersion: '{self.config.python_version}'
  imageName: '{self.config.organization}/{self.config.image_name}'

stages:
- stage: Test
  displayName: 'Test Stage'
  jobs:
  - job: TestJob
    displayName: 'Run Tests'
    pool:
      vmImage: 'ubuntu-latest'
    
    services:
      postgres: postgres:15
      redis: redis:7
    
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
      displayName: 'Install dependencies'
    
    - script: |
        flake8 . --statistics
        black --check .
      displayName: 'Code quality'
    
    - script: |
        pytest tests/ -v --junitxml=test-results.xml --cov=scripts --cov=api --cov-report=xml
      displayName: 'Run tests'
    
    - task: PublishTestResults@2
      inputs:
        testResultsFiles: 'test-results.xml'
        testRunTitle: 'Python $(pythonVersion)'
      condition: succeededOrFailed()

- stage: Security
  displayName: 'Security Stage'
  dependsOn: Test
  jobs:
  - job: SecurityJob
    displayName: 'Security Scan'
    pool:
      vmImage: 'ubuntu-latest'
    
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
    
    - script: |
        pip install bandit safety
        bandit -r scripts/ api/ -f json -o bandit-results.json
        safety check
      displayName: 'Security scan'

- stage: Build
  displayName: 'Build Stage'
  dependsOn: [Test, Security]
  condition: and(succeeded(), ne(variables['Build.Reason'], 'PullRequest'))
  jobs:
  - job: BuildJob
    displayName: 'Build Docker Image'
    pool:
      vmImage: 'ubuntu-latest'
    
    steps:
    - task: Docker@2
      displayName: 'Build and push image'
      inputs:
        command: 'buildAndPush'
        repository: '$(imageName)'
        dockerfile: 'Dockerfile'
        tags: |
          $(Build.BuildId)
          latest

- stage: Deploy
  displayName: 'Deploy Stage'
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/{self.config.main_branch}'))
  jobs:
  - deployment: DeployProduction
    displayName: 'Deploy to Production'
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - script: |
              kubectl apply -f k8s/
              kubectl set image deployment/stockpredictionpro stockpredictionpro=$(imageName):$(Build.BuildId)
            displayName: 'Deploy to Kubernetes'
"""

class CICDSetupOrchestrator:
    """Main orchestrator for CI/CD setup"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
    
    def setup_ci_cd(self) -> CICDSetupResult:
        """Setup CI/CD pipeline"""
        logger.info(f"Setting up CI/CD pipeline for {self.config.platform}")
        
        files_created = []
        secrets_required = []
        manual_steps = []
        
        try:
            if self.config.platform == 'github':
                result = self._setup_github_actions()
            elif self.config.platform == 'gitlab':
                result = self._setup_gitlab_ci()
            elif self.config.platform == 'jenkins':
                result = self._setup_jenkins()
            elif self.config.platform == 'azure':
                result = self._setup_azure_devops()
            else:
                raise ValueError(f"Unsupported platform: {self.config.platform}")
            
            files_created.extend(result['files_created'])
            secrets_required.extend(result['secrets_required'])
            manual_steps.extend(result['manual_steps'])
            
            # Generate common files
            common_files = self._generate_common_files()
            files_created.extend(common_files)
            
            setup_result = CICDSetupResult(
                platform=self.config.platform,
                files_created=files_created,
                secrets_required=secrets_required,
                manual_steps=manual_steps,
                setup_timestamp=datetime.now().isoformat()
            )
            
            self._print_setup_summary(setup_result)
            
            return setup_result
            
        except Exception as e:
            logger.error(f"CI/CD setup failed: {e}")
            raise
    
    def _setup_github_actions(self) -> Dict[str, Any]:
        """Setup GitHub Actions"""
        generator = GitHubActionsGenerator(self.config)
        
        workflows = {
            'ci-cd.yml': generator.generate_main_workflow(),
            'pr-check.yml': generator.generate_pr_workflow(),
            'security.yml': generator.generate_security_workflow()
        }
        
        files_created = []
        
        # Create workflow files
        for filename, content in workflows.items():
            filepath = WORKFLOWS_DIR / filename
            with open(filepath, 'w') as f:
                f.write(content)
            files_created.append(str(filepath))
        
        # Create dependabot config
        dependabot_config = self._generate_dependabot_config()
        dependabot_path = GITHUB_DIR / 'dependabot.yml'
        with open(dependabot_path, 'w') as f:
            f.write(dependabot_config)
        files_created.append(str(dependabot_path))
        
        secrets_required = [
            'REGISTRY_USERNAME',
            'REGISTRY_PASSWORD',
            'KUBECONFIG'
        ]
        
        manual_steps = [
            'Add repository secrets in GitHub Settings > Secrets',
            'Configure branch protection rules',
            'Set up environments with protection rules',
            'Enable GitHub Pages for documentation'
        ]
        
        return {
            'files_created': files_created,
            'secrets_required': secrets_required,
            'manual_steps': manual_steps
        }
    
    def _setup_gitlab_ci(self) -> Dict[str, Any]:
        """Setup GitLab CI"""
        generator = GitLabCIGenerator(self.config)
        
        gitlab_ci_content = generator.generate_gitlab_ci()
        
        with open(GITLAB_CI_FILE, 'w') as f:
            f.write(gitlab_ci_content)
        
        return {
            'files_created': [str(GITLAB_CI_FILE)],
            'secrets_required': ['REGISTRY_USER', 'REGISTRY_PASSWORD', 'KUBECONFIG'],
            'manual_steps': [
                'Add CI/CD variables in GitLab project settings',
                'Configure container registry',
                'Set up deployment environments'
            ]
        }
    
    def _setup_jenkins(self) -> Dict[str, Any]:
        """Setup Jenkins"""
        generator = JenkinsGenerator(self.config)
        
        jenkinsfile_content = generator.generate_jenkinsfile()
        
        with open(JENKINS_FILE, 'w') as f:
            f.write(jenkinsfile_content)
        
        return {
            'files_created': [str(JENKINS_FILE)],
            'secrets_required': ['docker-hub-credentials', 'kubeconfig'],
            'manual_steps': [
                'Install Jenkins plugins',
                'Configure credentials',
                'Set up webhook',
                'Install Python on agents'
            ]
        }
    
    def _setup_azure_devops(self) -> Dict[str, Any]:
        """Setup Azure DevOps"""
        generator = AzureDevOpsGenerator(self.config)
        
        azure_pipeline_content = generator.generate_azure_pipeline()
        
        with open(AZURE_FILE, 'w') as f:
            f.write(azure_pipeline_content)
        
        return {
            'files_created': [str(AZURE_FILE)],
            'secrets_required': ['dockerRegistryServiceConnection'],
            'manual_steps': [
                'Create service connections',
                'Configure pipeline in Azure DevOps',
                'Set up variable groups',
                'Configure approvals'
            ]
        }
    
    def _generate_common_files(self) -> List[str]:
        """Generate common configuration files"""
        files_created = []
        
        # Pre-commit configuration
        precommit_config = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
"""
        precommit_path = PROJECT_ROOT / '.pre-commit-config.yaml'
        with open(precommit_path, 'w') as f:
            f.write(precommit_config)
        files_created.append(str(precommit_path))
        
        # pytest configuration
        pytest_config = """[tool:pytest]
minversion = 6.0
addopts = -ra -q
testpaths = tests
python_files = test_*.py
python_functions = test_*
markers =
    slow: marks tests as slow
    integration: integration tests
    unit: unit tests
"""
        pytest_path = PROJECT_ROOT / 'pytest.ini'
        with open(pytest_path, 'w') as f:
            f.write(pytest_config)
        files_created.append(str(pytest_path))
        
        return files_created
    
    def _generate_dependabot_config(self) -> str:
        """Generate Dependabot configuration"""
        return """version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
  
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
  
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
"""
    
    def _print_setup_summary(self, result: CICDSetupResult) -> None:
        """Print setup summary"""
        print("\n" + "="*60)
        print("CI/CD SETUP SUMMARY")
        print("="*60)
        print(f"Platform: {result.platform.upper()}")
        print(f"Files Created: {len(result.files_created)}")
        
        print(f"\nFiles Created:")
        for file_path in result.files_created:
            print(f"  ✅ {file_path}")
        
        if result.secrets_required:
            print(f"\nSecrets Required:")
            for secret in result.secrets_required:
                print(f"  🔐 {secret}")
        
        if result.manual_steps:
            print(f"\nManual Steps:")
            for i, step in enumerate(result.manual_steps, 1):
                print(f"  {i}. {step}")
        
        print(f"\n🎉 CI/CD setup completed!")

def main():
    """Main CLI function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup CI/CD pipeline')
    parser.add_argument('--platform', choices=['github', 'gitlab', 'jenkins', 'azure'],
                       default='github', help='CI/CD platform')
    parser.add_argument('--organization', default='stockpredictionpro', help='Organization')
    parser.add_argument('--image-name', default='stockpredictionpro', help='Image name')
    parser.add_argument('--python-version', default='3.13', help='Python version')
    parser.add_argument('--main-branch', default='main', help='Main branch name')
    parser.add_argument('--develop-branch', default='develop', help='Development branch')
    
    args = parser.parse_args()
    
    try:
        config = CICDConfig(
            platform=args.platform,
            organization=args.organization,
            image_name=args.image_name,
            python_version=args.python_version,
            main_branch=args.main_branch,
            develop_branch=args.develop_branch
        )
        
        orchestrator = CICDSetupOrchestrator(config)
        result = orchestrator.setup_ci_cd()
        
        print(f"\n✅ Setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
