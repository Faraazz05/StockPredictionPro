# ============================================
# StockPredictionPro - Makefile
# Automation for development, testing, and deployment
# ============================================

# Variables
PYTHON := python
PIP := pip
VENV := .venv
DOCKER_IMAGE := stockpredictionpro
DOCKER_TAG := latest
APP_NAME := StockPredictionPro

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

.PHONY: help install dev-install test lint format clean docker-build docker-run docker-clean build-exe setup-env check-env all

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(BLUE)StockPredictionPro - Available Commands$(RESET)"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
setup-env: ## Setup development environment
	@echo "$(YELLOW)Setting up development environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created at $(VENV)$(RESET)"
	@echo "$(YELLOW)Activate with: source $(VENV)/bin/activate (Linux/Mac) or $(VENV)\\Scripts\\activate (Windows)$(RESET)"

check-env: ## Check if virtual environment is activated
	@echo "$(YELLOW)Checking environment...$(RESET)"
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(RED)Error: Virtual environment not activated!$(RESET)"; \
		echo "$(YELLOW)Run: source $(VENV)/bin/activate$(RESET)"; \
		exit 1; \
	else \
		echo "$(GREEN)Virtual environment active: $$VIRTUAL_ENV$(RESET)"; \
	fi

# Installation
install: check-env ## Install production dependencies
	@echo "$(YELLOW)Installing production dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Production dependencies installed!$(RESET)"

dev-install: check-env ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(RESET)"

# Code Quality
lint: check-env ## Run linting (flake8, mypy)
	@echo "$(YELLOW)Running linting...$(RESET)"
	flake8 src/ app/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ app/ tests/ --count --max-complexity=10 --max-line-length=88 --statistics
	mypy src/ app/ --ignore-missing-imports
	yamllint config/ -d relaxed
	@echo "$(GREEN)Linting passed!$(RESET)"

format: check-env ## Format code (black, isort)
	@echo "$(YELLOW)Formatting code...$(RESET)"
	black src/ app/ tests/ scripts/
	isort src/ app/ tests/ scripts/ --profile black
	@echo "$(GREEN)Code formatted!$(RESET)"

format-check: check-env ## Check code formatting
	@echo "$(YELLOW)Checking code format...$(RESET)"
	black --check src/ app/ tests/ scripts/
	isort --check-only src/ app/ tests/ scripts/ --profile black
	@echo "$(GREEN)Code format is correct!$(RESET)"

# Testing
test: check-env ## Run all tests
	@echo "$(YELLOW)Running tests...$(RESET)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Tests completed! Coverage report in htmlcov/$(RESET)"

test-unit: check-env ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(RESET)"
	pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(RESET)"

test-integration: check-env ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(RESET)"
	pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(RESET)"

test-performance: check-env ## Run performance tests
	@echo "$(YELLOW)Running performance tests...$(RESET)"
	pytest tests/performance/ -v
	@echo "$(GREEN)Performance tests completed!$(RESET)"

test-watch: check-env ## Run tests in watch mode
	@echo "$(YELLOW)Running tests in watch mode...$(RESET)"
	pytest-watch tests/ -- -v

# Data and Models
fetch-data: check-env ## Download sample data
	@echo "$(YELLOW)Fetching sample data...$(RESET)"
	$(PYTHON) scripts/data/download_data.py --symbols AAPL,MSFT,INFY.NS --days 365
	@echo "$(GREEN)Sample data downloaded!$(RESET)"

train-models: check-env ## Train all models
	@echo "$(YELLOW)Training models...$(RESET)"
	$(PYTHON) scripts/models/train_all_models.py
	@echo "$(GREEN)Models trained and saved!$(RESET)"

evaluate-models: check-env ## Evaluate trained models
	@echo "$(YELLOW)Evaluating models...$(RESET)"
	$(PYTHON) scripts/models/evaluate_models.py
	@echo "$(GREEN)Model evaluation completed!$(RESET)"

# Application
run: check-env ## Run Streamlit app locally
	@echo "$(YELLOW)Starting Streamlit application...$(RESET)"
	streamlit run app/streamlit_app.py --server.port 8501

run-api: check-env ## Run FastAPI server
	@echo "$(YELLOW)Starting FastAPI server...$(RESET)"
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dev: check-env ## Run in development mode with hot reload
	@echo "$(YELLOW)Starting development server...$(RESET)"
	streamlit run app/streamlit_app.py --server.port 8501 --server.runOnSave true

# Docker
docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(RESET)"

docker-run: ## Run Docker container
	@echo "$(YELLOW)Running Docker container...$(RESET)"
	docker run -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/logs:/app/logs $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start services with docker-compose
	@echo "$(YELLOW)Starting services with docker-compose...$(RESET)"
	docker-compose up --build

docker-compose-down: ## Stop docker-compose services
	@echo "$(YELLOW)Stopping docker-compose services...$(RESET)"
	docker-compose down

docker-clean: ## Clean Docker images and containers
	@echo "$(YELLOW)Cleaning Docker resources...$(RESET)"
	docker container prune -f
	docker image prune -f
	docker volume prune -f
	@echo "$(GREEN)Docker cleanup completed!$(RESET)"

# Build and Package
build-exe: check-env ## Build standalone executable
	@echo "$(YELLOW)Building standalone executable...$(RESET)"
	$(PYTHON) build_exe.py
	@echo "$(GREEN)Executable built in dist/ directory!$(RESET)"

build-wheel: check-env ## Build Python wheel
	@echo "$(YELLOW)Building Python wheel...$(RESET)"
	$(PYTHON) setup.py bdist_wheel
	@echo "$(GREEN)Wheel built in dist/ directory!$(RESET)"

# Deployment
deploy-local: docker-build docker-run ## Build and run locally with Docker

deploy-staging: ## Deploy to staging environment
	@echo "$(YELLOW)Deploying to staging...$(RESET)"
	@echo "$(RED)Staging deployment not configured yet$(RESET)"

deploy-prod: ## Deploy to production
	@echo "$(YELLOW)Deploying to production...$(RESET)"
	@echo "$(RED)Production deployment not configured yet$(RESET)"

# Cleanup
clean: ## Clean temporary files and caches
	@echo "$(YELLOW)Cleaning temporary files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf logs/*.log logs/audit/runs/* logs/audit/models/*
	@echo "$(GREEN)Cleanup completed!$(RESET)"

clean-data: ## Clean data files (keeps structure)
	@echo "$(YELLOW)Cleaning data files...$(RESET)"
	find data/raw/ -name "*.csv" -delete 2>/dev/null || true
	find data/processed/ -name "*.parquet" -delete 2>/dev/null || true
	find data/models/ -name "*.joblib" -delete 2>/dev/null || true
	find data/predictions/ -name "*.csv" -delete 2>/dev/null || true
	find data/backtests/ -name "*.csv" -delete 2>/dev/null || true
	@echo "$(GREEN)Data cleanup completed!$(RESET)"

# Development Workflow
all: format lint test ## Run complete development workflow (format, lint, test)
	@echo "$(GREEN)All development checks passed!$(RESET)"

ci: format-check lint test ## Run CI pipeline checks
	@echo "$(GREEN)CI pipeline checks passed!$(RESET)"

quick-check: format lint test-unit ## Quick development check (no integration tests)
	@echo "$(GREEN)Quick checks passed!$(RESET)"

# Git hooks
pre-commit: format-check lint test-unit ## Pre-commit hook (fast checks)
	@echo "$(GREEN)Pre-commit checks passed!$(RESET)"

# Utilities
show-config: ## Show current configuration
	@echo "$(BLUE)Current Configuration:$(RESET)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Virtual Env: $$VIRTUAL_ENV"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Git: $(shell git --version)"

install-system-deps: ## Install system dependencies (Ubuntu/Debian)
	@echo "$(YELLOW)Installing system dependencies...$(RESET)"
	sudo apt-get update
	sudo apt-get install -y python3-dev build-essential curl git
	@echo "$(GREEN)System dependencies installed!$(RESET)"

# Documentation
docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(RESET)"
	$(PYTHON) scripts/utilities/generate_docs.py
	@echo "$(GREEN)Documentation generated!$(RESET)"

# Backup
backup: ## Backup important files
	@echo "$(YELLOW)Creating backup...$(RESET)"
	$(PYTHON) scripts/utilities/backup_data.py
	@echo "$(GREEN)Backup created!$(RESET)"
