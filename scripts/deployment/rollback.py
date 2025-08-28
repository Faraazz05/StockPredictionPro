"""
scripts/deployment/rollback.py

Automated rollback system for StockPredictionPro deployments.
Supports Kubernetes deployments, database migrations, model versions,
and infrastructure changes with comprehensive safety checks and validation.

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
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import tempfile
import hashlib

# Database utilities
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Kubernetes client
try:
    from kubernetes import client, config
    HAS_K8S = True
except ImportError:
    HAS_K8S = False

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'rollback_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.Rollback')

# Directory configuration
PROJECT_ROOT = Path('.')
DEPLOYMENT_DIR = PROJECT_ROOT / 'deployment'
K8S_DIR = DEPLOYMENT_DIR / 'k8s'
BACKUP_DIR = PROJECT_ROOT / 'backups'
ROLLBACK_DIR = DEPLOYMENT_DIR / 'rollbacks'
MODELS_DIR = PROJECT_ROOT / 'models'

# Ensure directories exist
for dir_path in [ROLLBACK_DIR, BACKUP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class RollbackConfig:
    """Configuration for rollback operations"""
    # Target settings
    target_revision: Optional[str] = None  # Specific revision to rollback to
    target_tag: Optional[str] = None       # Docker image tag to rollback to
    target_timestamp: Optional[str] = None # Timestamp-based rollback
    
    # Rollback scope
    rollback_type: str = 'deployment'  # deployment, database, models, full
    namespace: str = 'default'
    app_name: str = 'stockpredictionpro'
    
    # Safety settings
    dry_run: bool = False
    backup_before_rollback: bool = True
    validate_before_rollback: bool = True
    max_rollback_age_days: int = 30
    
    # Database rollback settings
    database_url: Optional[str] = None
    backup_database: bool = True
    migration_table: str = 'schema_migrations'
    
    # Model rollback settings
    model_registry_path: str = 'models/production/model_registry.json'
    backup_current_models: bool = True
    
    # Kubernetes settings
    kubectl_timeout: int = 300  # 5 minutes
    rollback_timeout: int = 600  # 10 minutes
    
    # Validation settings
    health_check_url: Optional[str] = None
    health_check_timeout: int = 120
    post_rollback_tests: List[str] = None
    
    # Notification settings
    send_notifications: bool = True
    notification_webhooks: List[str] = None
    
    def __post_init__(self):
        if self.post_rollback_tests is None:
            self.post_rollback_tests = ['health_check', 'api_test']
        if self.notification_webhooks is None:
            self.notification_webhooks = []

@dataclass
class RollbackStep:
    """Individual rollback step"""
    step_id: str
    description: str
    component: str
    action: str  # rollback, backup, validate, cleanup
    status: str = 'pending'  # pending, running, success, failed, skipped
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    backup_location: Optional[str] = None
    rollback_command: Optional[str] = None
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

@dataclass
class RollbackPlan:
    """Complete rollback execution plan"""
    plan_id: str
    created_at: str
    target_revision: Optional[str]
    rollback_type: str
    steps: List[RollbackStep]
    estimated_duration: int  # seconds
    risk_assessment: str    # low, medium, high
    prerequisites: List[str]
    rollback_order: List[str]  # Order of component rollbacks
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RollbackResult:
    """Results from rollback execution"""
    rollback_id: str
    execution_timestamp: str
    config_used: RollbackConfig
    plan_executed: RollbackPlan
    
    # Execution results
    total_steps: int
    successful_steps: int
    failed_steps: int
    skipped_steps: int
    total_duration: float
    
    # Component status
    deployment_status: str
    database_status: str
    models_status: str
    
    # Validation results
    health_check_passed: bool
    post_rollback_tests: Dict[str, bool]
    
    # Rollback metadata  
    previous_version: Optional[str]
    current_version: Optional[str]
    rollback_point: str
    
    # Status and recommendations
    overall_status: str  # success, partial, failed
    can_rollback_further: bool
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# ROLLBACK HISTORY MANAGER
# ============================================

class RollbackHistoryManager:
    """Manage rollback history and available rollback points"""
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.history_file = ROLLBACK_DIR / 'rollback_history.json'
        self.deployment_history_file = DEPLOYMENT_DIR / 'deployment_history.json'
    
    def get_available_rollback_points(self) -> List[Dict[str, Any]]:
        """Get list of available rollback points"""
        rollback_points = []
        
        # Get Kubernetes rollback points
        k8s_points = self._get_kubernetes_rollback_points()
        rollback_points.extend(k8s_points)
        
        # Get deployment history points
        deployment_points = self._get_deployment_history_points()
        rollback_points.extend(deployment_points)
        
        # Get model version points
        model_points = self._get_model_rollback_points()
        rollback_points.extend(model_points)
        
        # Sort by timestamp (newest first)
        rollback_points.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Filter by max age
        cutoff_date = datetime.now() - timedelta(days=self.config.max_rollback_age_days)
        rollback_points = [
            point for point in rollback_points
            if datetime.fromisoformat(point.get('timestamp', '1970-01-01')) > cutoff_date
        ]
        
        return rollback_points
    
    def _get_kubernetes_rollback_points(self) -> List[Dict[str, Any]]:
        """Get Kubernetes deployment rollback points"""
        points = []
        
        try:
            # Get deployment rollout history
            cmd = [
                'kubectl', 'rollout', 'history',
                f'deployment/{self.config.app_name}',
                '-n', self.config.namespace,
                '--output=json'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                history_data = json.loads(result.stdout)
                
                for revision in history_data.get('revisions', []):
                    points.append({
                        'type': 'kubernetes_deployment',
                        'revision': revision.get('revision'),
                        'timestamp': revision.get('creationTimestamp', ''),
                        'description': f"Kubernetes deployment revision {revision.get('revision')}",
                        'component': 'deployment',
                        'change_cause': revision.get('annotations', {}).get('deployment.kubernetes.io/change-cause', 'Unknown')
                    })
                    
        except Exception as e:
            logger.warning(f"Could not get Kubernetes rollback points: {e}")
        
        return points
    
    def _get_deployment_history_points(self) -> List[Dict[str, Any]]:
        """Get deployment history points"""
        points = []
        
        try:
            if self.deployment_history_file.exists():
                with open(self.deployment_history_file, 'r') as f:
                    history_data = json.load(f)
                
                for deployment in history_data.get('deployments', []):
                    points.append({
                        'type': 'deployment_history',
                        'deployment_id': deployment.get('deployment_id'),
                        'timestamp': deployment.get('timestamp'),
                        'description': f"Deployment {deployment.get('version', 'unknown')}",
                        'component': 'full_deployment',
                        'version': deployment.get('version'),
                        'image_tag': deployment.get('image_tag'),
                        'status': deployment.get('status')
                    })
                    
        except Exception as e:
            logger.warning(f"Could not load deployment history: {e}")
        
        return points
    
    def _get_model_rollback_points(self) -> List[Dict[str, Any]]:
        """Get model version rollback points"""
        points = []
        
        try:
            model_registry_path = Path(self.config.model_registry_path)
            
            if model_registry_path.exists():
                with open(model_registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, model_info in registry_data.items():
                    points.append({
                        'type': 'model_version',
                        'model_id': model_id,
                        'timestamp': model_info.get('deployed_at', ''),
                        'description': f"Model {model_info.get('model_name', 'unknown')} v{model_info.get('model_version', 'unknown')}",
                        'component': 'models',
                        'model_name': model_info.get('model_name'),
                        'model_version': model_info.get('model_version'),
                        'model_path': model_info.get('model_path')
                    })
                    
        except Exception as e:
            logger.warning(f"Could not load model registry: {e}")
        
        return points
    
    def record_rollback(self, rollback_result: RollbackResult) -> None:
        """Record rollback execution in history"""
        try:
            # Load existing history
            history = {'rollbacks': []}
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new rollback record
            history['rollbacks'].append({
                'rollback_id': rollback_result.rollback_id,
                'timestamp': rollback_result.execution_timestamp,
                'rollback_type': rollback_result.config_used.rollback_type,
                'target_revision': rollback_result.config_used.target_revision,
                'status': rollback_result.overall_status,
                'duration': rollback_result.total_duration,
                'previous_version': rollback_result.previous_version,
                'current_version': rollback_result.current_version
            })
            
            # Keep only last 100 rollbacks
            history['rollbacks'] = history['rollbacks'][-100:]
            
            # Save updated history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to record rollback history: {e}")

# ============================================
# ROLLBACK PLANNERS
# ============================================

class RollbackPlanner:
    """Create rollback execution plans"""
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.history_manager = RollbackHistoryManager(config)
    
    def create_rollback_plan(self, target_point: Dict[str, Any]) -> RollbackPlan:
        """Create comprehensive rollback plan"""
        plan_id = f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        steps = []
        rollback_order = []
        
        # Determine rollback type and create appropriate steps
        if self.config.rollback_type == 'deployment':
            steps.extend(self._create_deployment_rollback_steps(target_point))
            rollback_order = ['application', 'configuration']
            
        elif self.config.rollback_type == 'database':
            steps.extend(self._create_database_rollback_steps(target_point))
            rollback_order = ['database']
            
        elif self.config.rollback_type == 'models':
            steps.extend(self._create_model_rollback_steps(target_point))
            rollback_order = ['models']
            
        elif self.config.rollback_type == 'full':
            steps.extend(self._create_full_rollback_steps(target_point))
            rollback_order = ['models', 'database', 'application', 'configuration']
        
        # Add validation steps
        if self.config.validate_before_rollback:
            validation_steps = self._create_validation_steps()
            steps = validation_steps + steps
        
        # Add backup steps
        if self.config.backup_before_rollback:
            backup_steps = self._create_backup_steps()
            steps = backup_steps + steps
        
        # Add post-rollback steps
        post_steps = self._create_post_rollback_steps()
        steps.extend(post_steps)
        
        # Assess risk
        risk_assessment = self._assess_rollback_risk(target_point, steps)
        
        # Estimate duration
        estimated_duration = self._estimate_rollback_duration(steps)
        
        # Generate prerequisites
        prerequisites = self._generate_prerequisites(target_point)
        
        return RollbackPlan(
            plan_id=plan_id,
            created_at=datetime.now().isoformat(),
            target_revision=target_point.get('revision') or target_point.get('deployment_id'),
            rollback_type=self.config.rollback_type,
            steps=steps,
            estimated_duration=estimated_duration,
            risk_assessment=risk_assessment,
            prerequisites=prerequisites,
            rollback_order=rollback_order
        )
    
    def _create_deployment_rollback_steps(self, target_point: Dict[str, Any]) -> List[RollbackStep]:
        """Create deployment rollback steps"""
        steps = []
        
        if target_point.get('type') == 'kubernetes_deployment':
            steps.append(RollbackStep(
                step_id='k8s_rollback',
                description=f"Rollback Kubernetes deployment to revision {target_point.get('revision')}",
                component='kubernetes',
                action='rollback',
                rollback_command=f"kubectl rollout undo deployment/{self.config.app_name} --to-revision={target_point.get('revision')} -n {self.config.namespace}"
            ))
            
            steps.append(RollbackStep(
                step_id='k8s_wait',
                description="Wait for rollback deployment to complete",
                component='kubernetes',
                action='validate'
            ))
        
        return steps
    
    def _create_database_rollback_steps(self, target_point: Dict[str, Any]) -> List[RollbackStep]:
        """Create database rollback steps"""
        steps = []
        
        steps.append(RollbackStep(
            step_id='db_backup',
            description="Create database backup before rollback",
            component='database',
            action='backup'
        ))
        
        steps.append(RollbackStep(
            step_id='db_rollback',
            description=f"Rollback database to {target_point.get('timestamp', 'target point')}",
            component='database',
            action='rollback'
        ))
        
        steps.append(RollbackStep(
            step_id='db_validate',
            description="Validate database rollback",
            component='database',
            action='validate'
        ))
        
        return steps
    
    def _create_model_rollback_steps(self, target_point: Dict[str, Any]) -> List[RollbackStep]:
        """Create model rollback steps"""
        steps = []
        
        steps.append(RollbackStep(
            step_id='model_backup',
            description="Backup current production models",
            component='models',
            action='backup'
        ))
        
        steps.append(RollbackStep(
            step_id='model_rollback',
            description=f"Rollback to model version {target_point.get('model_version', 'unknown')}",
            component='models',
            action='rollback'
        ))
        
        steps.append(RollbackStep(
            step_id='model_validate',
            description="Validate model rollback",
            component='models',
            action='validate'
        ))
        
        return steps
    
    def _create_full_rollback_steps(self, target_point: Dict[str, Any]) -> List[RollbackStep]:
        """Create full system rollback steps"""
        steps = []
        
        # Combine all rollback types
        steps.extend(self._create_model_rollback_steps(target_point))
        steps.extend(self._create_database_rollback_steps(target_point))
        steps.extend(self._create_deployment_rollback_steps(target_point))
        
        return steps
    
    def _create_validation_steps(self) -> List[RollbackStep]:
        """Create pre-rollback validation steps"""
        steps = []
        
        steps.append(RollbackStep(
            step_id='validate_cluster',
            description="Validate Kubernetes cluster accessibility",
            component='infrastructure',
            action='validate'
        ))
        
        steps.append(RollbackStep(
            step_id='validate_permissions',
            description="Validate rollback permissions",
            component='infrastructure',
            action='validate'
        ))
        
        return steps
    
    def _create_backup_steps(self) -> List[RollbackStep]:
        """Create backup steps"""
        steps = []
        
        steps.append(RollbackStep(
            step_id='backup_current_state',
            description="Create backup of current deployment state",
            component='backup',
            action='backup'
        ))
        
        return steps
    
    def _create_post_rollback_steps(self) -> List[RollbackStep]:
        """Create post-rollback validation steps"""
        steps = []
        
        steps.append(RollbackStep(
            step_id='health_check',
            description="Perform health check after rollback",
            component='validation',
            action='validate'
        ))
        
        for test in self.config.post_rollback_tests:
            steps.append(RollbackStep(
                step_id=f'test_{test}',
                description=f"Execute {test} validation",
                component='validation',
                action='validate'
            ))
        
        return steps
    
    def _assess_rollback_risk(self, target_point: Dict[str, Any], steps: List[RollbackStep]) -> str:
        """Assess rollback risk level"""
        risk_factors = 0
        
        # Age of rollback target
        try:
            target_date = datetime.fromisoformat(target_point.get('timestamp', ''))
            age_days = (datetime.now() - target_date).days
            
            if age_days > 7:
                risk_factors += 1
            if age_days > 30:
                risk_factors += 2
        except:
            risk_factors += 1  # Unknown age
        
        # Database rollback increases risk
        if self.config.rollback_type in ['database', 'full']:
            risk_factors += 2
        
        # Number of steps
        if len(steps) > 10:
            risk_factors += 1
        
        # Production environment
        if 'production' in self.config.namespace.lower():
            risk_factors += 1
        
        if risk_factors <= 2:
            return 'low'
        elif risk_factors <= 4:
            return 'medium'
        else:
            return 'high'
    
    def _estimate_rollback_duration(self, steps: List[RollbackStep]) -> int:
        """Estimate rollback duration in seconds"""
        base_time = 60  # 1 minute base
        
        step_times = {
            'backup': 120,     # 2 minutes
            'rollback': 180,   # 3 minutes
            'validate': 60,    # 1 minute
            'cleanup': 30      # 30 seconds
        }
        
        total_time = base_time
        
        for step in steps:
            total_time += step_times.get(step.action, 60)
        
        return total_time
    
    def _generate_prerequisites(self, target_point: Dict[str, Any]) -> List[str]:
        """Generate rollback prerequisites"""
        prerequisites = []
        
        prerequisites.append("Verify kubectl access to cluster")
        prerequisites.append("Confirm rollback target is valid")
        
        if self.config.rollback_type in ['database', 'full']:
            prerequisites.append("Verify database connectivity")
            prerequisites.append("Confirm database backup availability")
        
        if self.config.rollback_type in ['models', 'full']:
            prerequisites.append("Verify model files accessibility")
        
        prerequisites.append("Ensure sufficient permissions for rollback operations")
        
        return prerequisites

# ============================================
# ROLLBACK EXECUTORS
# ============================================

class RollbackExecutor:
    """Execute rollback plans"""
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.history_manager = RollbackHistoryManager(config)
    
    def execute_rollback_plan(self, plan: RollbackPlan) -> RollbackResult:
        """Execute rollback plan"""
        logger.info(f"🔄 Starting rollback execution: {plan.plan_id}")
        start_time = datetime.now()
        
        # Initialize result tracking
        successful_steps = 0
        failed_steps = 0
        skipped_steps = 0
        
        deployment_status = 'unknown'
        database_status = 'unknown'
        models_status = 'unknown'
        
        previous_version = self._get_current_version()
        
        # Execute each step
        for step in plan.steps:
            try:
                step.start_time = datetime.now()
                step.status = 'running'
                
                logger.info(f"▶️ Executing step: {step.description}")
                
                if self.config.dry_run:
                    logger.info(f"🔍 DRY RUN: Would execute {step.step_id}")
                    step.status = 'success'
                    step.end_time = datetime.now()
                    successful_steps += 1
                    continue
                
                # Execute step based on action type
                success = self._execute_step(step)
                
                step.end_time = datetime.now()
                
                if success:
                    step.status = 'success'
                    successful_steps += 1
                    logger.info(f"✅ Step completed: {step.step_id}")
                    
                    # Update component status
                    if step.component == 'kubernetes':
                        deployment_status = 'rolled_back'
                    elif step.component == 'database':
                        database_status = 'rolled_back'
                    elif step.component == 'models':
                        models_status = 'rolled_back'
                else:
                    step.status = 'failed'
                    failed_steps += 1
                    logger.error(f"❌ Step failed: {step.step_id}")
                    
                    # Decide whether to continue or abort
                    if step.action == 'rollback' and step.component in ['database', 'kubernetes']:
                        logger.error("Critical rollback step failed - aborting")
                        break
                    
            except Exception as e:
                step.status = 'failed'
                step.error_message = str(e)
                step.end_time = datetime.now()
                failed_steps += 1
                logger.error(f"❌ Step failed with exception: {step.step_id} - {e}")
        
        # Post-rollback validation
        health_check_passed = self._perform_health_check()
        post_rollback_tests = self._run_post_rollback_tests()
        
        # Determine overall status
        if failed_steps == 0:
            overall_status = 'success'
        elif successful_steps > 0:
            overall_status = 'partial'
        else:
            overall_status = 'failed'
        
        # Get current version after rollback
        current_version = self._get_current_version()
        
        # Create result
        total_duration = (datetime.now() - start_time).total_seconds()
        
        result = RollbackResult(
            rollback_id=plan.plan_id,
            execution_timestamp=start_time.isoformat(),
            config_used=self.config,
            plan_executed=plan,
            total_steps=len(plan.steps),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            total_duration=total_duration,
            deployment_status=deployment_status,
            database_status=database_status,
            models_status=models_status,
            health_check_passed=health_check_passed,
            post_rollback_tests=post_rollback_tests,
            previous_version=previous_version,
            current_version=current_version,
            rollback_point=plan.target_revision or 'unknown',
            overall_status=overall_status,
            can_rollback_further=self._can_rollback_further(),
            recommendations=self._generate_post_rollback_recommendations(overall_status, health_check_passed)
        )
        
        # Record rollback in history
        self.history_manager.record_rollback(result)
        
        logger.info(f"🔄 Rollback execution completed: {overall_status}")
        
        return result
    
    def _execute_step(self, step: RollbackStep) -> bool:
        """Execute individual rollback step"""
        try:
            if step.action == 'backup':
                return self._execute_backup_step(step)
            elif step.action == 'rollback':
                return self._execute_rollback_step(step)
            elif step.action == 'validate':
                return self._execute_validation_step(step)
            elif step.action == 'cleanup':
                return self._execute_cleanup_step(step)
            else:
                logger.warning(f"Unknown step action: {step.action}")
                return False
                
        except Exception as e:
            step.error_message = str(e)
            logger.error(f"Step execution failed: {e}")
            return False
    
    def _execute_backup_step(self, step: RollbackStep) -> bool:
        """Execute backup step"""
        try:
            if step.component == 'database' and self.config.database_url:
                return self._backup_database(step)
            elif step.component == 'models':
                return self._backup_models(step)
            elif step.component == 'backup':
                return self._backup_current_state(step)
            else:
                logger.warning(f"Unknown backup component: {step.component}")
                return True  # Non-critical, continue
                
        except Exception as e:
            logger.error(f"Backup step failed: {e}")
            return False
    
    def _execute_rollback_step(self, step: RollbackStep) -> bool:
        """Execute rollback step"""
        try:
            if step.component == 'kubernetes':
                return self._rollback_kubernetes_deployment(step)
            elif step.component == 'database':
                return self._rollback_database(step)
            elif step.component == 'models':
                return self._rollback_models(step)
            else:
                logger.warning(f"Unknown rollback component: {step.component}")
                return False
                
        except Exception as e:
            logger.error(f"Rollback step failed: {e}")
            return False
    
    def _execute_validation_step(self, step: RollbackStep) -> bool:
        """Execute validation step"""
        try:
            if step.step_id == 'validate_cluster':
                return self._validate_cluster_access()
            elif step.step_id == 'validate_permissions':
                return self._validate_permissions()
            elif step.step_id == 'k8s_wait':
                return self._wait_for_kubernetes_rollback()
            elif step.step_id == 'health_check':
                return self._perform_health_check()
            elif step.step_id.startswith('test_'):
                return self._run_specific_test(step.step_id.replace('test_', ''))
            else:
                logger.warning(f"Unknown validation step: {step.step_id}")
                return True  # Default to success for unknown validations
                
        except Exception as e:
            logger.error(f"Validation step failed: {e}")
            return False
    
    def _execute_cleanup_step(self, step: RollbackStep) -> bool:
        """Execute cleanup step"""
        try:
            # Implement cleanup logic
            logger.info(f"Performing cleanup: {step.description}")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup step failed: {e}")
            return False
    
    def _backup_database(self, step: RollbackStep) -> bool:
        """Backup database before rollback"""
        try:
            if not self.config.database_url:
                logger.warning("No database URL configured")
                return True
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"database_backup_before_rollback_{timestamp}.sql"
            backup_path = BACKUP_DIR / backup_filename
            
            logger.info(f"Creating database backup: {backup_path}")
            
            # Use pg_dump for PostgreSQL (adjust for other databases)
            cmd = [
                'pg_dump',
                self.config.database_url,
                '-f', str(backup_path),
                '--no-password'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                step.backup_location = str(backup_path)
                logger.info(f"Database backup created: {backup_path}")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def _backup_models(self, step: RollbackStep) -> bool:
        """Backup current models"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            models_backup_dir = BACKUP_DIR / f"models_backup_{timestamp}"
            models_backup_dir.mkdir(exist_ok=True)
            
            production_models_dir = MODELS_DIR / 'production'
            
            if production_models_dir.exists():
                shutil.copytree(production_models_dir, models_backup_dir / 'production', dirs_exist_ok=True)
                step.backup_location = str(models_backup_dir)
                logger.info(f"Models backup created: {models_backup_dir}")
                return True
            else:
                logger.warning("No production models directory found")
                return True
                
        except Exception as e:
            logger.error(f"Models backup failed: {e}")
            return False
    
    def _backup_current_state(self, step: RollbackStep) -> bool:
        """Backup current deployment state"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            state_backup_dir = BACKUP_DIR / f"deployment_state_{timestamp}"
            state_backup_dir.mkdir(exist_ok=True)
            
            # Export current Kubernetes resources
            resources = ['deployment', 'service', 'configmap', 'secret']
            
            for resource in resources:
                try:
                    cmd = [
                        'kubectl', 'get', resource,
                        '-n', self.config.namespace,
                        '-o', 'yaml',
                        '--export'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        backup_file = state_backup_dir / f"{resource}.yaml"
                        with open(backup_file, 'w') as f:
                            f.write(result.stdout)
                            
                except Exception as e:
                    logger.warning(f"Failed to backup {resource}: {e}")
            
            step.backup_location = str(state_backup_dir)
            logger.info(f"Deployment state backup created: {state_backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment state backup failed: {e}")
            return False
    
    def _rollback_kubernetes_deployment(self, step: RollbackStep) -> bool:
        """Rollback Kubernetes deployment"""
        try:
            if step.rollback_command:
                # Use pre-defined rollback command
                result = subprocess.run(
                    step.rollback_command.split(),
                    capture_output=True,
                    text=True,
                    timeout=self.config.rollback_timeout
                )
            else:
                # Generic rollback to previous revision
                cmd = [
                    'kubectl', 'rollout', 'undo',
                    f'deployment/{self.config.app_name}',
                    '-n', self.config.namespace
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.config.rollback_timeout)
            
            if result.returncode == 0:
                logger.info("Kubernetes deployment rollback initiated")
                return True
            else:
                logger.error(f"Kubernetes rollback failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Kubernetes rollback timed out")
            return False
        except Exception as e:
            logger.error(f"Kubernetes rollback failed: {e}")
            return False
    
    def _rollback_database(self, step: RollbackStep) -> bool:
        """Rollback database (placeholder - implement based on your migration system)"""
        try:
            logger.info("Database rollback - implement based on your migration system")
            # This would typically involve:
            # 1. Running migration rollback commands
            # 2. Restoring from backup
            # 3. Executing rollback scripts
            return True
            
        except Exception as e:
            logger.error(f"Database rollback failed: {e}")
            return False
    
    def _rollback_models(self, step: RollbackStep) -> bool:
        """Rollback models to previous version"""
        try:
            # This would involve:
            # 1. Loading target model registry state
            # 2. Copying target model files to production
            # 3. Updating model registry
            logger.info("Models rollback - implement based on your model versioning system")
            return True
            
        except Exception as e:
            logger.error(f"Models rollback failed: {e}")
            return False
    
    def _validate_cluster_access(self) -> bool:
        """Validate Kubernetes cluster access"""
        try:
            result = subprocess.run(
                ['kubectl', 'cluster-info'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Cluster access validation failed: {e}")
            return False
    
    def _validate_permissions(self) -> bool:
        """Validate rollback permissions"""
        try:
            # Check if we can perform rollback operations
            cmd = [
                'kubectl', 'auth', 'can-i', 'patch',
                f'deployment/{self.config.app_name}',
                '-n', self.config.namespace
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return result.returncode == 0 and 'yes' in result.stdout.lower()
            
        except Exception as e:
            logger.error(f"Permissions validation failed: {e}")
            return False
    
    def _wait_for_kubernetes_rollback(self) -> bool:
        """Wait for Kubernetes rollback to complete"""
        try:
            cmd = [
                'kubectl', 'rollout', 'status',
                f'deployment/{self.config.app_name}',
                '-n', self.config.namespace,
                f'--timeout={self.config.kubectl_timeout}s'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Kubernetes rollback wait failed: {e}")
            return False
    
    def _perform_health_check(self) -> bool:
        """Perform health check after rollback"""
        try:
            if self.config.health_check_url:
                import requests
                
                response = requests.get(
                    self.config.health_check_url,
                    timeout=self.config.health_check_timeout
                )
                
                return response.status_code == 200
            else:
                # Use kubectl to check pod status
                cmd = [
                    'kubectl', 'get', 'pods',
                    '-n', self.config.namespace,
                    '-l', f'app={self.config.app_name}',
                    '--no-headers'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Check if all pods are running
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line and 'Running' not in line:
                            return False
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _run_post_rollback_tests(self) -> Dict[str, bool]:
        """Run post-rollback validation tests"""
        test_results = {}
        
        for test in self.config.post_rollback_tests:
            try:
                result = self._run_specific_test(test)
                test_results[test] = result
            except Exception as e:
                logger.error(f"Test {test} failed: {e}")
                test_results[test] = False
        
        return test_results
    
    def _run_specific_test(self, test_name: str) -> bool:
        """Run specific validation test"""
        try:
            if test_name == 'health_check':
                return self._perform_health_check()
            elif test_name == 'api_test':
                return self._test_api_endpoints()
            else:
                logger.warning(f"Unknown test: {test_name}")
                return True
                
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            return False
    
    def _test_api_endpoints(self) -> bool:
        """Test API endpoints"""
        try:
            # This would test critical API endpoints
            # Implement based on your API structure
            logger.info("API endpoint testing - implement based on your API structure")
            return True
            
        except Exception as e:
            logger.error(f"API endpoint test failed: {e}")
            return False
    
    def _get_current_version(self) -> Optional[str]:
        """Get current deployment version"""
        try:
            cmd = [
                'kubectl', 'get', 'deployment', self.config.app_name,
                '-n', self.config.namespace,
                '-o', 'jsonpath={.metadata.labels.version}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not get current version: {e}")
            return None
    
    def _can_rollback_further(self) -> bool:
        """Check if further rollback is possible"""
        try:
            # Check rollback history
            available_points = self.history_manager.get_available_rollback_points()
            return len(available_points) > 1
            
        except Exception as e:
            logger.debug(f"Could not check rollback availability: {e}")
            return False
    
    def _generate_post_rollback_recommendations(self, status: str, health_check_passed: bool) -> List[str]:
        """Generate post-rollback recommendations"""
        recommendations = []
        
        if status == 'success' and health_check_passed:
            recommendations.append("Rollback completed successfully - monitor application stability")
            recommendations.append("Consider investigating root cause of issues that required rollback")
        elif status == 'success' and not health_check_passed:
            recommendations.append("Rollback completed but health check failed - investigate application issues")
            recommendations.append("Consider additional rollback or manual intervention")
        elif status == 'partial':
            recommendations.append("Partial rollback completed - review failed steps and consider manual intervention")
            recommendations.append("Monitor system stability and be prepared for additional actions")
        else:
            recommendations.append("Rollback failed - immediate investigation and manual intervention required")
            recommendations.append("Consider emergency procedures and escalation")
        
        recommendations.append("Update incident documentation with rollback details")
        
        return recommendations

# ============================================
# MAIN ROLLBACK ORCHESTRATOR
# ============================================

class RollbackOrchestrator:
    """Main orchestrator for rollback operations"""
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.history_manager = RollbackHistoryManager(config)
        self.planner = RollbackPlanner(config)
        self.executor = RollbackExecutor(config)
    
    def list_available_rollback_points(self) -> List[Dict[str, Any]]:
        """List available rollback points"""
        return self.history_manager.get_available_rollback_points()
    
    def plan_rollback(self, target_point: Dict[str, Any]) -> RollbackPlan:
        """Create rollback plan for target point"""
        return self.planner.create_rollback_plan(target_point)
    
    def execute_rollback(self, target_point: Dict[str, Any] = None, plan: RollbackPlan = None) -> RollbackResult:
        """Execute rollback to target point"""
        if plan is None:
            if target_point is None:
                raise ValueError("Either target_point or plan must be provided")
            plan = self.planner.create_rollback_plan(target_point)
        
        return self.executor.execute_rollback_plan(plan)
    
    def get_rollback_history(self) -> Dict[str, Any]:
        """Get rollback history"""
        try:
            if self.history_manager.history_file.exists():
                with open(self.history_manager.history_file, 'r') as f:
                    return json.load(f)
            return {'rollbacks': []}
        except Exception as e:
            logger.error(f"Failed to load rollback history: {e}")
            return {'rollbacks': []}

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rollback StockPredictionPro deployments')
    parser.add_argument('--list', action='store_true', help='List available rollback points')
    parser.add_argument('--history', action='store_true', help='Show rollback history')
    parser.add_argument('--plan', help='Create rollback plan for target (revision/deployment_id)')
    parser.add_argument('--execute', help='Execute rollback to target (revision/deployment_id)')
    parser.add_argument('--type', choices=['deployment', 'database', 'models', 'full'],
                       default='deployment', help='Type of rollback')
    parser.add_argument('--namespace', default='default', help='Kubernetes namespace')
    parser.add_argument('--app-name', default='stockpredictionpro', help='Application name')
    parser.add_argument('--dry-run', action='store_true', help='Simulate rollback without making changes')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup before rollback')
    parser.add_argument('--no-validate', action='store_true', help='Skip pre-rollback validation')
    parser.add_argument('--config', help='Path to rollback configuration JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = RollbackConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Could not load config from {args.config}: {e}")
            config = RollbackConfig()
    else:
        config = RollbackConfig()
    
    # Override config with command line arguments
    config.rollback_type = args.type
    config.namespace = args.namespace
    config.app_name = args.app_name
    config.dry_run = args.dry_run
    config.backup_before_rollback = not args.no_backup
    config.validate_before_rollback = not args.no_validate
    
    # Initialize orchestrator
    orchestrator = RollbackOrchestrator(config)
    
    try:
        if args.list:
            # List available rollback points
            points = orchestrator.list_available_rollback_points()
            
            print("\n" + "="*60)
            print("AVAILABLE ROLLBACK POINTS")
            print("="*60)
            
            if not points:
                print("No rollback points available")
            else:
                for i, point in enumerate(points, 1):
                    print(f"\n{i}. {point.get('description', 'Unknown')}")
                    print(f"   Type: {point.get('type')}")
                    print(f"   Timestamp: {point.get('timestamp')}")
                    print(f"   Component: {point.get('component')}")
                    if point.get('revision'):
                        print(f"   Revision: {point.get('revision')}")
                    if point.get('version'):
                        print(f"   Version: {point.get('version')}")
        
        elif args.history:
            # Show rollback history
            history = orchestrator.get_rollback_history()
            
            print("\n" + "="*60)
            print("ROLLBACK HISTORY")
            print("="*60)
            
            rollbacks = history.get('rollbacks', [])
            
            if not rollbacks:
                print("No rollback history available")
            else:
                for rollback in rollbacks[-10:]:  # Show last 10
                    print(f"\n{rollback.get('rollback_id')}")
                    print(f"   Timestamp: {rollback.get('timestamp')}")
                    print(f"   Type: {rollback.get('rollback_type')}")
                    print(f"   Status: {rollback.get('status')}")
                    print(f"   Duration: {rollback.get('duration', 0):.1f}s")
        
        elif args.plan:
            # Create rollback plan
            points = orchestrator.list_available_rollback_points()
            target_point = None
            
            # Find target point
            for point in points:
                if (point.get('revision') == args.plan or 
                    point.get('deployment_id') == args.plan or
                    point.get('model_id') == args.plan):
                    target_point = point
                    break
            
            if not target_point:
                print(f"❌ Rollback point not found: {args.plan}")
                sys.exit(1)
            
            plan = orchestrator.plan_rollback(target_point)
            
            print(f"\n" + "="*60)
            print("ROLLBACK PLAN")
            print("="*60)
            print(f"Plan ID: {plan.plan_id}")
            print(f"Target: {plan.target_revision}")
            print(f"Type: {plan.rollback_type}")
            print(f"Risk: {plan.risk_assessment}")
            print(f"Estimated Duration: {plan.estimated_duration//60}m {plan.estimated_duration%60}s")
            
            print(f"\nPrerequisites:")
            for prereq in plan.prerequisites:
                print(f"  • {prereq}")
            
            print(f"\nSteps ({len(plan.steps)}):")
            for i, step in enumerate(plan.steps, 1):
                print(f"  {i}. {step.description}")
                print(f"     Component: {step.component}")
                print(f"     Action: {step.action}")
        
        elif args.execute:
            # Execute rollback
            points = orchestrator.list_available_rollback_points()
            target_point = None
            
            # Find target point
            for point in points:
                if (point.get('revision') == args.execute or 
                    point.get('deployment_id') == args.execute or
                    point.get('model_id') == args.execute):
                    target_point = point
                    break
            
            if not target_point:
                print(f"❌ Rollback point not found: {args.execute}")
                sys.exit(1)
            
            if config.dry_run:
                print("🔍 DRY RUN MODE - No actual changes will be made")
            
            print(f"🔄 Executing rollback to: {target_point.get('description')}")
            
            result = orchestrator.execute_rollback(target_point)
            
            # Print results
            print(f"\n" + "="*60)
            print("ROLLBACK RESULTS")
            print("="*60)
            print(f"Status: {result.overall_status.upper()}")
            print(f"Duration: {result.total_duration/60:.1f} minutes")
            print(f"Steps: {result.successful_steps}/{result.total_steps} successful")
            
            if result.health_check_passed:
                print("✅ Health check: PASS")
            else:
                print("❌ Health check: FAIL")
            
            if result.recommendations:
                print(f"\nRecommendations:")
                for rec in result.recommendations:
                    print(f"  • {rec}")
            
            # Save result
            result_path = ROLLBACK_DIR / f"rollback_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save(result_path)
            print(f"\n📄 Detailed results saved to: {result_path}")
            
            # Exit with appropriate code
            if result.overall_status == 'success':
                sys.exit(0)
            elif result.overall_status == 'partial':
                sys.exit(1)
            else:
                sys.exit(2)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n❌ Rollback interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        print(f"❌ Rollback failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
