# ============================================
# StockPredictionPro - src/utils/governance.py
# Comprehensive governance, audit, and compliance system
# ============================================

import os
import json
import hashlib
import getpass
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from uuid import uuid4
import pandas as pd
from contextlib import contextmanager
import threading
from collections import defaultdict

from .exceptions import BusinessLogicError
from .logger import get_logger, get_audit_logger
from .helpers import ensure_directory, sanitize_filename
from .file_io import save_data, load_data
from .config_loader import get_config, get_environment

logger = get_logger('governance')
audit_logger = get_audit_logger()

# ============================================
# Core Governance Data Structures
# ============================================

@dataclass
class RunMetadata:
    """Metadata for a model training or prediction run"""
    run_id: str
    run_type: str  # 'training', 'prediction', 'backtest', 'analysis'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # User and environment info
    user: str = field(default_factory=getpass.getuser)
    environment: str = field(default_factory=get_environment)
    hostname: str = field(default_factory=lambda: os.uname().nodename if hasattr(os, 'uname') else 'unknown')
    
    # Code version info
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    code_version: Optional[str] = None
    
    # Configuration info
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    config_hash: Optional[str] = None
    
    # Data info
    input_data_hash: Optional[str] = None
    input_data_size: Optional[int] = None
    input_symbols: List[str] = field(default_factory=list)
    
    # Model info (for training runs)
    model_type: Optional[str] = None
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    model_version: Optional[str] = None
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Output info
    output_files: List[str] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    
    # Status and errors
    status: str = 'running'  # 'running', 'completed', 'failed', 'cancelled'
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        
        # Convert datetime objects to ISO format
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunMetadata':
        """Create from dictionary"""
        # Convert ISO format back to datetime
        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and isinstance(data['end_time'], str):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        return cls(**data)

@dataclass
class AuditEvent:
    """Individual audit event record"""
    event_id: str
    timestamp: datetime
    event_type: str  # 'model_training', 'prediction', 'data_access', 'config_change'
    user: str
    
    # Event details
    action: str
    resource: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    
    # Classification
    severity: str = 'info'  # 'debug', 'info', 'warning', 'error', 'critical'
    category: str = 'operational'  # 'operational', 'security', 'compliance', 'performance'
    
    # Compliance
    data_classification: str = 'public'  # 'public', 'internal', 'confidential', 'restricted'
    retention_days: int = 2555  # 7 years default for financial compliance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class DataLineage:
    """Track data lineage and transformations"""
    data_id: str
    source_type: str  # 'raw_data', 'processed_data', 'model_output'
    creation_time: datetime
    
    # Source information
    source_system: str
    source_location: str
    source_hash: Optional[str] = None
    
    # Transformation info
    transformation_steps: List[Dict[str, Any]] = field(default_factory=list)
    processing_pipeline: Optional[str] = None
    
    # Quality metrics
    quality_score: Optional[float] = None
    validation_status: str = 'unknown'  # 'passed', 'failed', 'warning', 'unknown'
    
    # Dependencies
    parent_data_ids: List[str] = field(default_factory=list)
    child_data_ids: List[str] = field(default_factory=list)
    
    # Metadata
    schema_version: Optional[str] = None
    format_type: str = 'unknown'
    size_bytes: Optional[int] = None
    
    def add_transformation(self, transformation_type: str, parameters: Dict[str, Any], 
                          timestamp: Optional[datetime] = None):
        """Add a transformation step"""
        step = {
            'transformation_type': transformation_type,
            'parameters': parameters,
            'timestamp': (timestamp or datetime.now(timezone.utc)).isoformat(),
            'step_id': str(uuid4())
        }
        self.transformation_steps.append(step)

# ============================================
# Governance Manager
# ============================================

class GovernanceManager:
    """
    Comprehensive governance, audit, and compliance manager
    
    Features:
    - Run tracking and metadata management
    - Audit trail generation and storage
    - Data lineage tracking
    - Compliance reporting
    - Configuration management
    - Code version tracking
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize GovernanceManager
        
        Args:
            base_path: Base path for governance data storage
        """
        if base_path is None:
            # Default to project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]  # Go up to project root
            base_path = project_root / "logs" / "audit"
        
        self.base_path = Path(base_path)
        ensure_directory(self.base_path)
        
        # Initialize subdirectories
        self.runs_dir = self.base_path / "runs"
        self.events_dir = self.base_path / "events"
        self.lineage_dir = self.base_path / "lineage"
        self.compliance_dir = self.base_path / "compliance"
        
        for directory in [self.runs_dir, self.events_dir, self.lineage_dir, self.compliance_dir]:
            ensure_directory(directory)
        
        # Current session tracking
        self.current_session_id = str(uuid4())
        self.current_run: Optional[RunMetadata] = None
        
        # Thread-local storage for context
        self._context = threading.local()
        
        # Event queues and batching
        self._event_queue: List[AuditEvent] = []
        self._queue_lock = threading.Lock()
        self._batch_size = 100
        
        logger.info(f"Governance system initialized at {self.base_path}")
    
    # ============================================
    # Run Management
    # ============================================
    
    def start_run(self, run_type: str, symbols: Optional[List[str]] = None, 
                  model_type: Optional[str] = None, **kwargs) -> str:
        """
        Start a new governed run
        
        Args:
            run_type: Type of run ('training', 'prediction', 'backtest', 'analysis')
            symbols: Stock symbols involved
            model_type: Type of model being used
            **kwargs: Additional metadata
            
        Returns:
            Run ID
        """
        run_id = f"{run_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
        
        # Create run metadata
        self.current_run = RunMetadata(
            run_id=run_id,
            run_type=run_type,
            start_time=datetime.now(timezone.utc),
            input_symbols=symbols or [],
            model_type=model_type,
            **kwargs
        )
        
        # Capture environment information
        self._capture_environment_info()
        self._capture_git_info()
        self._capture_config_snapshot()
        
        # Log run start
        audit_logger.log_user_action(
            user=self.current_run.user,
            action='start_run',
            resource=run_id,
            run_type=run_type,
            model_type=model_type,
            symbols=symbols
        )
        
        logger.info(f"Started {run_type} run: {run_id}")
        return run_id
    
    def end_run(self, status: str = 'completed', error_message: Optional[str] = None,
                performance_metrics: Optional[Dict[str, float]] = None,
                output_files: Optional[List[str]] = None) -> bool:
        """
        End the current run
        
        Args:
            status: Run status ('completed', 'failed', 'cancelled')
            error_message: Error message if failed
            performance_metrics: Performance metrics to record
            output_files: List of output files created
            
        Returns:
            True if successful
        """
        if not self.current_run:
            logger.warning("No active run to end")
            return False
        
        # Update run metadata
        self.current_run.end_time = datetime.now(timezone.utc)
        self.current_run.duration_seconds = (
            self.current_run.end_time - self.current_run.start_time
        ).total_seconds()
        self.current_run.status = status
        
        if error_message:
            self.current_run.error_message = error_message
        
        if performance_metrics:
            self.current_run.performance_metrics.update(performance_metrics)
        
        if output_files:
            self.current_run.output_files.extend(output_files)
        
        # Capture final resource usage
        self._capture_resource_usage()
        
        # Save run metadata
        success = self._save_run_metadata()
        
        # Log run end
        audit_logger.log_user_action(
            user=self.current_run.user,
            action='end_run',
            resource=self.current_run.run_id,
            status=status,
            duration_seconds=self.current_run.duration_seconds,
            performance_metrics=performance_metrics
        )
        
        logger.info(f"Ended run {self.current_run.run_id} with status: {status}")
        
        # Clear current run
        self.current_run = None
        
        return success
    
    def update_run_metadata(self, **updates):
        """Update current run metadata"""
        if not self.current_run:
            logger.warning("No active run to update")
            return
        
        for key, value in updates.items():
            if hasattr(self.current_run, key):
                setattr(self.current_run, key, value)
            else:
                self.current_run.custom_metadata[key] = value
    
    def _capture_environment_info(self):
        """Capture environment information"""
        if not self.current_run:
            return
        
        try:
            import psutil
            self.current_run.memory_usage_mb = psutil.virtual_memory().available / (1024 * 1024)
            self.current_run.cpu_usage_percent = psutil.cpu_percent(interval=1)
        except ImportError:
            logger.debug("psutil not available for resource monitoring")
    
    def _capture_git_info(self):
        """Capture Git repository information"""
        if not self.current_run:
            return
        
        try:
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.current_run.git_commit = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.current_run.git_branch = result.stdout.strip()
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("Git information not available")
    
    def _capture_config_snapshot(self):
        """Capture current configuration snapshot"""
        if not self.current_run:
            return
        
        try:
            # Get all configurations
            configs = {
                'app_config': get_config('app_config'),
                'model_config': get_config('model_config'),
                'trading_config': get_config('trading_config'),
                'indicators_config': get_config('indicators_config')
            }
            
            self.current_run.config_snapshot = configs
            
            # Calculate configuration hash
            config_str = json.dumps(configs, sort_keys=True, default=str)
            self.current_run.config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        except Exception as e:
            logger.warning(f"Failed to capture config snapshot: {e}")
    
    def _capture_resource_usage(self):
        """Capture final resource usage"""
        if not self.current_run:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.current_run.memory_usage_mb = memory_info.rss / (1024 * 1024)
        except ImportError:
            pass
    
    def _save_run_metadata(self) -> bool:
        """Save run metadata to file"""
        if not self.current_run:
            return False
        
        try:
            # Create run file
            run_file = self.runs_dir / f"{self.current_run.run_id}.json"
            
            # Save metadata
            success = save_data(self.current_run.to_dict(), run_file, format_hint='json')
            
            if success:
                logger.debug(f"Saved run metadata: {run_file}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to save run metadata: {e}")
            return False
    
    # ============================================
    # Audit Events
    # ============================================
    
    def log_event(self, event_type: str, action: str, resource: str,
                  details: Optional[Dict[str, Any]] = None,
                  severity: str = 'info', category: str = 'operational',
                  data_classification: str = 'public') -> str:
        """
        Log an audit event
        
        Args:
            event_type: Type of event ('model_training', 'prediction', etc.)
            action: Action performed
            resource: Resource affected
            details: Additional event details
            severity: Event severity
            category: Event category
            data_classification: Data classification level
            
        Returns:
            Event ID
        """
        event = AuditEvent(
            event_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            user=getpass.getuser(),
            action=action,
            resource=resource,
            details=details or {},
            session_id=self.current_session_id,
            run_id=self.current_run.run_id if self.current_run else None,
            severity=severity,
            category=category,
            data_classification=data_classification
        )
        
        # Add to queue
        with self._queue_lock:
            self._event_queue.append(event)
            
            # Flush if queue is full
            if len(self._event_queue) >= self._batch_size:
                self._flush_event_queue()
        
        # Log critical events immediately
        if severity in ['error', 'critical']:
            self._flush_event_queue()
        
        return event.event_id
    
    def _flush_event_queue(self):
        """Flush event queue to storage"""
        if not self._event_queue:
            return
        
        try:
            # Create daily event file
            today = datetime.now().strftime('%Y%m%d')
            event_file = self.events_dir / f"events_{today}.jsonl"
            
            # Append events to file
            with open(event_file, 'a', encoding='utf-8') as f:
                for event in self._event_queue:
                    f.write(json.dumps(event.to_dict(), default=str) + '\n')
            
            logger.debug(f"Flushed {len(self._event_queue)} audit events")
            self._event_queue.clear()
        
        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")
    
    # ============================================
    # Data Lineage
    # ============================================
    
    def create_data_lineage(self, data_id: str, source_type: str, 
                           source_system: str, source_location: str,
                           **kwargs) -> DataLineage:
        """
        Create data lineage record
        
        Args:
            data_id: Unique identifier for the data
            source_type: Type of data source
            source_system: Source system name
            source_location: Location of the source data
            **kwargs: Additional lineage metadata
            
        Returns:
            DataLineage object
        """
        lineage = DataLineage(
            data_id=data_id,
            source_type=source_type,
            creation_time=datetime.now(timezone.utc),
            source_system=source_system,
            source_location=source_location,
            **kwargs
        )
        
        # Save lineage record
        self._save_data_lineage(lineage)
        
        # Log lineage creation
        self.log_event(
            event_type='data_lineage',
            action='create_lineage',
            resource=data_id,
            details={
                'source_type': source_type,
                'source_system': source_system,
                'source_location': source_location
            },
            category='compliance'
        )
        
        return lineage
    
    def update_data_lineage(self, data_id: str, transformation_type: str,
                           parameters: Dict[str, Any]):
        """Update data lineage with transformation"""
        try:
            lineage = self._load_data_lineage(data_id)
            if lineage:
                lineage.add_transformation(transformation_type, parameters)
                self._save_data_lineage(lineage)
                
                # Log transformation
                self.log_event(
                    event_type='data_transformation',
                    action='add_transformation',
                    resource=data_id,
                    details={
                        'transformation_type': transformation_type,
                        'parameters': parameters
                    },
                    category='compliance'
                )
        
        except Exception as e:
            logger.error(f"Failed to update data lineage for {data_id}: {e}")
    
    def _save_data_lineage(self, lineage: DataLineage):
        """Save data lineage to file"""
        lineage_file = self.lineage_dir / f"{lineage.data_id}.json"
        save_data(asdict(lineage), lineage_file, format_hint='json')
    
    def _load_data_lineage(self, data_id: str) -> Optional[DataLineage]:
        """Load data lineage from file"""
        lineage_file = self.lineage_dir / f"{data_id}.json"
        
        if lineage_file.exists():
            try:
                data = load_data(lineage_file, format_hint='json')
                return DataLineage(**data)
            except Exception as e:
                logger.error(f"Failed to load lineage for {data_id}: {e}")
        
        return None
    
    # ============================================
    # Compliance Reporting
    # ============================================
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime,
                                  report_type: str = 'audit_summary') -> Dict[str, Any]:
        """
        Generate compliance report
        
        Args:
            start_date: Report start date
            end_date: Report end date
            report_type: Type of report to generate
            
        Returns:
            Report data dictionary
        """
        report = {
            'report_type': report_type,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'generated_by': getpass.getuser()
        }
        
        if report_type == 'audit_summary':
            report.update(self._generate_audit_summary(start_date, end_date))
        elif report_type == 'model_governance':
            report.update(self._generate_model_governance_report(start_date, end_date))
        elif report_type == 'data_lineage':
            report.update(self._generate_data_lineage_report(start_date, end_date))
        
        # Save report
        report_filename = f"{report_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        report_file = self.compliance_dir / report_filename
        save_data(report, report_file, format_hint='json')
        
        logger.info(f"Generated compliance report: {report_file}")
        return report
    
    def _generate_audit_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate audit event summary"""
        events = self._load_events_in_period(start_date, end_date)
        
        summary = {
            'total_events': len(events),
            'events_by_type': defaultdict(int),
            'events_by_severity': defaultdict(int),
            'events_by_category': defaultdict(int),
            'events_by_user': defaultdict(int),
            'critical_events': []
        }
        
        for event in events:
            summary['events_by_type'][event.event_type] += 1
            summary['events_by_severity'][event.severity] += 1
            summary['events_by_category'][event.category] += 1
            summary['events_by_user'][event.user] += 1
            
            if event.severity in ['error', 'critical']:
                summary['critical_events'].append(event.to_dict())
        
        return summary
    
    def _generate_model_governance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate model governance report"""
        runs = self._load_runs_in_period(start_date, end_date)
        
        model_stats = {
            'total_runs': len(runs),
            'runs_by_type': defaultdict(int),
            'runs_by_status': defaultdict(int),
            'model_types_used': defaultdict(int),
            'performance_summary': {},
            'failed_runs': []
        }
        
        performance_metrics = defaultdict(list)
        
        for run in runs:
            model_stats['runs_by_type'][run.run_type] += 1
            model_stats['runs_by_status'][run.status] += 1
            
            if run.model_type:
                model_stats['model_types_used'][run.model_type] += 1
            
            # Collect performance metrics
            for metric, value in run.performance_metrics.items():
                if isinstance(value, (int, float)):
                    performance_metrics[metric].append(value)
            
            if run.status == 'failed':
                model_stats['failed_runs'].append({
                    'run_id': run.run_id,
                    'error_message': run.error_message,
                    'model_type': run.model_type
                })
        
        # Calculate performance summaries
        for metric, values in performance_metrics.items():
            if values:
                model_stats['performance_summary'][metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return model_stats
    
    def _generate_data_lineage_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate data lineage report"""
        # This would involve scanning lineage files created in the period
        # For now, return basic structure
        return {
            'data_assets_tracked': 0,
            'transformations_logged': 0,
            'data_quality_issues': []
        }
    
    def _load_events_in_period(self, start_date: datetime, end_date: datetime) -> List[AuditEvent]:
        """Load audit events in date range"""
        events = []
        
        # Iterate through daily event files
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            event_file = self.events_dir / f"events_{current_date.strftime('%Y%m%d')}.jsonl"
            
            if event_file.exists():
                try:
                    with open(event_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            event_data = json.loads(line.strip())
                            event_time = datetime.fromisoformat(event_data['timestamp'])
                            
                            if start_date <= event_time <= end_date:
                                events.append(AuditEvent(**event_data))
                
                except Exception as e:
                    logger.error(f"Failed to load events from {event_file}: {e}")
            
            current_date += timedelta(days=1)
        
        return events
    
    def _load_runs_in_period(self, start_date: datetime, end_date: datetime) -> List[RunMetadata]:
        """Load runs in date range"""
        runs = []
        
        # Scan run files
        for run_file in self.runs_dir.glob("*.json"):
            try:
                run_data = load_data(run_file, format_hint='json')
                run = RunMetadata.from_dict(run_data)
                
                if start_date <= run.start_time <= end_date:
                    runs.append(run)
            
            except Exception as e:
                logger.error(f"Failed to load run from {run_file}: {e}")
        
        return runs
    
    # ============================================
    # Context Management
    # ============================================
    
    @contextmanager
    def run_context(self, run_type: str, **kwargs):
        """Context manager for governed runs"""
        run_id = self.start_run(run_type, **kwargs)
        
        try:
            yield run_id
            self.end_run(status='completed')
        except Exception as e:
            self.end_run(status='failed', error_message=str(e))
            raise
    
    def set_context(self, **context):
        """Set context for current thread"""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        
        self._context.data.update(context)
    
    def get_context(self, key: str, default=None):
        """Get context value for current thread"""
        if hasattr(self._context, 'data'):
            return self._context.data.get(key, default)
        return default
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def cleanup_old_records(self, retention_days: int = 2555):
        """Clean up old governance records"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Cleanup old runs
        for run_file in self.runs_dir.glob("*.json"):
            try:
                if run_file.stat().st_mtime < cutoff_date.timestamp():
                    run_file.unlink()
                    logger.debug(f"Deleted old run file: {run_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {run_file}: {e}")
        
        # Cleanup old events
        for event_file in self.events_dir.glob("events_*.jsonl"):
            try:
                if event_file.stat().st_mtime < cutoff_date.timestamp():
                    event_file.unlink()
                    logger.debug(f"Deleted old event file: {event_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {event_file}: {e}")
    
    def get_governance_statistics(self) -> Dict[str, Any]:
        """Get governance system statistics"""
        stats = {
            'total_runs': len(list(self.runs_dir.glob("*.json"))),
            'total_events': 0,
            'total_lineage_records': len(list(self.lineage_dir.glob("*.json"))),
            'current_session_id': self.current_session_id,
            'active_run': self.current_run.run_id if self.current_run else None,
            'queued_events': len(self._event_queue)
        }
        
        # Count events in all files
        for event_file in self.events_dir.glob("events_*.jsonl"):
            try:
                with open(event_file, 'r') as f:
                    stats['total_events'] += sum(1 for _ in f)
            except:
                pass
        
        return stats
    
    def flush_all_data(self):
        """Flush all pending data to storage"""
        with self._queue_lock:
            self._flush_event_queue()
        
        if self.current_run:
            self._save_run_metadata()

# ============================================
# Global Governance Instance
# ============================================

# Create global governance manager
governance = GovernanceManager()

# Convenience functions
def start_run(run_type: str, **kwargs) -> str:
    """Start a governed run"""
    return governance.start_run(run_type, **kwargs)

def end_run(**kwargs) -> bool:
    """End the current run"""
    return governance.end_run(**kwargs)

def log_governance_event(event_type: str, action: str, resource: str, **kwargs) -> str:
    """Log a governance event"""
    return governance.log_event(event_type, action, resource, **kwargs)

def run_context(run_type: str, **kwargs):
    """Context manager for governed runs"""
    return governance.run_context(run_type, **kwargs)

def create_data_lineage(data_id: str, source_type: str, source_system: str, 
                       source_location: str, **kwargs) -> DataLineage:
    """Create data lineage record"""
    return governance.create_data_lineage(data_id, source_type, source_system, 
                                        source_location, **kwargs)

def generate_compliance_report(start_date: datetime, end_date: datetime, 
                             report_type: str = 'audit_summary') -> Dict[str, Any]:
    """Generate compliance report"""
    return governance.generate_compliance_report(start_date, end_date, report_type)

# Cleanup on module shutdown
import atexit
atexit.register(governance.flush_all_data)
