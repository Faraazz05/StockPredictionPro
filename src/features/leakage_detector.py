# ============================================
# StockPredictionPro - src/features/leakage_detector.py
# Advanced data leakage detection for financial machine learning pipelines
# ============================================

import ast
import os
import re
import inspect
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.exceptions import ValidationError, CalculationError
from ..utils.logger import get_logger
from ..utils.timing import time_it

logger = get_logger('features.leakage_detector')

# ============================================
# Configuration and Data Classes
# ============================================

@dataclass
class LeakageRule:
    """Configuration for a leakage detection rule"""
    name: str
    description: str
    severity: str  # 'high', 'medium', 'low'
    rule_type: str  # 'temporal', 'target', 'data_split', 'preprocessing'
    enabled: bool = True

@dataclass
class LeakageIssue:
    """Represents a detected leakage issue"""
    rule_name: str
    severity: str
    file_path: str
    line_number: int
    column_number: int = 0
    issue_type: str = ""
    message: str = ""
    code_snippet: str = ""
    suggestion: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LeakageDetectorConfig:
    """Configuration for leakage detector"""
    enabled_rules: List[str] = field(default_factory=list)
    severity_threshold: str = 'medium'  # 'low', 'medium', 'high'
    output_format: str = 'detailed'  # 'summary', 'detailed', 'json'
    include_suggestions: bool = True
    analyze_code_files: bool = True
    analyze_data_flow: bool = True
    analyze_pipeline: bool = True
    exclude_patterns: List[str] = field(default_factory=list)

# ============================================
# Static Code Analysis Detector
# ============================================

class StaticCodeLeakageDetector:
    """
    Static analysis detector for finding leakage patterns in Python code
    """
    
    def __init__(self, config: Optional[LeakageDetectorConfig] = None):
        self.config = config or LeakageDetectorConfig()
        self.issues: List[LeakageIssue] = []
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[LeakageRule]:
        """Initialize default leakage detection rules"""
        return [
            LeakageRule(
                name="future_target_access",
                description="Accessing future target values in feature creation",
                severity="high",
                rule_type="temporal"
            ),
            LeakageRule(
                name="train_test_mixup",
                description="Using test data in training or vice versa",
                severity="high",
                rule_type="data_split"
            ),
            LeakageRule(
                name="target_in_features",
                description="Target variable included in feature set",
                severity="high",
                rule_type="target"
            ),
            LeakageRule(
                name="preprocessing_leakage",
                description="Preprocessing on combined train/test data",
                severity="medium",
                rule_type="preprocessing"
            ),
            LeakageRule(
                name="lookahead_bias",
                description="Using future information for current predictions",
                severity="high",
                rule_type="temporal"
            ),
            LeakageRule(
                name="group_leakage",
                description="Data leakage across groups (e.g., same entity in train/test)",
                severity="medium",
                rule_type="data_split"
            )
        ]
    
    def analyze_file(self, file_path: str) -> List[LeakageIssue]:
        """Analyze a single Python file for leakage patterns"""
        
        if not os.path.exists(file_path) or not file_path.endswith('.py'):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse the code into AST
            tree = ast.parse(source_code, filename=file_path)
            self._annotate_parents(tree)
            
            # Store file info for reference
            self.current_file = file_path
            self.current_source = source_code
            self.current_lines = source_code.split('\n')
            
            # Run various checks
            for node in ast.walk(tree):
                self._check_future_target_access(node)
                self._check_train_test_mixup(node)
                self._check_target_in_features(node)
                self._check_preprocessing_leakage(node)
                self._check_lookahead_bias(node)
            
            return self.issues
            
        except Exception as e:
            logger.warning(f"Error analyzing file {file_path}: {e}")
            return []
    
    def _annotate_parents(self, tree: ast.AST):
        """Add parent references to AST nodes"""
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
    
    def _check_future_target_access(self, node: ast.AST):
        """Check for accessing future target values"""
        
        # Pattern: df['target'].shift(-n) or df.target.shift(-n)
        if isinstance(node, ast.Call):
            func_name = self._get_attribute_name(node.func)
            
            if func_name == 'shift' and node.args:
                # Check if shift has negative argument
                shift_arg = node.args[0]
                if isinstance(shift_arg, ast.UnaryOp) and isinstance(shift_arg.op, ast.USub):
                    # Check if this is being called on a target-like column
                    if hasattr(node.func, 'value'):
                        column_access = node.func.value
                        if self._is_target_column_access(column_access):
                            self._add_issue(
                                "future_target_access",
                                node,
                                "Future target values accessed with negative shift",
                                "Use only past values: .shift(positive_number)"
                            )
        
        # Pattern: df.loc[future_index, 'target']
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'loc':
                if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) >= 2:
                    col_access = node.slice.elts[1]
                    if self._is_target_column_reference(col_access):
                        # Check if row indexing suggests future access
                        row_access = node.slice.elts[0]
                        if self._suggests_future_access(row_access):
                            self._add_issue(
                                "future_target_access",
                                node,
                                "Potential future target access in .loc indexing",
                                "Ensure row indexing doesn't access future data"
                            )
    
    def _check_train_test_mixup(self, node: ast.AST):
        """Check for train/test data mixup"""
        
        if isinstance(node, ast.Call):
            func_name = self._get_attribute_name(node.func)
            
            # Check model.fit() calls
            if func_name == 'fit' and node.args:
                data_arg = node.args[0]
                data_name = self._get_variable_name(data_arg)
                
                if data_name and ('test' in data_name.lower() or 'val' in data_name.lower()):
                    self._add_issue(
                        "train_test_mixup",
                        node,
                        f"Training model with test/validation data: {data_name}",
                        "Use training data for model.fit()"
                    )
            
            # Check predict() calls
            if func_name == 'predict' and node.args:
                data_arg = node.args[0]
                data_name = self._get_variable_name(data_arg)
                
                if data_name and 'train' in data_name.lower():
                    self._add_issue(
                        "train_test_mixup",
                        node,
                        f"Predicting on training data: {data_name}",
                        "Use separate test data for prediction",
                        severity="medium"
                    )
    
    def _check_target_in_features(self, node: ast.AST):
        """Check if target variable is included in features"""
        
        # Pattern: X = df[features] where features contains target
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target_var = self._get_variable_name(node.targets[0])
                
                if target_var and target_var.lower() in ['x', 'features', 'feature_matrix']:
                    # Check if assignment involves selecting columns that might include target
                    if isinstance(node.value, ast.Subscript):
                        if isinstance(node.value.slice, ast.List):
                            # Check list of column names
                            for elt in node.value.slice.elts:
                                if self._is_target_column_reference(elt):
                                    self._add_issue(
                                        "target_in_features",
                                        node,
                                        "Target variable included in feature matrix",
                                        "Remove target variable from feature selection"
                                    )
    
    def _check_preprocessing_leakage(self, node: ast.AST):
        """Check for preprocessing leakage"""
        
        if isinstance(node, ast.Call):
            func_name = self._get_attribute_name(node.func)
            
            # Check for fit_transform on combined data
            if func_name == 'fit_transform' and node.args:
                data_arg = node.args[0]
                data_name = self._get_variable_name(data_arg)
                
                # Look for patterns suggesting combined train/test data
                if data_name and any(keyword in data_name.lower() 
                                   for keyword in ['combined', 'all', 'full', 'entire']):
                    self._add_issue(
                        "preprocessing_leakage",
                        node,
                        f"fit_transform on combined dataset: {data_name}",
                        "Fit preprocessing on training data only, then transform test data"
                    )
            
            # Check for StandardScaler.fit() on full dataset
            if func_name == 'fit':
                if hasattr(node.func, 'value'):
                    scaler_obj = self._get_variable_name(node.func.value)
                    if scaler_obj and 'scaler' in scaler_obj.lower():
                        if node.args:
                            data_name = self._get_variable_name(node.args[0])
                            if data_name and not any(keyword in data_name.lower() 
                                                   for keyword in ['train', 'x_train']):
                                self._add_issue(
                                    "preprocessing_leakage",
                                    node,
                                    "Scaler fitted on non-training data",
                                    "Fit scaler on training data only"
                                )
    
    def _check_lookahead_bias(self, node: ast.AST):
        """Check for lookahead bias in feature creation"""
        
        # Pattern: rolling operations with future data
        if isinstance(node, ast.Call):
            func_name = self._get_attribute_name(node.func)
            
            if func_name == 'rolling':
                # Check if rolling window parameters suggest future data usage
                if node.keywords:
                    for keyword in node.keywords:
                        if keyword.arg == 'center' and isinstance(keyword.value, ast.Constant):
                            if keyword.value.value is True:
                                self._add_issue(
                                    "lookahead_bias",
                                    node,
                                    "Rolling operation with center=True uses future data",
                                    "Use center=False for time series data"
                                )
    
    def _is_target_column_access(self, node: ast.AST) -> bool:
        """Check if node represents target column access"""
        
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, (ast.Str, ast.Constant)):
                col_name = node.slice.s if isinstance(node.slice, ast.Str) else node.slice.value
                return self._is_target_column_name(str(col_name))
        
        return False
    
    def _is_target_column_reference(self, node: ast.AST) -> bool:
        """Check if node references a target column"""
        
        if isinstance(node, (ast.Str, ast.Constant)):
            col_name = node.s if isinstance(node, ast.Str) else node.value
            return self._is_target_column_name(str(col_name))
        
        return False
    
    def _is_target_column_name(self, col_name: str) -> bool:
        """Check if column name suggests it's a target variable"""
        
        target_keywords = [
            'target', 'label', 'y', 'outcome', 'response', 'dependent',
            'price_next', 'return_next', 'future', 'tomorrow', 'next_day'
        ]
        
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in target_keywords)
    
    def _suggests_future_access(self, node: ast.AST) -> bool:
        """Check if indexing suggests accessing future data"""
        
        # Simple heuristic: look for positive increments or "next" keywords
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return True
        
        if isinstance(node, (ast.Str, ast.Constant)):
            value = node.s if isinstance(node, ast.Str) else str(node.value)
            future_keywords = ['next', 'future', 'ahead', 'forward']
            return any(keyword in value.lower() for keyword in future_keywords)
        
        return False
    
    def _get_attribute_name(self, node: ast.AST) -> Optional[str]:
        """Get attribute name from node"""
        
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        
        return None
    
    def _get_variable_name(self, node: ast.AST) -> Optional[str]:
        """Get variable name from node"""
        
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # For chained attributes, get the base name
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                return base.id
        
        return None
    
    def _add_issue(self, rule_name: str, node: ast.AST, message: str, 
                   suggestion: str, severity: str = None):
        """Add a leakage issue"""
        
        # Find the rule
        rule = next((r for r in self.rules if r.name == rule_name), None)
        if not rule or not rule.enabled:
            return
        
        # Get code snippet
        line_start = max(0, node.lineno - 2)
        line_end = min(len(self.current_lines), node.lineno + 1)
        code_snippet = '\n'.join(self.current_lines[line_start:line_end])
        
        issue = LeakageIssue(
            rule_name=rule_name,
            severity=severity or rule.severity,
            file_path=self.current_file,
            line_number=node.lineno,
            column_number=getattr(node, 'col_offset', 0),
            issue_type=rule.rule_type,
            message=message,
            code_snippet=code_snippet,
            suggestion=suggestion,
            confidence=0.8,  # Default confidence for static analysis
            metadata={'rule_description': rule.description}
        )
        
        self.issues.append(issue)

# ============================================
# Data Flow Leakage Detector
# ============================================

class DataFlowLeakageDetector:
    """
    Detects leakage by analyzing data flow and feature dependencies
    """
    
    def __init__(self, config: Optional[LeakageDetectorConfig] = None):
        self.config = config or LeakageDetectorConfig()
        self.issues: List[LeakageIssue] = []
    
    def analyze_dataframe(self, 
                         df: pd.DataFrame, 
                         target_column: str,
                         feature_columns: List[str],
                         time_column: Optional[str] = None) -> List[LeakageIssue]:
        """Analyze DataFrame for potential leakage"""
        
        self.issues = []
        
        # Check for target leakage in features
        self._check_target_in_feature_names(feature_columns, target_column)
        
        # Check temporal consistency
        if time_column:
            self._check_temporal_consistency(df, time_column, target_column)
        
        # Check for perfect correlations (potential leakage)
        self._check_perfect_correlations(df, feature_columns, target_column)
        
        # Check for duplicate rows
        self._check_duplicate_rows(df)
        
        # Check for data leakage via group dependencies
        self._check_group_leakage(df, feature_columns, target_column)
        
        return self.issues
    
    def _check_target_in_feature_names(self, feature_columns: List[str], target_column: str):
        """Check if target column is included in features"""
        
        if target_column in feature_columns:
            issue = LeakageIssue(
                rule_name="target_in_features",
                severity="high",
                file_path="dataframe_analysis",
                line_number=0,
                issue_type="target",
                message=f"Target column '{target_column}' found in feature list",
                suggestion=f"Remove '{target_column}' from feature columns",
                confidence=1.0
            )
            self.issues.append(issue)
    
    def _check_temporal_consistency(self, df: pd.DataFrame, time_column: str, target_column: str):
        """Check for temporal leakage patterns"""
        
        if time_column not in df.columns:
            return
        
        # Check if data is sorted by time
        time_series = pd.to_datetime(df[time_column])
        if not time_series.is_monotonic_increasing:
            issue = LeakageIssue(
                rule_name="temporal_ordering",
                severity="medium",
                file_path="dataframe_analysis",
                line_number=0,
                issue_type="temporal",
                message="Time series data is not properly sorted",
                suggestion="Sort data by time column before creating features",
                confidence=0.9
            )
            self.issues.append(issue)
    
    def _check_perfect_correlations(self, df: pd.DataFrame, feature_columns: List[str], target_column: str):
        """Check for suspiciously high correlations with target"""
        
        if target_column not in df.columns:
            return
        
        try:
            # Calculate correlations
            correlations = df[feature_columns + [target_column]].corr()[target_column]
            
            # Check for perfect or near-perfect correlations
            high_corr_features = correlations[
                (correlations.abs() > 0.95) & (correlations.index != target_column)
            ]
            
            for feature, corr in high_corr_features.items():
                issue = LeakageIssue(
                    rule_name="perfect_correlation",
                    severity="high",
                    file_path="dataframe_analysis",
                    line_number=0,
                    issue_type="target",
                    message=f"Feature '{feature}' has suspiciously high correlation with target: {corr:.3f}",
                    suggestion=f"Investigate feature '{feature}' for potential leakage",
                    confidence=min(0.9, abs(corr)),
                    metadata={'correlation': corr}
                )
                self.issues.append(issue)
                
        except Exception as e:
            logger.warning(f"Error checking correlations: {e}")
    
    def _check_duplicate_rows(self, df: pd.DataFrame):
        """Check for duplicate rows that might indicate leakage"""
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_ratio = duplicate_count / len(df)
            
            if duplicate_ratio > 0.01:  # More than 1% duplicates
                issue = LeakageIssue(
                    rule_name="duplicate_rows",
                    severity="medium",
                    file_path="dataframe_analysis",
                    line_number=0,
                    issue_type="data_split",
                    message=f"Found {duplicate_count} duplicate rows ({duplicate_ratio:.1%})",
                    suggestion="Remove duplicates to prevent train/test contamination",
                    confidence=0.8,
                    metadata={'duplicate_count': duplicate_count, 'duplicate_ratio': duplicate_ratio}
                )
                self.issues.append(issue)
    
    def _check_group_leakage(self, df: pd.DataFrame, feature_columns: List[str], target_column: str):
        """Check for potential group-based leakage"""
        
        # Look for ID-like columns that might cause group leakage
        potential_id_columns = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'key', 'identifier', 'code']):
                # Check if this column has mostly unique values
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5:  # More than 50% unique values
                    potential_id_columns.append(col)
        
        if potential_id_columns:
            issue = LeakageIssue(
                rule_name="group_leakage",
                severity="medium",
                file_path="dataframe_analysis",
                line_number=0,
                issue_type="data_split",
                message=f"Potential ID columns found: {potential_id_columns}",
                suggestion="Ensure train/test split doesn't have same IDs in both sets",
                confidence=0.7,
                metadata={'id_columns': potential_id_columns}
            )
            self.issues.append(issue)

# ============================================
# Pipeline Leakage Detector
# ============================================

class PipelineLeakageDetector:
    """
    Detects leakage in ML pipelines by analyzing the feature engineering workflow
    """
    
    def __init__(self, config: Optional[LeakageDetectorConfig] = None):
        self.config = config or LeakageDetectorConfig()
        self.issues: List[LeakageIssue] = []
    
    def analyze_pipeline(self, pipeline) -> List[LeakageIssue]:
        """Analyze a feature pipeline for leakage issues"""
        
        self.issues = []
        
        # Check if pipeline has feature store integration
        if hasattr(pipeline, 'feature_store') and pipeline.feature_store:
            self._check_feature_store_leakage(pipeline)
        
        # Check pipeline stages for temporal issues
        if hasattr(pipeline, 'stages'):
            self._check_pipeline_stages(pipeline.stages)
        
        # Check execution order
        self._check_execution_order(pipeline)
        
        return self.issues
    
    def _check_feature_store_leakage(self, pipeline):
        """Check feature store for potential leakage"""
        
        try:
            if pipeline.feature_store:
                # Check if target features are stored
                features = pipeline.feature_store.list_features()
                target_features = [f for f in features if 'target' in f.lower()]
                
                if target_features:
                    issue = LeakageIssue(
                        rule_name="target_in_feature_store",
                        severity="medium",
                        file_path="pipeline_analysis",
                        line_number=0,
                        issue_type="target",
                        message=f"Target-like features found in feature store: {target_features}",
                        suggestion="Ensure target features are not accidentally used as inputs",
                        confidence=0.7
                    )
                    self.issues.append(issue)
        except Exception as e:
            logger.warning(f"Error checking feature store: {e}")
    
    def _check_pipeline_stages(self, stages):
        """Check pipeline stages for leakage patterns"""
        
        for stage in stages:
            # Check for future data access in stage parameters
            if hasattr(stage, 'parameters'):
                self._check_stage_parameters(stage)
            
            # Check stage dependencies
            if hasattr(stage, 'dependencies'):
                self._check_stage_dependencies(stage)
    
    def _check_stage_parameters(self, stage):
        """Check stage parameters for leakage indicators"""
        
        if not hasattr(stage, 'parameters'):
            return
        
        params = stage.parameters
        
        # Check for negative lags or shifts
        for param_name, param_value in params.items():
            if 'lag' in param_name.lower() or 'shift' in param_name.lower():
                if isinstance(param_value, (int, float)) and param_value < 0:
                    issue = LeakageIssue(
                        rule_name="negative_lag",
                        severity="high",
                        file_path="pipeline_analysis",
                        line_number=0,
                        issue_type="temporal",
                        message=f"Stage '{stage.name}' has negative lag/shift: {param_name}={param_value}",
                        suggestion="Use positive lag values to avoid future data",
                        confidence=0.9
                    )
                    self.issues.append(issue)
            
            # Check for 'center' parameter in rolling operations
            if param_name == 'center' and param_value is True:
                if hasattr(stage, 'stage_type') and 'rolling' in str(stage.stage_type).lower():
                    issue = LeakageIssue(
                        rule_name="centered_rolling",
                        severity="medium",
                        file_path="pipeline_analysis",
                        line_number=0,
                        issue_type="temporal",
                        message=f"Stage '{stage.name}' uses centered rolling window",
                        suggestion="Use causal (non-centered) rolling windows for time series",
                        confidence=0.8
                    )
                    self.issues.append(issue)
    
    def _check_stage_dependencies(self, stage):
        """Check stage dependencies for circular or invalid references"""
        
        if not hasattr(stage, 'dependencies') or not stage.dependencies:
            return
        
        # Check if stage depends on target-like features
        target_deps = [dep for dep in stage.dependencies if 'target' in dep.lower()]
        
        if target_deps:
            issue = LeakageIssue(
                rule_name="target_dependency",
                severity="high",
                file_path="pipeline_analysis",
                line_number=0,
                issue_type="target",
                message=f"Stage '{stage.name}' depends on target-like features: {target_deps}",
                suggestion="Remove target dependencies from feature stages",
                confidence=0.9
            )
            self.issues.append(issue)
    
    def _check_execution_order(self, pipeline):
        """Check pipeline execution order for temporal consistency"""
        
        if not hasattr(pipeline, 'stages'):
            return
        
        # Look for target creation stages that come before feature stages
        target_stages = []
        feature_stages = []
        
        for i, stage in enumerate(pipeline.stages):
            if hasattr(stage, 'stage_type'):
                if stage.stage_type == 'target':
                    target_stages.append(i)
                elif stage.stage_type in ['indicator', 'transformer']:
                    feature_stages.append(i)
        
        # Check if any target stages come before feature stages
        for target_idx in target_stages:
            for feature_idx in feature_stages:
                if target_idx < feature_idx:
                    # This might be okay, but flag for review
                    issue = LeakageIssue(
                        rule_name="execution_order",
                        severity="low",
                        file_path="pipeline_analysis",
                        line_number=0,
                        issue_type="temporal",
                        message="Target creation stage comes before feature stages",
                        suggestion="Review execution order to ensure no information leakage",
                        confidence=0.5
                    )
                    self.issues.append(issue)
                    break

# ============================================
# Main Leakage Detector Class
# ============================================

class FinancialLeakageDetector:
    """
    Comprehensive leakage detection system for financial ML pipelines.
    Combines static analysis, data flow analysis, and pipeline analysis.
    """
    
    def __init__(self, config: Optional[LeakageDetectorConfig] = None):
        self.config = config or LeakageDetectorConfig()
        self.static_detector = StaticCodeLeakageDetector(config)
        self.dataflow_detector = DataFlowLeakageDetector(config)
        self.pipeline_detector = PipelineLeakageDetector(config)
        self.all_issues: List[LeakageIssue] = []
    
    @time_it("leakage_detection")
    def analyze_project(self, project_path: str) -> List[LeakageIssue]:
        """Analyze entire project for leakage issues"""
        
        self.all_issues = []
        
        if self.config.analyze_code_files:
            # Find and analyze Python files
            python_files = self._find_python_files(project_path)
            for file_path in python_files:
                issues = self.static_detector.analyze_file(file_path)
                self.all_issues.extend(issues)
        
        return self.all_issues
    
    def analyze_dataframe(self, 
                         df: pd.DataFrame,
                         target_column: str,
                         feature_columns: List[str],
                         time_column: Optional[str] = None) -> List[LeakageIssue]:
        """Analyze DataFrame for leakage"""
        
        if self.config.analyze_data_flow:
            issues = self.dataflow_detector.analyze_dataframe(
                df, target_column, feature_columns, time_column
            )
            return issues
        
        return []
    
    def analyze_pipeline(self, pipeline) -> List[LeakageIssue]:
        """Analyze ML pipeline for leakage"""
        
        if self.config.analyze_pipeline:
            issues = self.pipeline_detector.analyze_pipeline(pipeline)
            return issues
        
        return []
    
    def _find_python_files(self, directory: str) -> List[str]:
        """Recursively find Python files in directory"""
        
        python_files = []
        exclude_patterns = self.config.exclude_patterns or [
            '__pycache__', '.git', '.pytest_cache', 'venv', 'env'
        ]
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)
        
        return python_files
    
    def generate_report(self, issues: List[LeakageIssue]) -> str:
        """Generate a comprehensive leakage report"""
        
        if not issues:
            return "✅ No leakage issues detected!"
        
        # Group issues by severity
        high_issues = [i for i in issues if i.severity == 'high']
        medium_issues = [i for i in issues if i.severity == 'medium']
        low_issues = [i for i in issues if i.severity == 'low']
        
        report = []
        report.append("🔍 FINANCIAL ML LEAKAGE DETECTION REPORT")
        report.append("=" * 50)
        report.append(f"Total Issues Found: {len(issues)}")
        report.append(f"  🔴 High Severity: {len(high_issues)}")
        report.append(f"  🟡 Medium Severity: {len(medium_issues)}")
        report.append(f"  🟢 Low Severity: {len(low_issues)}")
        report.append("")
        
        # Report high severity issues first
        if high_issues:
            report.append("🔴 HIGH SEVERITY ISSUES (Must Fix)")
            report.append("-" * 40)
            for i, issue in enumerate(high_issues, 1):
                report.append(f"{i}. {issue.message}")
                report.append(f"   File: {issue.file_path}:{issue.line_number}")
                report.append(f"   Type: {issue.issue_type}")
                if self.config.include_suggestions:
                    report.append(f"   💡 Suggestion: {issue.suggestion}")
                report.append("")
        
        # Report medium severity issues
        if medium_issues and self.config.severity_threshold in ['low', 'medium']:
            report.append("🟡 MEDIUM SEVERITY ISSUES (Should Fix)")
            report.append("-" * 40)
            for i, issue in enumerate(medium_issues, 1):
                report.append(f"{i}. {issue.message}")
                report.append(f"   File: {issue.file_path}:{issue.line_number}")
                if self.config.include_suggestions:
                    report.append(f"   💡 Suggestion: {issue.suggestion}")
                report.append("")
        
        # Report low severity issues
        if low_issues and self.config.severity_threshold == 'low':
            report.append("🟢 LOW SEVERITY ISSUES (Consider Reviewing)")
            report.append("-" * 40)
            for i, issue in enumerate(low_issues, 1):
                report.append(f"{i}. {issue.message}")
                report.append(f"   File: {issue.file_path}:{issue.line_number}")
                report.append("")
        
        # Summary and recommendations
        report.append("📋 SUMMARY & RECOMMENDATIONS")
        report.append("-" * 40)
        
        if high_issues:
            report.append("⚠️  CRITICAL: Address high severity issues immediately")
            report.append("   These issues can severely impact model performance")
        
        if medium_issues:
            report.append("⚠️  IMPORTANT: Review medium severity issues")
            report.append("   These may cause subtle performance degradation")
        
        report.append("")
        report.append("🛡️  Best Practices to Prevent Leakage:")
        report.append("   • Always use proper train/validation/test splits")
        report.append("   • Fit preprocessors only on training data")
        report.append("   • Use only past data for feature engineering")
        report.append("   • Exclude target variables from feature sets")
        report.append("   • Be careful with group-based splits")
        
        return "\n".join(report)
    
    def export_issues_json(self, issues: List[LeakageIssue]) -> Dict[str, Any]:
        """Export issues as JSON for programmatic processing"""
        
        return {
            'summary': {
                'total_issues': len(issues),
                'high_severity': len([i for i in issues if i.severity == 'high']),
                'medium_severity': len([i for i in issues if i.severity == 'medium']),
                'low_severity': len([i for i in issues if i.severity == 'low']),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'issues': [
                {
                    'rule_name': issue.rule_name,
                    'severity': issue.severity,
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'issue_type': issue.issue_type,
                    'message': issue.message,
                    'suggestion': issue.suggestion,
                    'confidence': issue.confidence,
                    'metadata': issue.metadata
                }
                for issue in issues
            ]
        }

# ============================================
# Utility Functions
# ============================================

def detect_leakage_in_project(project_path: str, 
                             config: Optional[LeakageDetectorConfig] = None) -> str:
    """Quick utility to detect leakage in a project"""
    
    detector = FinancialLeakageDetector(config)
    issues = detector.analyze_project(project_path)
    return detector.generate_report(issues)

def detect_leakage_in_dataframe(df: pd.DataFrame,
                               target_column: str,
                               feature_columns: List[str],
                               time_column: Optional[str] = None) -> str:
    """Quick utility to detect leakage in a DataFrame"""
    
    detector = FinancialLeakageDetector()
    issues = detector.analyze_dataframe(df, target_column, feature_columns, time_column)
    return detector.generate_report(issues)

def detect_leakage_in_pipeline(pipeline) -> str:
    """Quick utility to detect leakage in a pipeline"""
    
    detector = FinancialLeakageDetector()
    issues = detector.analyze_pipeline(pipeline)
    return detector.generate_report(issues)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Financial Leakage Detector")
    
    # Test with sample DataFrame
    print("\n1. Testing DataFrame Analysis")
    
    # Create sample data with intentional leakage
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    
    # Target (intentionally correlated with feature for testing)
    target = feature1 * 2 + feature2 + np.random.randn(n_samples) * 0.1
    
    # Create a leaky feature (almost identical to target)
    leaky_feature = target + np.random.randn(n_samples) * 0.01
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'leaky_feature': leaky_feature,
        'target': target,
        'time': pd.date_range('2020-01-01', periods=n_samples, freq='D')
    })
    
    # Test leakage detection
    feature_columns = ['feature1', 'feature2', 'leaky_feature']
    
    leakage_report = detect_leakage_in_dataframe(
        test_df, 
        target_column='target',
        feature_columns=feature_columns,
        time_column='time'
    )
    
    print("DataFrame Leakage Report:")
    print(leakage_report)
    
    # Test project analysis
    print("\n2. Testing Project Analysis")
    
    # Create a simple test configuration
    config = LeakageDetectorConfig(
        severity_threshold='low',
        include_suggestions=True,
        analyze_code_files=True
    )
    
    # Create detector
    detector = FinancialLeakageDetector(config)
    
    # Test the report generation with no issues
    no_issues_report = detector.generate_report([])
    print("No Issues Report:")
    print(no_issues_report)
    
    print("\nLeakage detector testing completed successfully!")
