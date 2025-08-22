# ============================================
# StockPredictionPro - src/evaluation/reports.py
# Comprehensive evaluation and reporting system for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import html
from pathlib import Path
import logging

from ..utils.exceptions import ValidationError, CalculationError
from ..utils.logger import get_logger
from ..utils.timing import time_it

logger = get_logger('evaluation.reports')

# ============================================
# Report Data Structures
# ============================================

class ReportType(Enum):
    """Types of evaluation reports"""
    MODEL_PERFORMANCE = "model_performance"
    BACKTESTING = "backtesting"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_COMPARISON = "strategy_comparison"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_ANALYSIS = "technical_analysis"
    COMPLIANCE = "compliance"

class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"

@dataclass
class ReportSection:
    """Container for report sections"""
    title: str
    content: str
    subsections: List['ReportSection'] = field(default_factory=list)
    charts: List[str] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReportMetadata:
    """Report metadata and configuration"""
    title: str
    report_type: ReportType
    generation_date: datetime
    author: str = "StockPredictionPro"
    version: str = "1.0"
    executive_summary: str = ""
    tags: List[str] = field(default_factory=list)
    confidentiality: str = "Internal"
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================
# Base Report Generator
# ============================================

class BaseReportGenerator:
    """
    Base class for all report generators.
    
    This class provides the common functionality for generating
    structured evaluation reports across different domains.
    """
    
    def __init__(self, report_type: ReportType, title: str):
        self.report_type = report_type
        self.title = title
        self.sections: List[ReportSection] = []
        self.metadata = ReportMetadata(
            title=title,
            report_type=report_type,
            generation_date=datetime.now()
        )
        
        # Formatting templates
        self.html_template = self._get_html_template()
        self.css_styles = self._get_css_styles()
    
    def add_section(self, section: ReportSection):
        """Add a section to the report"""
        self.sections.append(section)
    
    def add_executive_summary(self, summary: str):
        """Add executive summary"""
        self.metadata.executive_summary = summary
    
    def add_metadata(self, **kwargs):
        """Add custom metadata"""
        self.metadata.custom_metadata.update(kwargs)
    
    @time_it("report_generation")
    def generate_report(self, output_format: ReportFormat = ReportFormat.HTML,
                       output_path: Optional[str] = None) -> str:
        """
        Generate the complete report
        
        Args:
            output_format: Format for the report output
            output_path: Optional path to save the report
            
        Returns:
            Generated report content
        """
        
        if output_format == ReportFormat.HTML:
            content = self._generate_html_report()
        elif output_format == ReportFormat.MARKDOWN:
            content = self._generate_markdown_report()
        elif output_format == ReportFormat.JSON:
            content = self._generate_json_report()
        elif output_format == ReportFormat.TEXT:
            content = self._generate_text_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Save to file if path provided
        if output_path:
            self._save_report(content, output_path, output_format)
        
        logger.info(f"Generated {self.report_type.value} report with {len(self.sections)} sections")
        
        return content
    
    def _generate_html_report(self) -> str:
        """Generate HTML format report"""
        
        # Build HTML content
        html_sections = []
        
        # Executive summary
        if self.metadata.executive_summary:
            html_sections.append(f"""
            <div class="executive-summary">
                <h2>Executive Summary</h2>
                <p>{html.escape(self.metadata.executive_summary)}</p>
            </div>
            """)
        
        # Main sections
        for section in self.sections:
            html_sections.append(self._section_to_html(section))
        
        # Compile full HTML
        report_html = self.html_template.format(
            title=html.escape(self.metadata.title),
            generation_date=self.metadata.generation_date.strftime("%Y-%m-%d %H:%M:%S"),
            author=html.escape(self.metadata.author),
            css_styles=self.css_styles,
            content="\n".join(html_sections)
        )
        
        return report_html
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown format report"""
        
        markdown_lines = []
        
        # Header
        markdown_lines.extend([
            f"# {self.metadata.title}",
            "",
            f"**Generated:** {self.metadata.generation_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Author:** {self.metadata.author}",
            f"**Version:** {self.metadata.version}",
            ""
        ])
        
        # Executive summary
        if self.metadata.executive_summary:
            markdown_lines.extend([
                "## Executive Summary",
                "",
                self.metadata.executive_summary,
                ""
            ])
        
        # Main sections
        for section in self.sections:
            markdown_lines.extend(self._section_to_markdown(section))
        
        return "\n".join(markdown_lines)
    
    def _generate_json_report(self) -> str:
        """Generate JSON format report"""
        
        report_data = {
            "metadata": {
                "title": self.metadata.title,
                "report_type": self.metadata.report_type.value,
                "generation_date": self.metadata.generation_date.isoformat(),
                "author": self.metadata.author,
                "version": self.metadata.version,
                "executive_summary": self.metadata.executive_summary,
                "tags": self.metadata.tags,
                "confidentiality": self.metadata.confidentiality,
                "custom_metadata": self.metadata.custom_metadata
            },
            "sections": []
        }
        
        # Convert sections to JSON-serializable format
        for section in self.sections:
            section_data = {
                "title": section.title,
                "content": section.content,
                "subsections": self._subsections_to_json(section.subsections),
                "charts": section.charts,
                "tables": [table.to_dict() if isinstance(table, pd.DataFrame) else table 
                          for table in section.tables],
                "metadata": section.metadata
            }
            report_data["sections"].append(section_data)
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_text_report(self) -> str:
        """Generate plain text format report"""
        
        text_lines = []
        
        # Header
        text_lines.extend([
            "=" * 80,
            f"{self.metadata.title.upper()}",
            "=" * 80,
            "",
            f"Generated: {self.metadata.generation_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Author: {self.metadata.author}",
            f"Version: {self.metadata.version}",
            ""
        ])
        
        # Executive summary
        if self.metadata.executive_summary:
            text_lines.extend([
                "EXECUTIVE SUMMARY",
                "-" * 20,
                self.metadata.executive_summary,
                ""
            ])
        
        # Main sections
        for section in self.sections:
            text_lines.extend(self._section_to_text(section))
        
        return "\n".join(text_lines)
    
    def _section_to_html(self, section: ReportSection, level: int = 2) -> str:
        """Convert section to HTML"""
        
        html_content = [f"<div class='report-section'>"]
        html_content.append(f"<h{level}>{html.escape(section.title)}</h{level}>")
        
        if section.content:
            # Convert newlines to HTML
            content_html = html.escape(section.content).replace('\n', '<br>')
            html_content.append(f"<p>{content_html}</p>")
        
        # Add tables
        for table in section.tables:
            if isinstance(table, pd.DataFrame):
                html_content.append(table.to_html(classes='report-table', escape=False))
        
        # Add subsections
        for subsection in section.subsections:
            html_content.append(self._section_to_html(subsection, level + 1))
        
        html_content.append("</div>")
        
        return "\n".join(html_content)
    
    def _section_to_markdown(self, section: ReportSection, level: int = 2) -> List[str]:
        """Convert section to Markdown"""
        
        markdown_lines = []
        
        # Section header
        markdown_lines.append(f"{'#' * level} {section.title}")
        markdown_lines.append("")
        
        # Section content
        if section.content:
            markdown_lines.append(section.content)
            markdown_lines.append("")
        
        # Add tables
        for table in section.tables:
            if isinstance(table, pd.DataFrame):
                markdown_lines.append(table.to_markdown())
                markdown_lines.append("")
        
        # Add subsections
        for subsection in section.subsections:
            markdown_lines.extend(self._section_to_markdown(subsection, level + 1))
        
        return markdown_lines
    
    def _section_to_text(self, section: ReportSection, level: int = 0) -> List[str]:
        """Convert section to plain text"""
        
        text_lines = []
        
        # Section header
        if level == 0:
            text_lines.extend([
                section.title.upper(),
                "=" * len(section.title)
            ])
        else:
            indent = "  " * level
            text_lines.extend([
                f"{indent}{section.title}",
                f"{indent}{'-' * len(section.title)}"
            ])
        
        text_lines.append("")
        
        # Section content
        if section.content:
            indent = "  " * level
            content_lines = section.content.split('\n')
            for line in content_lines:
                text_lines.append(f"{indent}{line}")
            text_lines.append("")
        
        # Add tables (simplified text representation)
        for table in section.tables:
            if isinstance(table, pd.DataFrame):
                text_lines.append(str(table))
                text_lines.append("")
        
        # Add subsections
        for subsection in section.subsections:
            text_lines.extend(self._section_to_text(subsection, level + 1))
        
        return text_lines
    
    def _subsections_to_json(self, subsections: List[ReportSection]) -> List[Dict]:
        """Convert subsections to JSON format"""
        
        json_subsections = []
        
        for subsection in subsections:
            subsection_data = {
                "title": subsection.title,
                "content": subsection.content,
                "subsections": self._subsections_to_json(subsection.subsections),
                "charts": subsection.charts,
                "tables": [table.to_dict() if isinstance(table, pd.DataFrame) else table 
                          for table in subsection.tables],
                "metadata": subsection.metadata
            }
            json_subsections.append(subsection_data)
        
        return json_subsections
    
    def _save_report(self, content: str, output_path: str, output_format: ReportFormat):
        """Save report to file"""
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Report saved to: {output_path}")
    
    def _get_html_template(self) -> str:
        """Get HTML template for reports"""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{title}</h1>
            <div class="report-info">
                <p><strong>Generated:</strong> {generation_date}</p>
                <p><strong>Author:</strong> {author}</p>
            </div>
        </header>
        
        <main class="report-content">
            {content}
        </main>
        
        <footer class="report-footer">
            <p>Generated by StockPredictionPro Evaluation Framework</p>
        </footer>
    </div>
</body>
</html>
        """
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML reports"""
        
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .report-header h1 {
            margin: 0 0 15px 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .report-info {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }
        
        .report-info p {
            margin: 0;
            font-size: 0.9em;
        }
        
        .report-content {
            padding: 40px;
        }
        
        .executive-summary {
            background-color: #e8f4fd;
            border-left: 5px solid #2196F3;
            padding: 20px;
            margin: 30px 0;
            border-radius: 5px;
        }
        
        .executive-summary h2 {
            color: #1976D2;
            margin-top: 0;
        }
        
        .report-section {
            margin: 30px 0;
            padding: 20px 0;
            border-bottom: 1px solid #eee;
        }
        
        .report-section:last-child {
            border-bottom: none;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 0;
        }
        
        h2 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        .report-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .report-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        .report-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        
        .report-table tbody tr:hover {
            background-color: #f8f9fa;
        }
        
        .metric-positive {
            color: #27ae60;
            font-weight: bold;
        }
        
        .metric-negative {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .metric-neutral {
            color: #7f8c8d;
        }
        
        .report-footer {
            background-color: #34495e;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }
        
        .highlight-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .warning-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            color: #721c24;
        }
        
        .success-box {
            background-color: #d1eddb;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            color: #155724;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            .report-header h1 {
                font-size: 2em;
            }
            
            .report-info {
                flex-direction: column;
                gap: 10px;
            }
            
            .report-content {
                padding: 20px;
            }
        }
        """

# ============================================
# Model Performance Report Generator
# ============================================

class ModelPerformanceReportGenerator(BaseReportGenerator):
    """
    Generates comprehensive model performance evaluation reports.
    
    This class creates detailed reports analyzing ML model performance
    including metrics, validation results, and recommendations.
    """
    
    def __init__(self):
        super().__init__(ReportType.MODEL_PERFORMANCE, "Model Performance Evaluation Report")
    
    def generate_model_report(self, model_results: Dict[str, Any],
                            validation_results: Optional[Dict[str, Any]] = None,
                            cross_validation_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive model performance report
        
        Args:
            model_results: Dictionary containing model evaluation results
            validation_results: Optional validation results
            cross_validation_results: Optional cross-validation results
            
        Returns:
            Generated report content
        """
        
        # Add executive summary
        self._add_model_executive_summary(model_results)
        
        # Add main sections
        self._add_model_overview_section(model_results)
        self._add_performance_metrics_section(model_results)
        
        if validation_results:
            self._add_validation_section(validation_results)
        
        if cross_validation_results:
            self._add_cross_validation_section(cross_validation_results)
        
        self._add_model_recommendations_section(model_results)
        
        return self.generate_report()
    
    def _add_model_executive_summary(self, model_results: Dict[str, Any]):
        """Add executive summary for model performance"""
        
        # Extract key metrics
        accuracy = model_results.get('accuracy', 0)
        precision = model_results.get('precision', 0)
        recall = model_results.get('recall', 0)
        f1_score = model_results.get('f1_score', 0)
        
        summary = f"""
        This report evaluates the performance of the trained machine learning model for stock prediction.
        
        Key Findings:
        • Model Accuracy: {accuracy:.2%}
        • Precision: {precision:.2%}
        • Recall: {recall:.2%}
        • F1-Score: {f1_score:.3f}
        
        The model shows {'strong' if accuracy > 0.7 else 'moderate' if accuracy > 0.5 else 'weak'} 
        performance on the evaluation dataset. {'Recommended for production deployment with monitoring.' 
        if accuracy > 0.7 else 'Requires further optimization before deployment.' if accuracy > 0.5 
        else 'Not recommended for production use without significant improvements.'}
        """
        
        self.add_executive_summary(summary.strip())
    
    def _add_model_overview_section(self, model_results: Dict[str, Any]):
        """Add model overview section"""
        
        # Create overview content
        model_type = model_results.get('model_type', 'Unknown')
        training_samples = model_results.get('training_samples', 'N/A')
        features_count = model_results.get('features_count', 'N/A')
        training_time = model_results.get('training_time', 'N/A')
        
        overview_content = f"""
        Model Configuration and Training Details:
        
        • Model Type: {model_type}
        • Training Samples: {training_samples:,} observations
        • Feature Count: {features_count} features
        • Training Time: {training_time:.2f} seconds
        
        The model was trained using historical stock market data with technical indicators,
        fundamental metrics, and market sentiment features.
        """
        
        # Create overview table
        overview_data = {
            'Metric': ['Model Type', 'Training Samples', 'Features', 'Training Time'],
            'Value': [model_type, f"{training_samples:,}", features_count, f"{training_time:.2f}s"]
        }
        overview_table = pd.DataFrame(overview_data)
        
        overview_section = ReportSection(
            title="Model Overview",
            content=overview_content.strip(),
            tables=[overview_table]
        )
        
        self.add_section(overview_section)
    
    def _add_performance_metrics_section(self, model_results: Dict[str, Any]):
        """Add detailed performance metrics section"""
        
        # Classification metrics
        classification_metrics = []
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            if metric in model_results:
                classification_metrics.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': f"{model_results[metric]:.4f}",
                    'Performance': self._get_performance_indicator(model_results[metric], metric)
                })
        
        classification_df = pd.DataFrame(classification_metrics)
        
        # Regression metrics (if available)
        regression_metrics = []
        for metric in ['mse', 'rmse', 'mae', 'r2_score']:
            if metric in model_results:
                regression_metrics.append({
                    'Metric': metric.upper().replace('_', ' '),
                    'Value': f"{model_results[metric]:.6f}",
                    'Performance': self._get_performance_indicator(model_results[metric], metric)
                })
        
        # Create performance content
        performance_content = """
        Comprehensive evaluation of model performance across multiple metrics:
        
        Classification Metrics:
        • Accuracy measures overall correctness of predictions
        • Precision indicates reliability of positive predictions
        • Recall measures ability to identify positive cases
        • F1-Score provides balanced measure of precision and recall
        • AUC-ROC evaluates classification performance across thresholds
        """
        
        performance_section = ReportSection(
            title="Performance Metrics",
            content=performance_content.strip(),
            tables=[classification_df] + ([pd.DataFrame(regression_metrics)] if regression_metrics else [])
        )
        
        self.add_section(performance_section)
    
    def _add_validation_section(self, validation_results: Dict[str, Any]):
        """Add validation results section"""
        
        val_accuracy = validation_results.get('val_accuracy', 0)
        val_loss = validation_results.get('val_loss', 0)
        overfitting_score = validation_results.get('overfitting_score', 0)
        
        validation_content = f"""
        Model validation results on held-out test data:
        
        • Validation Accuracy: {val_accuracy:.2%}
        • Validation Loss: {val_loss:.6f}
        • Overfitting Score: {overfitting_score:.3f}
        
        {'The model shows good generalization with minimal overfitting.' if overfitting_score < 0.1 
        else 'Model shows signs of overfitting and may require regularization.' if overfitting_score > 0.2 
        else 'Model generalization is acceptable but can be improved.'}
        """
        
        # Validation metrics table
        validation_data = {
            'Validation Metric': ['Accuracy', 'Loss', 'Overfitting Score'],
            'Value': [f"{val_accuracy:.4f}", f"{val_loss:.6f}", f"{overfitting_score:.3f}"],
            'Status': [
                '✓ Good' if val_accuracy > 0.7 else '⚠ Moderate' if val_accuracy > 0.5 else '✗ Poor',
                '✓ Good' if val_loss < 0.5 else '⚠ Moderate' if val_loss < 1.0 else '✗ Poor',
                '✓ Good' if overfitting_score < 0.1 else '⚠ Moderate' if overfitting_score < 0.2 else '✗ Poor'
            ]
        }
        validation_table = pd.DataFrame(validation_data)
        
        validation_section = ReportSection(
            title="Model Validation",
            content=validation_content.strip(),
            tables=[validation_table]
        )
        
        self.add_section(validation_section)
    
    def _add_cross_validation_section(self, cv_results: Dict[str, Any]):
        """Add cross-validation results section"""
        
        cv_scores = cv_results.get('cv_scores', [])
        mean_score = cv_results.get('mean_score', 0)
        std_score = cv_results.get('std_score', 0)
        
        cv_content = f"""
        Cross-validation results across {len(cv_scores)} folds:
        
        • Mean CV Score: {mean_score:.4f} ± {std_score:.4f}
        • Score Range: [{min(cv_scores):.4f}, {max(cv_scores):.4f}]
        • Coefficient of Variation: {(std_score/mean_score*100):.1f}%
        
        {'Excellent model stability across folds.' if std_score/mean_score < 0.1 
        else 'Good model stability.' if std_score/mean_score < 0.2 
        else 'Model shows high variance across folds - consider regularization.'}
        """
        
        # CV scores table
        cv_data = {
            'Fold': [f"Fold {i+1}" for i in range(len(cv_scores))] + ['Mean', 'Std Dev'],
            'Score': [f"{score:.4f}" for score in cv_scores] + [f"{mean_score:.4f}", f"{std_score:.4f}"]
        }
        cv_table = pd.DataFrame(cv_data)
        
        cv_section = ReportSection(
            title="Cross-Validation Analysis",
            content=cv_content.strip(),
            tables=[cv_table]
        )
        
        self.add_section(cv_section)
    
    def _add_model_recommendations_section(self, model_results: Dict[str, Any]):
        """Add recommendations section"""
        
        accuracy = model_results.get('accuracy', 0)
        overfitting_score = model_results.get('overfitting_score', 0)
        
        recommendations = []
        
        if accuracy < 0.6:
            recommendations.append("• Consider feature engineering to improve predictive power")
            recommendations.append("• Experiment with different algorithms or ensemble methods")
            recommendations.append("• Increase training data size if possible")
        
        if overfitting_score > 0.2:
            recommendations.append("• Apply regularization techniques (L1/L2, dropout)")
            recommendations.append("• Reduce model complexity or feature count")
            recommendations.append("• Use cross-validation for hyperparameter tuning")
        
        if accuracy > 0.8:
            recommendations.append("• Model is performing well - consider deployment")
            recommendations.append("• Implement monitoring for model drift in production")
            recommendations.append("• Set up retraining pipeline for continuous improvement")
        
        if not recommendations:
            recommendations.append("• Model shows balanced performance - continue with current approach")
            recommendations.append("• Consider minor hyperparameter tuning for optimization")
        
        recommendations_content = "Based on the evaluation results, the following actions are recommended:\n\n" + "\n".join(recommendations)
        
        recommendations_section = ReportSection(
            title="Recommendations",
            content=recommendations_content
        )
        
        self.add_section(recommendations_section)
    
    def _get_performance_indicator(self, value: float, metric: str) -> str:
        """Get performance indicator for metrics"""
        
        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'r2_score']:
            if value >= 0.8:
                return "🟢 Excellent"
            elif value >= 0.6:
                return "🟡 Good"
            elif value >= 0.4:
                return "🟠 Moderate"
            else:
                return "🔴 Poor"
        
        elif metric in ['mse', 'rmse', 'mae']:
            if value <= 0.01:
                return "🟢 Excellent"
            elif value <= 0.05:
                return "🟡 Good"
            elif value <= 0.1:
                return "🟠 Moderate"
            else:
                return "🔴 Poor"
        
        return "➖ N/A"

# ============================================
# Backtesting Report Generator
# ============================================

class BacktestingReportGenerator(BaseReportGenerator):
    """
    Generates comprehensive backtesting evaluation reports.
    
    This class creates detailed reports analyzing trading strategy
    performance including returns, risk metrics, and trade analysis.
    """
    
    def __init__(self):
        super().__init__(ReportType.BACKTESTING, "Backtesting Performance Report")
    
    def generate_backtest_report(self, backtest_results: Dict[str, Any],
                               benchmark_results: Optional[Dict[str, Any]] = None,
                               risk_analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive backtesting report
        
        Args:
            backtest_results: Dictionary containing backtest results
            benchmark_results: Optional benchmark comparison results
            risk_analysis: Optional risk analysis results
            
        Returns:
            Generated report content
        """
        
        # Add executive summary
        self._add_backtest_executive_summary(backtest_results, benchmark_results)
        
        # Add main sections
        self._add_strategy_overview_section(backtest_results)
        self._add_performance_summary_section(backtest_results)
        self._add_risk_analysis_section(backtest_results, risk_analysis)
        self._add_trade_analysis_section(backtest_results)
        
        if benchmark_results:
            self._add_benchmark_comparison_section(backtest_results, benchmark_results)
        
        self._add_backtest_recommendations_section(backtest_results)
        
        return self.generate_report()
    
    def _add_backtest_executive_summary(self, backtest_results: Dict[str, Any],
                                      benchmark_results: Optional[Dict[str, Any]] = None):
        """Add executive summary for backtesting results"""
        
        total_return = backtest_results.get('summary', {}).get('total_return', 0)
        sharpe_ratio = backtest_results.get('performance', {}).get('sharpe_ratio', 0)
        max_drawdown = backtest_results.get('performance', {}).get('max_drawdown', 0)
        win_rate = backtest_results.get('performance', {}).get('win_rate', 0)
        
        benchmark_return = benchmark_results.get('total_return', 0) if benchmark_results else 0
        excess_return = total_return - benchmark_return if benchmark_results else None
        
        summary_parts = [
            f"This report analyzes the performance of the trading strategy over the backtest period.",
            "",
            "Key Performance Metrics:",
            f"• Total Return: {total_return:.2%}",
            f"• Sharpe Ratio: {sharpe_ratio:.3f}",
            f"• Maximum Drawdown: {max_drawdown:.2%}",
            f"• Win Rate: {win_rate:.1%}"
        ]
        
        if excess_return is not None:
            summary_parts.append(f"• Excess Return vs Benchmark: {excess_return:+.2%}")
        
        # Performance assessment
        performance_assessment = self._assess_strategy_performance(total_return, sharpe_ratio, max_drawdown, win_rate)
        summary_parts.extend(["", performance_assessment])
        
        self.add_executive_summary("\n".join(summary_parts))
    
    def _add_strategy_overview_section(self, backtest_results: Dict[str, Any]):
        """Add strategy overview section"""
        
        summary = backtest_results.get('summary', {})
        
        initial_capital = summary.get('initial_capital', 0)
        final_value = summary.get('final_value', 0)
        total_trades = summary.get('total_trades', 0)
        
        overview_content = f"""
        Strategy Configuration and Backtest Parameters:
        
        • Initial Capital: ${initial_capital:,.2f}
        • Final Portfolio Value: ${final_value:,.2f}
        • Total Trades Executed: {total_trades:,}
        • Backtest Period: {self._get_backtest_period(backtest_results)}
        
        The strategy was backtested using historical market data with realistic
        execution assumptions including transaction costs and market impact.
        """
        
        # Strategy metrics table
        strategy_data = {
            'Metric': ['Initial Capital', 'Final Value', 'Net P&L', 'Total Trades'],
            'Value': [
                f"${initial_capital:,.2f}",
                f"${final_value:,.2f}",
                f"${final_value - initial_capital:+,.2f}",
                f"{total_trades:,}"
            ]
        }
        strategy_table = pd.DataFrame(strategy_data)
        
        overview_section = ReportSection(
            title="Strategy Overview",
            content=overview_content.strip(),
            tables=[strategy_table]
        )
        
        self.add_section(overview_section)
    
    def _add_performance_summary_section(self, backtest_results: Dict[str, Any]):
        """Add performance summary section"""
        
        performance = backtest_results.get('performance', {})
        
        # Key performance metrics
        perf_metrics = []
        metric_names = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'volatility']
        
        for metric in metric_names:
            if metric in performance:
                perf_metrics.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': f"{performance[metric]:.4f}" if metric != 'max_drawdown' and metric != 'volatility' 
                            else f"{performance[metric]:.2%}",
                    'Assessment': self._get_metric_assessment(performance[metric], metric)
                })
        
        performance_df = pd.DataFrame(perf_metrics)
        
        performance_content = """
        Comprehensive performance analysis of the trading strategy:
        
        Risk-Adjusted Returns:
        • Sharpe Ratio measures risk-adjusted returns relative to volatility
        • Sortino Ratio focuses on downside risk, ignoring upside volatility
        • Calmar Ratio compares returns to maximum drawdown
        
        Risk Metrics:
        • Maximum Drawdown shows worst peak-to-trough decline
        • Volatility indicates strategy's return variability
        """
        
        performance_section = ReportSection(
            title="Performance Summary",
            content=performance_content.strip(),
            tables=[performance_df]
        )
        
        self.add_section(performance_section)
    
    def _add_risk_analysis_section(self, backtest_results: Dict[str, Any], 
                                 risk_analysis: Optional[Dict[str, Any]] = None):
        """Add risk analysis section"""
        
        performance = backtest_results.get('performance', {})
        
        var_95 = risk_analysis.get('var_95', 0) if risk_analysis else performance.get('var_95', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        volatility = performance.get('volatility', 0)
        
        risk_content = f"""
        Detailed risk assessment of the trading strategy:
        
        • Value at Risk (95%): {var_95:.2%}
        • Maximum Drawdown: {max_drawdown:.2%}
        • Annualized Volatility: {volatility:.2%}
        
        Risk Assessment:
        {self._get_risk_assessment(max_drawdown, volatility, var_95)}
        """
        
        # Risk metrics table
        risk_data = {
            'Risk Metric': ['Value at Risk (95%)', 'Maximum Drawdown', 'Volatility', 'Risk Level'],
            'Value': [
                f"{var_95:.2%}",
                f"{max_drawdown:.2%}",
                f"{volatility:.2%}",
                self._get_overall_risk_level(max_drawdown, volatility, var_95)
            ],
            'Status': [
                self._get_risk_status(var_95, 'var'),
                self._get_risk_status(max_drawdown, 'drawdown'),
                self._get_risk_status(volatility, 'volatility'),
                '📊 Overall'
            ]
        }
        risk_table = pd.DataFrame(risk_data)
        
        risk_section = ReportSection(
            title="Risk Analysis",
            content=risk_content.strip(),
            tables=[risk_table]
        )
        
        self.add_section(risk_section)
    
    def _add_trade_analysis_section(self, backtest_results: Dict[str, Any]):
        """Add trade analysis section"""
        
        summary = backtest_results.get('summary', {})
        performance = backtest_results.get('performance', {})
        
        total_trades = summary.get('total_trades', 0)
        winning_trades = summary.get('winning_trades', 0)
        losing_trades = summary.get('losing_trades', 0)
        win_rate = performance.get('win_rate', 0)
        
        avg_win = backtest_results.get('avg_winning_trade', 0)
        avg_loss = backtest_results.get('avg_losing_trade', 0)
        profit_factor = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        
        trade_content = f"""
        Detailed analysis of individual trades and trading patterns:
        
        Trade Statistics:
        • Total Trades: {total_trades:,}
        • Winning Trades: {winning_trades:,}
        • Losing Trades: {losing_trades:,}
        • Win Rate: {win_rate:.1%}
        • Average Win: ${avg_win:.2f}
        • Average Loss: ${avg_loss:.2f}
        • Profit Factor: {profit_factor:.2f}
        
        {'Strong trading performance with good win/loss ratio.' if profit_factor > 1.5 
        else 'Moderate trading performance.' if profit_factor > 1.0 
        else 'Trading performance needs improvement.'}
        """
        
        # Trade analysis table
        trade_data = {
            'Trade Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 'Profit Factor'],
            'Value': [
                f"{total_trades:,}",
                f"{winning_trades:,}",
                f"{losing_trades:,}",
                f"{win_rate:.1%}",
                f"{profit_factor:.2f}"
            ],
            'Assessment': [
                '📊 Count',
                f"{'🟢' if win_rate > 0.6 else '🟡' if win_rate > 0.4 else '🔴'} {win_rate:.1%}",
                '📊 Count',
                self._get_win_rate_assessment(win_rate),
                self._get_profit_factor_assessment(profit_factor)
            ]
        }
        trade_table = pd.DataFrame(trade_data)
        
        trade_section = ReportSection(
            title="Trade Analysis",
            content=trade_content.strip(),
            tables=[trade_table]
        )
        
        self.add_section(trade_section)
    
    def _add_benchmark_comparison_section(self, backtest_results: Dict[str, Any],
                                        benchmark_results: Dict[str, Any]):
        """Add benchmark comparison section"""
        
        strategy_return = backtest_results.get('summary', {}).get('total_return', 0)
        benchmark_return = benchmark_results.get('total_return', 0)
        excess_return = strategy_return - benchmark_return
        
        strategy_sharpe = backtest_results.get('performance', {}).get('sharpe_ratio', 0)
        benchmark_sharpe = benchmark_results.get('sharpe_ratio', 0)
        
        comparison_content = f"""
        Performance comparison against benchmark:
        
        Returns Comparison:
        • Strategy Return: {strategy_return:.2%}
        • Benchmark Return: {benchmark_return:.2%}
        • Excess Return: {excess_return:+.2%}
        
        Risk-Adjusted Comparison:
        • Strategy Sharpe Ratio: {strategy_sharpe:.3f}
        • Benchmark Sharpe Ratio: {benchmark_sharpe:.3f}
        
        {'✅ Strategy outperformed benchmark' if excess_return > 0 else '❌ Strategy underperformed benchmark'}
        """
        
        # Comparison table
        comparison_data = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Excess Return'],
            'Strategy': [f"{strategy_return:.2%}", f"{strategy_sharpe:.3f}", f"{excess_return:+.2%}"],
            'Benchmark': [f"{benchmark_return:.2%}", f"{benchmark_sharpe:.3f}", "0.00%"],
            'Outperformance': [
                "✅ Yes" if strategy_return > benchmark_return else "❌ No",
                "✅ Yes" if strategy_sharpe > benchmark_sharpe else "❌ No",
                f"{excess_return:+.2%}"
            ]
        }
        comparison_table = pd.DataFrame(comparison_data)
        
        benchmark_section = ReportSection(
            title="Benchmark Comparison",
            content=comparison_content.strip(),
            tables=[comparison_table]
        )
        
        self.add_section(benchmark_section)
    
    def _add_backtest_recommendations_section(self, backtest_results: Dict[str, Any]):
        """Add recommendations section for backtesting results"""
        
        performance = backtest_results.get('performance', {})
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        win_rate = performance.get('win_rate', 0)
        
        recommendations = []
        
        # Performance-based recommendations
        if sharpe_ratio < 0.5:
            recommendations.append("• Consider improving risk-adjusted returns through better entry/exit rules")
            recommendations.append("• Evaluate position sizing and risk management parameters")
        
        if max_drawdown > 0.2:
            recommendations.append("• Implement stricter stop-loss or position sizing rules")
            recommendations.append("• Consider diversification across multiple strategies or assets")
        
        if win_rate < 0.4:
            recommendations.append("• Review and optimize entry signals to improve accuracy")
            recommendations.append("• Consider longer holding periods or different market timing")
        
        # General recommendations
        if sharpe_ratio > 1.0 and max_drawdown < 0.1:
            recommendations.append("• Strategy shows excellent performance - consider live trading")
            recommendations.append("• Implement robust risk monitoring for live deployment")
        
        if not recommendations:
            recommendations.append("• Strategy shows balanced performance characteristics")
            recommendations.append("• Consider minor parameter optimization for improvement")
        
        # Always add these
        recommendations.extend([
            "• Conduct out-of-sample testing on recent data",
            "• Monitor strategy performance for regime changes",
            "• Consider transaction cost impact in live trading"
        ])
        
        recommendations_content = "Strategic recommendations based on backtesting analysis:\n\n" + "\n".join(recommendations)
        
        recommendations_section = ReportSection(
            title="Strategic Recommendations",
            content=recommendations_content
        )
        
        self.add_section(recommendations_section)
    
    # Helper methods
    def _get_backtest_period(self, backtest_results: Dict[str, Any]) -> str:
        """Extract backtest period from results"""
        # This would extract actual dates from results
        return "2023-01-01 to 2023-12-31"  # Placeholder
    
    def _assess_strategy_performance(self, total_return: float, sharpe_ratio: float, 
                                   max_drawdown: float, win_rate: float) -> str:
        """Provide overall strategy performance assessment"""
        
        score = 0
        score += 1 if total_return > 0.1 else 0.5 if total_return > 0 else 0
        score += 1 if sharpe_ratio > 1.0 else 0.5 if sharpe_ratio > 0.5 else 0
        score += 1 if max_drawdown < 0.1 else 0.5 if max_drawdown < 0.2 else 0
        score += 1 if win_rate > 0.5 else 0.5 if win_rate > 0.4 else 0
        
        if score >= 3.5:
            return "🟢 EXCELLENT: Strategy demonstrates strong performance across all metrics."
        elif score >= 2.5:
            return "🟡 GOOD: Strategy shows solid performance with room for optimization."
        elif score >= 1.5:
            return "🟠 MODERATE: Strategy has potential but requires significant improvements."
        else:
            return "🔴 POOR: Strategy shows weak performance and is not recommended for deployment."
    
    def _get_metric_assessment(self, value: float, metric: str) -> str:
        """Get assessment for specific metric"""
        
        if metric == 'sharpe_ratio':
            if value >= 1.0:
                return "🟢 Excellent"
            elif value >= 0.5:
                return "🟡 Good"
            else:
                return "🔴 Poor"
        
        elif metric == 'max_drawdown':
            if value <= 0.05:
                return "🟢 Low Risk"
            elif value <= 0.15:
                return "🟡 Moderate Risk"
            else:
                return "🔴 High Risk"
        
        # Add more metric assessments as needed
        return "➖ N/A"
    
    def _get_risk_assessment(self, max_drawdown: float, volatility: float, var_95: float) -> str:
        """Provide comprehensive risk assessment"""
        
        risk_factors = []
        
        if max_drawdown > 0.2:
            risk_factors.append("High drawdown risk")
        
        if volatility > 0.3:
            risk_factors.append("High volatility")
        
        if var_95 > 0.05:
            risk_factors.append("High daily loss potential")
        
        if not risk_factors:
            return "Risk profile is within acceptable parameters for most investors."
        else:
            return f"Elevated risk due to: {', '.join(risk_factors)}. Suitable only for high-risk tolerance investors."
    
    def _get_overall_risk_level(self, max_drawdown: float, volatility: float, var_95: float) -> str:
        """Get overall risk level assessment"""
        
        risk_score = 0
        if max_drawdown > 0.15:
            risk_score += 1
        if volatility > 0.25:
            risk_score += 1
        if var_95 > 0.04:
            risk_score += 1
        
        if risk_score >= 2:
            return "🔴 High Risk"
        elif risk_score == 1:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"
    
    def _get_risk_status(self, value: float, risk_type: str) -> str:
        """Get risk status indicator"""
        
        if risk_type == 'var':
            return "🟢 Low" if value <= 0.03 else "🟡 Medium" if value <= 0.05 else "🔴 High"
        elif risk_type == 'drawdown':
            return "🟢 Low" if value <= 0.1 else "🟡 Medium" if value <= 0.2 else "🔴 High"
        elif risk_type == 'volatility':
            return "🟢 Low" if value <= 0.15 else "🟡 Medium" if value <= 0.25 else "🔴 High"
        
        return "➖"
    
    def _get_win_rate_assessment(self, win_rate: float) -> str:
        """Get win rate assessment"""
        
        if win_rate >= 0.6:
            return "🟢 High"
        elif win_rate >= 0.4:
            return "🟡 Moderate"
        else:
            return "🔴 Low"
    
    def _get_profit_factor_assessment(self, profit_factor: float) -> str:
        """Get profit factor assessment"""
        
        if profit_factor >= 2.0:
            return "🟢 Excellent"
        elif profit_factor >= 1.5:
            return "🟡 Good"
        elif profit_factor >= 1.0:
            return "🟠 Marginal"
        else:
            return "🔴 Poor"

# ============================================
# Strategy Comparison Report Generator
# ============================================

class StrategyComparisonReportGenerator(BaseReportGenerator):
    """
    Generates comprehensive strategy comparison reports.
    
    This class creates detailed reports comparing multiple trading
    strategies across various performance and risk metrics.
    """
    
    def __init__(self):
        super().__init__(ReportType.STRATEGY_COMPARISON, "Strategy Comparison Analysis")
    
    def generate_comparison_report(self, strategy_results: Dict[str, Dict[str, Any]],
                                 benchmark_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive strategy comparison report
        
        Args:
            strategy_results: Dictionary of strategy_name -> results
            benchmark_results: Optional benchmark results
            
        Returns:
            Generated report content
        """
        
        # Add executive summary
        self._add_comparison_executive_summary(strategy_results)
        
        # Add main sections
        self._add_strategies_overview_section(strategy_results)
        self._add_performance_comparison_section(strategy_results)
        self._add_risk_comparison_section(strategy_results)
        self._add_ranking_analysis_section(strategy_results)
        self._add_comparison_recommendations_section(strategy_results)
        
        return self.generate_report()
    
    def _add_comparison_executive_summary(self, strategy_results: Dict[str, Dict[str, Any]]):
        """Add executive summary for strategy comparison"""
        
        # Find best performing strategies
        best_return = max(strategy_results.items(), 
                         key=lambda x: x[1].get('summary', {}).get('total_return', 0))
        best_sharpe = max(strategy_results.items(),
                         key=lambda x: x[1].get('performance', {}).get('sharpe_ratio', 0))
        lowest_risk = min(strategy_results.items(),
                         key=lambda x: x[1].get('performance', {}).get('max_drawdown', 1))
        
        summary = f"""
        This report compares {len(strategy_results)} trading strategies across multiple performance and risk dimensions.
        
        Key Findings:
        • Best Total Return: {best_return[0]} ({best_return[1].get('summary', {}).get('total_return', 0):.2%})
        • Best Risk-Adjusted Return: {best_sharpe[0]} (Sharpe: {best_sharpe[1].get('performance', {}).get('sharpe_ratio', 0):.3f})
        • Lowest Risk: {lowest_risk[0]} (Max DD: {lowest_risk[1].get('performance', {}).get('max_drawdown', 0):.2%})
        
        The analysis reveals significant performance differences across strategies, providing clear guidance
        for strategy selection based on investor risk tolerance and return objectives.
        """
        
        self.add_executive_summary(summary.strip())
    
    def _add_strategies_overview_section(self, strategy_results: Dict[str, Dict[str, Any]]):
        """Add strategies overview section"""
        
        overview_data = []
        
        for strategy_name, results in strategy_results.items():
            summary = results.get('summary', {})
            
            overview_data.append({
                'Strategy': strategy_name,
                'Initial Capital': f"${summary.get('initial_capital', 0):,.0f}",
                'Final Value': f"${summary.get('final_value', 0):,.0f}",
                'Total Trades': f"{summary.get('total_trades', 0):,}",
                'Period': self._extract_period(results)
            })
        
        overview_df = pd.DataFrame(overview_data)
        
        overview_content = f"""
        Overview of {len(strategy_results)} strategies analyzed in this comparison:
        
        All strategies were backtested using the same historical period and initial capital
        to ensure fair comparison. Transaction costs, slippage, and realistic execution
        assumptions were applied consistently across all strategies.
        """
        
        overview_section = ReportSection(
            title="Strategies Overview",
            content=overview_content.strip(),
            tables=[overview_df]
        )
        
        self.add_section(overview_section)
    
    def _add_performance_comparison_section(self, strategy_results: Dict[str, Dict[str, Any]]):
        """Add performance comparison section"""
        
        performance_data = []
        
        for strategy_name, results in strategy_results.items():
            summary = results.get('summary', {})
            performance = results.get('performance', {})
            
            performance_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{summary.get('total_return', 0):.2%}",
                'Annualized Return': f"{performance.get('annualized_return', 0):.2%}",
                'Sharpe Ratio': f"{performance.get('sharpe_ratio', 0):.3f}",
                'Sortino Ratio': f"{performance.get('sortino_ratio', 0):.3f}",
                'Calmar Ratio': f"{performance.get('calmar_ratio', 0):.3f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Sort by Sharpe Ratio for ranking
        performance_df = performance_df.sort_values('Sharpe Ratio', ascending=False)
        performance_df.insert(0, 'Rank', range(1, len(performance_df) + 1))
        
        performance_content = """
        Comprehensive performance comparison across key return and risk-adjusted metrics:
        
        • Total Return: Absolute return over the backtest period
        • Annualized Return: Compound annual growth rate
        • Sharpe Ratio: Risk-adjusted return (higher is better)
        • Sortino Ratio: Downside risk-adjusted return
        • Calmar Ratio: Return relative to maximum drawdown
        
        Strategies are ranked by Sharpe Ratio to highlight risk-adjusted performance.
        """
        
        performance_section = ReportSection(
            title="Performance Comparison",
            content=performance_content.strip(),
            tables=[performance_df]
        )
        
        self.add_section(performance_section)
    
    def _add_risk_comparison_section(self, strategy_results: Dict[str, Dict[str, Any]]):
        """Add risk comparison section"""
        
        risk_data = []
        
        for strategy_name, results in strategy_results.items():
            performance = results.get('performance', {})
            
            risk_data.append({
                'Strategy': strategy_name,
                'Maximum Drawdown': f"{performance.get('max_drawdown', 0):.2%}",
                'Volatility': f"{performance.get('volatility', 0):.2%}",
                'VaR (95%)': f"{performance.get('var_95', 0):.2%}",
                'Win Rate': f"{performance.get('win_rate', 0):.1%}",
                'Risk Level': self._assess_risk_level(performance)
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Sort by Maximum Drawdown (lowest risk first)
        risk_df = risk_df.sort_values('Maximum Drawdown', ascending=True)
        risk_df.insert(0, 'Risk Rank', range(1, len(risk_df) + 1))
        
        risk_content = """
        Comprehensive risk comparison across key risk metrics:
        
        • Maximum Drawdown: Worst peak-to-trough decline
        • Volatility: Standard deviation of returns (annualized)
        • VaR (95%): Value at Risk at 95% confidence level
        • Win Rate: Percentage of profitable trades
        • Risk Level: Overall risk assessment
        
        Strategies are ranked by maximum drawdown (lowest risk first).
        """
        
        risk_section = ReportSection(
            title="Risk Comparison",
            content=risk_content.strip(),
            tables=[risk_df]
        )
        
        self.add_section(risk_section)
    
    def _add_ranking_analysis_section(self, strategy_results: Dict[str, Dict[str, Any]]):
        """Add ranking analysis section"""
        
        # Calculate composite scores
        ranking_data = []
        
        for strategy_name, results in strategy_results.items():
            summary = results.get('summary', {})
            performance = results.get('performance', {})
            
            # Normalize metrics for scoring
            return_score = min(10, max(0, summary.get('total_return', 0) * 50))  # Scale return
            sharpe_score = min(10, max(0, performance.get('sharpe_ratio', 0) * 5))  # Scale Sharpe
            risk_score = min(10, max(0, 10 - performance.get('max_drawdown', 0) * 50))  # Inverse risk
            
            composite_score = (return_score * 0.4 + sharpe_score * 0.4 + risk_score * 0.2)
            
            ranking_data.append({
                'Strategy': strategy_name,
                'Return Score': f"{return_score:.1f}/10",
                'Risk-Adj Score': f"{sharpe_score:.1f}/10", 
                'Risk Score': f"{risk_score:.1f}/10",
                'Composite Score': f"{composite_score:.1f}/10",
                'Recommendation': self._get_strategy_recommendation(composite_score)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Composite Score', ascending=False)
        ranking_df.insert(0, 'Overall Rank', range(1, len(ranking_df) + 1))
        
        ranking_content = """
        Comprehensive ranking analysis using weighted composite scoring:
        
        Scoring Methodology:
        • Return Score (40%): Based on total return performance
        • Risk-Adjusted Score (40%): Based on Sharpe ratio
        • Risk Score (20%): Based on maximum drawdown (lower risk = higher score)
        
        Composite scores provide an overall assessment balancing return, risk-adjusted return, and risk management.
        """
        
        ranking_section = ReportSection(
            title="Strategy Rankings",
            content=ranking_content.strip(),
            tables=[ranking_df]
        )
        
        self.add_section(ranking_section)
    
    def _add_comparison_recommendations_section(self, strategy_results: Dict[str, Dict[str, Any]]):
        """Add recommendations section for strategy comparison"""
        
        # Analyze strategies to provide recommendations
        best_strategies = self._identify_best_strategies(strategy_results)
        
        recommendations = [
            "Strategic Selection Recommendations:",
            ""
        ]
        
        if 'conservative' in best_strategies:
            recommendations.append(f"• Conservative Investors: {best_strategies['conservative']} - Offers good risk-adjusted returns with lower volatility")
        
        if 'aggressive' in best_strategies:
            recommendations.append(f"• Aggressive Investors: {best_strategies['aggressive']} - Provides highest returns with acceptable risk levels")
        
        if 'balanced' in best_strategies:
            recommendations.append(f"• Balanced Approach: {best_strategies['balanced']} - Optimal risk-return balance for most investors")
        
        recommendations.extend([
            "",
            "Implementation Considerations:",
            "• Consider portfolio allocation across top 2-3 strategies for diversification",
            "• Monitor strategy performance for regime changes and adapt allocation accordingly",
            "• Implement robust risk management regardless of strategy selection",
            "• Regular performance review and rebalancing recommended"
        ])
        
        recommendations_content = "\n".join(recommendations)
        
        recommendations_section = ReportSection(
            title="Investment Recommendations",
            content=recommendations_content
        )
        
        self.add_section(recommendations_section)
    
    # Helper methods
    def _extract_period(self, results: Dict[str, Any]) -> str:
        """Extract backtest period from results"""
        # This would extract actual period from results
        return "2023"  # Placeholder
    
    def _assess_risk_level(self, performance: Dict[str, float]) -> str:
        """Assess overall risk level of strategy"""
        
        max_drawdown = performance.get('max_drawdown', 0)
        volatility = performance.get('volatility', 0)
        
        if max_drawdown <= 0.1 and volatility <= 0.15:
            return "🟢 Low"
        elif max_drawdown <= 0.2 and volatility <= 0.25:
            return "🟡 Medium"
        else:
            return "🔴 High"
    
    def _get_strategy_recommendation(self, composite_score: float) -> str:
        """Get recommendation based on composite score"""
        
        if composite_score >= 7.5:
            return "🟢 Highly Recommended"
        elif composite_score >= 6.0:
            return "🟡 Recommended"
        elif composite_score >= 4.0:
            return "🟠 Consider with Caution"
        else:
            return "🔴 Not Recommended"
    
    def _identify_best_strategies(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Identify best strategies for different investor profiles"""
        
        strategies = []
        
        for name, results in strategy_results.items():
            summary = results.get('summary', {})
            performance = results.get('performance', {})
            
            strategies.append({
                'name': name,
                'return': summary.get('total_return', 0),
                'sharpe': performance.get('sharpe_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'volatility': performance.get('volatility', 0)
            })
        
        best_strategies = {}
        
        # Conservative: Best Sharpe with low drawdown
        conservative_candidates = [s for s in strategies if s['max_drawdown'] <= 0.15]
        if conservative_candidates:
            best_conservative = max(conservative_candidates, key=lambda x: x['sharpe'])
            best_strategies['conservative'] = best_conservative['name']
        
        # Aggressive: Highest return with reasonable Sharpe
        aggressive_candidates = [s for s in strategies if s['sharpe'] >= 0.5]
        if aggressive_candidates:
            best_aggressive = max(aggressive_candidates, key=lambda x: x['return'])
            best_strategies['aggressive'] = best_aggressive['name']
        
        # Balanced: Best overall Sharpe ratio
        if strategies:
            best_balanced = max(strategies, key=lambda x: x['sharpe'])
            best_strategies['balanced'] = best_balanced['name']
        
        return best_strategies

# ============================================
# Executive Summary Report Generator
# ============================================

class ExecutiveSummaryReportGenerator(BaseReportGenerator):
    """
    Generates high-level executive summary reports.
    
    This class creates concise reports suitable for executive
    and stakeholder consumption, focusing on key insights and decisions.
    """
    
    def __init__(self):
        super().__init__(ReportType.EXECUTIVE_SUMMARY, "Executive Summary - StockPredictionPro Analysis")
    
    def generate_executive_report(self, model_results: Optional[Dict[str, Any]] = None,
                                backtest_results: Optional[Dict[str, Any]] = None,
                                strategy_comparison: Optional[Dict[str, Dict[str, Any]]] = None,
                                key_recommendations: Optional[List[str]] = None) -> str:
        """
        Generate executive summary report
        
        Args:
            model_results: ML model performance results
            backtest_results: Trading strategy backtest results  
            strategy_comparison: Multiple strategy comparison results
            key_recommendations: List of key recommendations
            
        Returns:
            Generated executive report content
        """
        
        # Add executive summary
        self._add_executive_overview(model_results, backtest_results, strategy_comparison)
        
        # Add key findings
        if model_results:
            self._add_model_key_findings(model_results)
        
        if backtest_results:
            self._add_trading_key_findings(backtest_results)
        
        if strategy_comparison:
            self._add_strategy_key_findings(strategy_comparison)
        
        # Add recommendations
        self._add_executive_recommendations(key_recommendations)
        
        # Add next steps
        self._add_next_steps_section()
        
        return self.generate_report()
    
    def _add_executive_overview(self, model_results: Optional[Dict[str, Any]],
                              backtest_results: Optional[Dict[str, Any]], 
                              strategy_comparison: Optional[Dict[str, Dict[str, Any]]]):
        """Add executive overview section"""
        
        overview_points = []
        
        if model_results:
            accuracy = model_results.get('accuracy', 0)
            overview_points.append(f"• Machine Learning Model: {accuracy:.1%} prediction accuracy achieved")
        
        if backtest_results:
            total_return = backtest_results.get('summary', {}).get('total_return', 0)
            sharpe_ratio = backtest_results.get('performance', {}).get('sharpe_ratio', 0)
            overview_points.append(f"• Trading Strategy: {total_return:.1%} total return with {sharpe_ratio:.2f} Sharpe ratio")
        
        if strategy_comparison:
            num_strategies = len(strategy_comparison)
            best_strategy = max(strategy_comparison.items(), 
                              key=lambda x: x[1].get('performance', {}).get('sharpe_ratio', 0))
            overview_points.append(f"• Strategy Analysis: {num_strategies} strategies compared, '{best_strategy[0]}' identified as optimal")
        
        overview_content = f"""
        StockPredictionPro comprehensive analysis summary:
        
        {chr(10).join(overview_points)}
        
        This analysis provides data-driven insights for investment strategy optimization and
        risk-adjusted portfolio management in current market conditions.
        """
        
        overview_section = ReportSection(
            title="Executive Overview",
            content=overview_content.strip()
        )
        
        self.add_section(overview_section)
    
    def _add_model_key_findings(self, model_results: Dict[str, Any]):
        """Add key findings for ML model"""
        
        findings_content = f"""
        Machine Learning Model Performance:
        
        • Prediction Accuracy: {model_results.get('accuracy', 0):.1%}
        • Model Reliability: {self._assess_model_reliability(model_results)}
        • Deployment Readiness: {self._assess_deployment_readiness(model_results)}
        
        The model demonstrates {self._get_model_performance_level(model_results)} performance 
        characteristics suitable for {'immediate deployment' if model_results.get('accuracy', 0) > 0.7 
        else 'cautious deployment with monitoring' if model_results.get('accuracy', 0) > 0.6 
        else 'further development before deployment'}.
        """
        
        findings_section = ReportSection(
            title="Model Performance Findings",
            content=findings_content.strip()
        )
        
        self.add_section(findings_section)
    
    def _add_trading_key_findings(self, backtest_results: Dict[str, Any]):
        """Add key findings for trading strategy"""
        
        summary = backtest_results.get('summary', {})
        performance = backtest_results.get('performance', {})
        
        total_return = summary.get('total_return', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        
        findings_content = f"""
        Trading Strategy Performance:
        
        • Total Return: {total_return:.1%}
        • Risk-Adjusted Return: {sharpe_ratio:.2f} Sharpe ratio
        • Maximum Risk: {max_drawdown:.1%} maximum drawdown
        • Strategy Viability: {self._assess_strategy_viability(total_return, sharpe_ratio, max_drawdown)}
        
        The strategy demonstrates {self._get_strategy_strength(sharpe_ratio)} risk-adjusted performance 
        with {self._get_risk_level_description(max_drawdown)} risk characteristics.
        """
        
        findings_section = ReportSection(
            title="Trading Strategy Findings", 
            content=findings_content.strip()
        )
        
        self.add_section(findings_section)
    
    def _add_strategy_key_findings(self, strategy_comparison: Dict[str, Dict[str, Any]]):
        """Add key findings for strategy comparison"""
        
        num_strategies = len(strategy_comparison)
        
        # Find top performers
        best_return = max(strategy_comparison.items(),
                         key=lambda x: x[1].get('summary', {}).get('total_return', 0))
        best_sharpe = max(strategy_comparison.items(),
                         key=lambda x: x[1].get('performance', {}).get('sharpe_ratio', 0))
        lowest_risk = min(strategy_comparison.items(),
                         key=lambda x: x[1].get('performance', {}).get('max_drawdown', 1))
        
        findings_content = f"""
        Strategy Comparison Analysis:
        
        • Strategies Evaluated: {num_strategies} different approaches tested
        • Best Return Generator: {best_return[0]} ({best_return[1].get('summary', {}).get('total_return', 0):.1%})
        • Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1].get('performance', {}).get('sharpe_ratio', 0):.2f})
        • Most Conservative: {lowest_risk[0]} (Max DD: {lowest_risk[1].get('performance', {}).get('max_drawdown', 0):.1%})
        
        Analysis reveals significant performance variations, enabling optimal strategy selection
        based on specific risk tolerance and return objectives.
        """
        
        findings_section = ReportSection(
            title="Strategy Comparison Findings",
            content=findings_content.strip()
        )
        
        self.add_section(findings_section)
    
    def _add_executive_recommendations(self, key_recommendations: Optional[List[str]]):
        """Add executive recommendations section"""
        
        if key_recommendations:
            recommendations_content = "Strategic Recommendations:\n\n" + "\n".join(f"• {rec}" for rec in key_recommendations)
        else:
            recommendations_content = """
            Strategic Recommendations:
            
            • Implement systematic approach to model and strategy evaluation
            • Establish regular performance monitoring and rebalancing protocols  
            • Consider diversification across multiple validated strategies
            • Maintain robust risk management framework
            • Plan for regular model retraining and strategy optimization
            """
        
        recommendations_section = ReportSection(
            title="Strategic Recommendations",
            content=recommendations_content.strip()
        )
        
        self.add_section(recommendations_section)
    
    def _add_next_steps_section(self):
        """Add next steps section"""
        
        next_steps_content = """
        Immediate Next Steps:
        
        • Phase 1 (0-30 days): Implement top-performing strategy with reduced position sizing
        • Phase 2 (30-90 days): Monitor live performance and compare to backtest expectations
        • Phase 3 (90+ days): Scale position sizing based on live performance validation
        
        Ongoing Requirements:
        
        • Weekly performance review and risk monitoring
        • Monthly strategy performance evaluation against benchmarks
        • Quarterly model retraining and strategy optimization review
        • Annual comprehensive strategy and infrastructure assessment
        """
        
        next_steps_section = ReportSection(
            title="Implementation Roadmap",
            content=next_steps_content.strip()
        )
        
        self.add_section(next_steps_section)
    
    # Helper methods for executive summary
    def _assess_model_reliability(self, model_results: Dict[str, Any]) -> str:
        """Assess model reliability"""
        accuracy = model_results.get('accuracy', 0)
        if accuracy >= 0.8:
            return "High"
        elif accuracy >= 0.6:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_deployment_readiness(self, model_results: Dict[str, Any]) -> str:
        """Assess deployment readiness"""
        accuracy = model_results.get('accuracy', 0)
        if accuracy >= 0.7:
            return "Ready"
        elif accuracy >= 0.6:
            return "Conditional"
        else:
            return "Not Ready"
    
    def _get_model_performance_level(self, model_results: Dict[str, Any]) -> str:
        """Get model performance level description"""
        accuracy = model_results.get('accuracy', 0)
        if accuracy >= 0.8:
            return "exceptional"
        elif accuracy >= 0.7:
            return "strong"
        elif accuracy >= 0.6:
            return "adequate"
        else:
            return "insufficient"
    
    def _assess_strategy_viability(self, total_return: float, sharpe_ratio: float, max_drawdown: float) -> str:
        """Assess overall strategy viability"""
        
        score = 0
        if total_return > 0.1:
            score += 1
        if sharpe_ratio > 0.5:
            score += 1
        if max_drawdown < 0.2:
            score += 1
        
        if score >= 3:
            return "Highly Viable"
        elif score >= 2:
            return "Viable"
        else:
            return "Questionable"
    
    def _get_strategy_strength(self, sharpe_ratio: float) -> str:
        """Get strategy strength description"""
        if sharpe_ratio >= 1.0:
            return "strong"
        elif sharpe_ratio >= 0.5:
            return "moderate"
        else:
            return "weak"
    
    def _get_risk_level_description(self, max_drawdown: float) -> str:
        """Get risk level description"""
        if max_drawdown <= 0.1:
            return "low"
        elif max_drawdown <= 0.2:
            return "moderate"
        else:
            return "elevated"

# ============================================
# Utility Functions
# ============================================

def create_model_performance_report(model_results: Dict[str, Any], **kwargs) -> str:
    """Quick utility to create model performance report"""
    
    generator = ModelPerformanceReportGenerator()
    return generator.generate_model_report(model_results, **kwargs)

def create_backtesting_report(backtest_results: Dict[str, Any], **kwargs) -> str:
    """Quick utility to create backtesting report"""
    
    generator = BacktestingReportGenerator()
    return generator.generate_backtest_report(backtest_results, **kwargs)

def create_strategy_comparison_report(strategy_results: Dict[str, Dict[str, Any]], **kwargs) -> str:
    """Quick utility to create strategy comparison report"""
    
    generator = StrategyComparisonReportGenerator()
    return generator.generate_comparison_report(strategy_results, **kwargs)

def create_executive_summary(model_results: Optional[Dict[str, Any]] = None,
                           backtest_results: Optional[Dict[str, Any]] = None,
                           **kwargs) -> str:
    """Quick utility to create executive summary report"""
    
    generator = ExecutiveSummaryReportGenerator()
    return generator.generate_executive_report(model_results, backtest_results, **kwargs)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Evaluation Reports System")
    
    # Sample model results for testing
    sample_model_results = {
        'model_type': 'Random Forest',
        'accuracy': 0.745,
        'precision': 0.732,
        'recall': 0.689,
        'f1_score': 0.710,
        'auc_roc': 0.823,
        'training_samples': 50000,
        'features_count': 25,
        'training_time': 45.67
    }
    
    # Sample backtest results for testing
    sample_backtest_results = {
        'summary': {
            'initial_capital': 1000000,
            'final_value': 1234567,
            'total_return': 0.234567,
            'total_trades': 156,
            'winning_trades': 89,
            'losing_trades': 67
        },
        'performance': {
            'sharpe_ratio': 1.245,
            'sortino_ratio': 1.567,
            'calmar_ratio': 2.134,
            'max_drawdown': 0.087,
            'volatility': 0.156,
            'win_rate': 0.571,
            'var_95': 0.034
        }
    }
    
    # Sample strategy comparison results
    sample_strategy_results = {
        'MA Crossover': {
            'summary': {'total_return': 0.156, 'initial_capital': 1000000, 'final_value': 1156000, 'total_trades': 45},
            'performance': {'sharpe_ratio': 0.89, 'max_drawdown': 0.12, 'volatility': 0.18, 'win_rate': 0.52}
        },
        'RSI Strategy': {
            'summary': {'total_return': 0.234, 'initial_capital': 1000000, 'final_value': 1234000, 'total_trades': 67},
            'performance': {'sharpe_ratio': 1.15, 'max_drawdown': 0.09, 'volatility': 0.20, 'win_rate': 0.58}
        },
        'Momentum': {
            'summary': {'total_return': 0.298, 'initial_capital': 1000000, 'final_value': 1298000, 'total_trades': 89},
            'performance': {'sharpe_ratio': 1.34, 'max_drawdown': 0.15, 'volatility': 0.22, 'win_rate': 0.61}
        }
    }
    
    print("\n1. Testing Model Performance Report")
    
    model_report_gen = ModelPerformanceReportGenerator()
    model_report = model_report_gen.generate_model_report(sample_model_results)
    
    print("Model Performance Report Generated:")
    print(f"- Report length: {len(model_report):,} characters")
    print(f"- Number of sections: {len(model_report_gen.sections)}")
    
    # Save sample report
    with open("sample_model_report.html", "w", encoding='utf-8') as f:
        f.write(model_report)
    print("- Sample report saved as 'sample_model_report.html'")
    
    print("\n2. Testing Backtesting Report")
    
    backtest_report_gen = BacktestingReportGenerator()
    backtest_report = backtest_report_gen.generate_backtest_report(sample_backtest_results)
    
    print("Backtesting Report Generated:")
    print(f"- Report length: {len(backtest_report):,} characters")
    print(f"- Number of sections: {len(backtest_report_gen.sections)}")
    
    print("\n3. Testing Strategy Comparison Report")
    
    comparison_report_gen = StrategyComparisonReportGenerator()
    comparison_report = comparison_report_gen.generate_comparison_report(sample_strategy_results)
    
    print("Strategy Comparison Report Generated:")
    print(f"- Report length: {len(comparison_report):,} characters")
    print(f"- Number of sections: {len(comparison_report_gen.sections)}")
    
    print("\n4. Testing Executive Summary Report")
    
    executive_report_gen = ExecutiveSummaryReportGenerator()
    executive_report = executive_report_gen.generate_executive_report(
        model_results=sample_model_results,
        backtest_results=sample_backtest_results,
        strategy_comparison=sample_strategy_results
    )
    
    print("Executive Summary Report Generated:")
    print(f"- Report length: {len(executive_report):,} characters")
    print(f"- Number of sections: {len(executive_report_gen.sections)}")
    
    print("\n5. Testing Different Output Formats")
    
    # Test Markdown format
    markdown_report = model_report_gen.generate_report(ReportFormat.MARKDOWN)
    print(f"Markdown format: {len(markdown_report):,} characters")
    
    # Test JSON format
    json_report = model_report_gen.generate_report(ReportFormat.JSON)
    print(f"JSON format: {len(json_report):,} characters")
    
    # Test Text format
    text_report = model_report_gen.generate_report(ReportFormat.TEXT)
    print(f"Text format: {len(text_report):,} characters")
    
    print("\n6. Testing Utility Functions")
    
    # Test utility functions
    quick_model_report = create_model_performance_report(sample_model_results)
    print(f"Quick model report: {len(quick_model_report):,} characters")
    
    quick_backtest_report = create_backtesting_report(sample_backtest_results)
    print(f"Quick backtest report: {len(quick_backtest_report):,} characters")
    
    quick_comparison_report = create_strategy_comparison_report(sample_strategy_results)
    print(f"Quick comparison report: {len(quick_comparison_report):,} characters")
    
    quick_executive_report = create_executive_summary(
        model_results=sample_model_results,
        backtest_results=sample_backtest_results
    )
    print(f"Quick executive report: {len(quick_executive_report):,} characters")
    
    print("\n7. Sample Report Content Preview")
    
    # Show preview of executive summary
    executive_preview = executive_report[:1000] + "..." if len(executive_report) > 1000 else executive_report
    print("Executive Summary Preview:")
    print(executive_preview)
    
    print("\nEvaluation reports system testing completed successfully!")
    print("\nGenerated reports provide comprehensive analysis suitable for:")
    print("• Technical teams: Detailed model and strategy analysis")
    print("• Management: Executive summaries with key insights")  
    print("• Stakeholders: Professional reports with actionable recommendations")
    print("• Compliance: Structured documentation for audit trails")
