"""
scripts/utilities/generate_docs.py

Automated documentation generation system for StockPredictionPro.
Creates comprehensive documentation from code, docstrings, markdown files,
and API endpoints with support for multiple output formats.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import ast
import inspect
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import subprocess
import tempfile

# Documentation libraries
try:
    import markdown
    from markdown.extensions import codehilite, toc, tables
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'docs_generation_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.DocsGeneration')

# Directory configuration
PROJECT_ROOT = Path('.')
DOCS_DIR = PROJECT_ROOT / 'docs'
OUTPUT_DIR = DOCS_DIR / 'generated'
TEMPLATES_DIR = DOCS_DIR / 'templates'
API_DIR = PROJECT_ROOT / 'api'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

# Ensure directories exist
for dir_path in [DOCS_DIR, OUTPUT_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class DocsConfig:
    """Configuration for documentation generation"""
    # Output settings
    output_formats: List[str] = None  # html, markdown
    output_directory: str = 'docs/generated'
    
    # Content settings
    include_api_docs: bool = True
    include_code_docs: bool = True
    include_tutorials: bool = True
    include_examples: bool = True
    include_changelog: bool = True
    
    # Code analysis settings
    analyze_directories: List[str] = None
    exclude_patterns: List[str] = None
    min_docstring_length: int = 10
    
    # API documentation
    api_base_url: str = 'http://localhost:8000'
    api_endpoints_file: Optional[str] = None
    generate_api_examples: bool = True
    
    # Styling and formatting
    theme: str = 'default'
    custom_css: Optional[str] = None
    logo_path: Optional[str] = None
    
    # Project metadata
    project_name: str = 'StockPredictionPro'
    project_version: str = '1.0.0'
    author: str = 'StockPredictionPro Team'
    description: str = 'Advanced stock prediction platform'
    
    # Advanced settings
    generate_diagrams: bool = False
    include_metrics: bool = True
    auto_generate_examples: bool = True
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['html', 'markdown']
        if self.analyze_directories is None:
            self.analyze_directories = ['scripts', 'api']
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                '__pycache__', '*.pyc', '.git', '.pytest_cache',
                'node_modules', '*.log', 'temp', 'tmp'
            ]

@dataclass
class DocItem:
    """Individual documentation item"""
    name: str
    type: str  # module, class, function, endpoint
    description: str
    source_file: str
    line_number: int
    docstring: Optional[str] = None
    parameters: List[Dict[str, Any]] = None
    returns: Optional[str] = None
    examples: List[str] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.examples is None:
            self.examples = []

@dataclass
class DocsGenerationResult:
    """Results from documentation generation"""
    generation_timestamp: str
    output_formats: List[str]
    files_generated: List[str]
    modules_documented: int
    functions_documented: int
    classes_documented: int
    api_endpoints_documented: int
    generation_time: float
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CodeAnalyzer:
    """Analyze Python code for documentation"""
    
    def __init__(self, config: DocsConfig):
        self.config = config
    
    def analyze_project(self) -> List[DocItem]:
        """Analyze entire project for documentation"""
        logger.info("Analyzing project code for documentation...")
        
        doc_items = []
        
        for directory in self.config.analyze_directories:
            dir_path = PROJECT_ROOT / directory
            
            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
                continue
            
            logger.info(f"Analyzing directory: {dir_path}")
            
            for python_file in dir_path.rglob('*.py'):
                if self._should_exclude_file(python_file):
                    continue
                
                try:
                    file_items = self._analyze_python_file(python_file)
                    doc_items.extend(file_items)
                except Exception as e:
                    logger.error(f"Failed to analyze {python_file}: {e}")
        
        logger.info(f"Found {len(doc_items)} documentable items")
        return doc_items
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded"""
        file_str = str(file_path)
        
        for pattern in self.config.exclude_patterns:
            if pattern.startswith('*'):
                if file_str.endswith(pattern[1:]):
                    return True
            elif pattern in file_str:
                return True
        
        return False
    
    def _analyze_python_file(self, file_path: Path) -> List[DocItem]:
        """Analyze single Python file"""
        doc_items = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            tree = ast.parse(source_code)
            
            # Extract module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring and len(module_docstring) >= self.config.min_docstring_length:
                doc_items.append(DocItem(
                    name=file_path.stem,
                    type='module',
                    description=module_docstring.split('\n')[0],
                    source_file=str(file_path),
                    line_number=1,
                    docstring=module_docstring
                ))
            
            # Analyze classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_item = self._analyze_class(node, file_path)
                    if class_item:
                        doc_items.append(class_item)
                
                elif isinstance(node, ast.FunctionDef):
                    func_item = self._analyze_function(node, file_path)
                    if func_item:
                        doc_items.append(func_item)
        
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
        
        return doc_items
    
    def _analyze_class(self, node: ast.ClassDef, file_path: Path) -> Optional[DocItem]:
        """Analyze class definition"""
        docstring = ast.get_docstring(node)
        
        if not docstring or len(docstring) < self.config.min_docstring_length:
            return None
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = ast.get_docstring(item)
                if method_doc:
                    methods.append({
                        'name': item.name,
                        'docstring': method_doc.split('\n')[0],
                        'line': item.lineno
                    })
        
        return DocItem(
            name=node.name,
            type='class',
            description=docstring.split('\n')[0],
            source_file=str(file_path),
            line_number=node.lineno,
            docstring=docstring,
            parameters=methods
        )
    
    def _analyze_function(self, node: ast.FunctionDef, file_path: Path) -> Optional[DocItem]:
        """Analyze function definition"""
        docstring = ast.get_docstring(node)
        
        if not docstring or len(docstring) < self.config.min_docstring_length:
            return None
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            parameters.append({
                'name': arg.arg,
                'type': 'Any',
                'description': ''
            })
        
        # Extract return type info from docstring
        returns = None
        if 'Returns:' in docstring or 'return' in docstring.lower():
            returns = self._extract_returns_info(docstring)
        
        return DocItem(
            name=node.name,
            type='function',
            description=docstring.split('\n')[0],
            source_file=str(file_path),
            line_number=node.lineno,
            docstring=docstring,
            parameters=parameters,
            returns=returns
        )
    
    def _extract_returns_info(self, docstring: str) -> str:
        """Extract return information from docstring"""
        lines = docstring.split('\n')
        
        for i, line in enumerate(lines):
            if 'Returns:' in line or 'Return:' in line:
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
        
        return "Return value"

class APIAnalyzer:
    """Analyze API endpoints for documentation"""
    
    def __init__(self, config: DocsConfig):
        self.config = config
    
    def analyze_api_endpoints(self) -> List[DocItem]:
        """Analyze API endpoints"""
        logger.info("Analyzing API endpoints...")
        
        doc_items = []
        
        try:
            # Try to get endpoints from OpenAPI/Swagger
            if HAS_REQUESTS:
                endpoints = self._get_openapi_endpoints()
                if endpoints:
                    doc_items.extend(endpoints)
                    return doc_items
            
            # Fallback: analyze FastAPI files
            endpoints = self._analyze_fastapi_files()
            doc_items.extend(endpoints)
            
        except Exception as e:
            logger.error(f"API analysis failed: {e}")
        
        return doc_items
    
    def _get_openapi_endpoints(self) -> List[DocItem]:
        """Get endpoints from OpenAPI/Swagger spec"""
        try:
            response = requests.get(f"{self.config.api_base_url}/openapi.json", timeout=10)
            
            if response.status_code == 200:
                openapi_spec = response.json()
                return self._parse_openapi_spec(openapi_spec)
            
        except Exception as e:
            logger.debug(f"Could not fetch OpenAPI spec: {e}")
        
        return []
    
    def _parse_openapi_spec(self, spec: Dict[str, Any]) -> List[DocItem]:
        """Parse OpenAPI specification"""
        doc_items = []
        
        paths = spec.get('paths', {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    
                    doc_item = DocItem(
                        name=f"{method.upper()} {path}",
                        type='endpoint',
                        description=details.get('summary', ''),
                        source_file='API',
                        line_number=0,
                        docstring=details.get('description', ''),
                        parameters=self._extract_api_parameters(details),
                        returns=self._extract_api_responses(details)
                    )
                    
                    # Generate examples
                    if self.config.generate_api_examples:
                        examples = self._generate_api_examples(path, method, details)
                        doc_item.examples = examples
                    
                    doc_items.append(doc_item)
        
        return doc_items
    
    def _analyze_fastapi_files(self) -> List[DocItem]:
        """Analyze FastAPI files directly"""
        doc_items = []
        
        if not API_DIR.exists():
            return doc_items
        
        for python_file in API_DIR.rglob('*.py'):
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for FastAPI route decorators
                routes = self._extract_fastapi_routes(content, python_file)
                doc_items.extend(routes)
                
            except Exception as e:
                logger.error(f"Failed to analyze API file {python_file}: {e}")
        
        return doc_items
    
    def _extract_fastapi_routes(self, content: str, file_path: Path) -> List[DocItem]:
        """Extract FastAPI routes from file content"""
        doc_items = []
        
        # Simple regex patterns for FastAPI routes
        patterns = [
            r'@app\.(get|post|put|delete|patch)\([\'"](.*?)[\'"].*?\)',
            r'@router\.(get|post|put|delete|patch)\([\'"](.*?)[\'"].*?\)'
        ]
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line.strip())
                if match:
                    method, path = match.groups()
                    
                    # Look for function definition in following lines
                    func_name = ''
                    docstring = ''
                    
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if lines[j].strip().startswith('def '):
                            func_name = lines[j].strip().split('(')[0].replace('def ', '')
                            
                            # Extract docstring
                            docstring = self._extract_function_docstring(lines, j + 1)
                            break
                    
                    if func_name:
                        doc_item = DocItem(
                            name=f"{method.upper()} {path}",
                            type='endpoint',
                            description=docstring.split('\n')[0] if docstring else f"{method.upper()} {path}",
                            source_file=str(file_path),
                            line_number=i + 1,
                            docstring=docstring
                        )
                        doc_items.append(doc_item)
        
        return doc_items
    
    def _extract_function_docstring(self, lines: List[str], start_idx: int) -> str:
        """Extract docstring from function"""
        docstring_lines = []
        in_docstring = False
        quote_type = None
        
        for i in range(start_idx, min(start_idx + 20, len(lines))):
            line = lines[i].strip()
            
            if not in_docstring:
                if line.startswith('"""') or line.startswith("'''"):
                    quote_type = line[:3]
                    in_docstring = True
                    
                    # Check if docstring is on same line
                    if line.count(quote_type) == 2:
                        return line[3:-3].strip()
                    
                    # Multi-line docstring
                    if len(line) > 3:
                        docstring_lines.append(line[3:])
                    
                elif line.startswith('"') or line.startswith("'"):
                    quote_type = line[0]
                    if line.count(quote_type) >= 2:
                        return line[1:-1].strip()
            
            else:
                if quote_type in line:
                    # End of docstring
                    docstring_lines.append(line.replace(quote_type, ''))
                    break
                else:
                    docstring_lines.append(line)
        
        return '\n'.join(docstring_lines).strip()
    
    def _extract_api_parameters(self, details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract API parameters from OpenAPI spec"""
        parameters = []
        
        # Path parameters
        for param in details.get('parameters', []):
            parameters.append({
                'name': param.get('name', ''),
                'type': param.get('schema', {}).get('type', 'string'),
                'description': param.get('description', ''),
                'required': param.get('required', False),
                'location': param.get('in', 'query')
            })
        
        # Request body
        request_body = details.get('requestBody', {})
        if request_body:
            content = request_body.get('content', {})
            for media_type, schema in content.items():
                parameters.append({
                    'name': 'request_body',
                    'type': media_type,
                    'description': request_body.get('description', ''),
                    'required': request_body.get('required', False),
                    'location': 'body'
                })
        
        return parameters
    
    def _extract_api_responses(self, details: Dict[str, Any]) -> str:
        """Extract API response information"""
        responses = details.get('responses', {})
        
        success_responses = []
        for status_code, response in responses.items():
            if status_code.startswith('2'):  # 2xx success codes
                description = response.get('description', f'HTTP {status_code}')
                success_responses.append(description)
        
        return '; '.join(success_responses) if success_responses else 'Response'
    
    def _generate_api_examples(self, path: str, method: str, details: Dict[str, Any]) -> List[str]:
        """Generate API usage examples"""
        examples = []
        
        # cURL example
        curl_example = f"curl -X {method.upper()} \\\n  {self.config.api_base_url}{path}"
        
        # Add parameters if any
        parameters = details.get('parameters', [])
        if parameters:
            query_params = [p for p in parameters if p.get('in') == 'query']
            if query_params:
                params_str = '&'.join([f"{p['name']}=value" for p in query_params])
                curl_example += f"?{params_str}"
        
        # Add headers
        curl_example += " \\\n  -H 'Content-Type: application/json'"
        
        examples.append(f"``````")
        
        # Python requests example
        python_example = f"""```
import requests

response = requests.{method.lower()}(
    '{self.config.api_base_url}{path}',
    headers={{'Content-Type': 'application/json'}}
)
print(response.json())
```"""
        
        examples.append(python_example)
        
        return examples

class MarkdownGenerator:
    """Generate Markdown documentation"""
    
    def __init__(self, config: DocsConfig):
        self.config = config
    
    def generate_documentation(self, doc_items: List[DocItem]) -> List[str]:
        """Generate Markdown documentation"""
        logger.info("Generating Markdown documentation...")
        
        generated_files = []
        
        # Group items by type
        modules = [item for item in doc_items if item.type == 'module']
        classes = [item for item in doc_items if item.type == 'class']
        functions = [item for item in doc_items if item.type == 'function']
        endpoints = [item for item in doc_items if item.type == 'endpoint']
        
        # Generate main index
        index_content = self._generate_index(modules, classes, functions, endpoints)
        index_file = OUTPUT_DIR / 'README.md'
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        generated_files.append(str(index_file))
        
        # Generate API documentation
        if endpoints:
            api_content = self._generate_api_docs(endpoints)
            api_file = OUTPUT_DIR / 'api.md'
            with open(api_file, 'w', encoding='utf-8') as f:
                f.write(api_content)
            generated_files.append(str(api_file))
        
        # Generate code documentation
        if classes or functions:
            code_content = self._generate_code_docs(classes, functions)
            code_file = OUTPUT_DIR / 'code.md'
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            generated_files.append(str(code_file))
        
        # Generate modules documentation
        if modules:
            modules_content = self._generate_modules_docs(modules)
            modules_file = OUTPUT_DIR / 'modules.md'
            with open(modules_file, 'w', encoding='utf-8') as f:
                f.write(modules_content)
            generated_files.append(str(modules_file))
        
        return generated_files
    
    def _generate_index(self, modules: List[DocItem], classes: List[DocItem], 
                       functions: List[DocItem], endpoints: List[DocItem]) -> str:
        """Generate main index page"""
        content = f"""# {self.config.project_name} Documentation

**Version:** {self.config.project_version}  
**Author:** {self.config.author}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

{self.config.description}

## Documentation Sections

### 📚 [Modules](modules.md)
Detailed documentation for all project modules.
- **{len(modules)} modules** documented

### 🔧 [Code Reference](code.md)
Classes and functions documentation.
- **{len(classes)} classes** documented
- **{len(functions)} functions** documented

### 🌐 [API Reference](api.md)
REST API endpoints and usage examples.
- **{len(endpoints)} endpoints** documented

## Quick Start

### Installation

pip install -r requirements.txt


### Basic Usage

from scripts.models.train_all_models import ModelTrainer

Initialize trainer
trainer = ModelTrainer()

Train models
results = trainer.train_all_models('data/processed/stock_data.csv')


### API Usage

Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Make a prediction request
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{{"symbol": "AAPL", "features": [...]}}'


## Project Structure

stockpredictionpro/
├── scripts/ # Core functionality
├── config/ # Configuration files
├── deployment/ # Deployment scripts
├── app/ # Application code
├── monitoring/ # Monitoring
├── logs/ # Log files
├── notebooks/ # Jupyter notebooks
├── security/ # Security-related files
├── data/ # Data storage
├── src/ # Source code
├── docs/ # Documentation
└── tests/ # Test files



## Contributing

Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
        return content
    
    def _generate_api_docs(self, endpoints: List[DocItem]) -> str:
        """Generate API documentation"""
        content = f"""# API Reference

Base URL: `{self.config.api_base_url}`

## Endpoints

"""
        
        # Group by path prefix
        grouped_endpoints = {}
        for endpoint in endpoints:
            path = endpoint.name.split(' ')[1] if ' ' in endpoint.name else endpoint.name
            prefix = path.split('/')[1] if '/' in path else 'root'
            
            if prefix not in grouped_endpoints:
                grouped_endpoints[prefix] = []
            grouped_endpoints[prefix].append(endpoint)
        
        for prefix, group_endpoints in grouped_endpoints.items():
            content += f"### {prefix.title()} Endpoints\n\n"
            
            for endpoint in group_endpoints:
                content += self._format_endpoint_docs(endpoint)
                content += "\n---\n\n"
        
        return content
    
    def _format_endpoint_docs(self, endpoint: DocItem) -> str:
        """Format single endpoint documentation"""
        content = f"#### {endpoint.name}\n\n"
        
        if endpoint.description:
            content += f"{endpoint.description}\n\n"
        
        if endpoint.docstring and endpoint.docstring != endpoint.description:
            content += f"**Description:**\n{endpoint.docstring}\n\n"
        
        # Parameters
        if endpoint.parameters:
            content += "**Parameters:**\n\n"
            content += "| Name | Type | Location | Required | Description |\n"
            content += "|------|------|----------|----------|-------------|\n"
            
            for param in endpoint.parameters:
                required = "✓" if param.get('required', False) else "✗"
                content += f"| `{param['name']}` | `{param['type']}` | {param.get('location', 'query')} | {required} | {param.get('description', '')} |\n"
            
            content += "\n"
        
        # Response
        if endpoint.returns:
            content += f"**Returns:** {endpoint.returns}\n\n"
        
        # Examples
        if endpoint.examples:
            content += "**Examples:**\n\n"
            for example in endpoint.examples:
                content += f"{example}\n\n"
        
        return content
    
    def _generate_code_docs(self, classes: List[DocItem], functions: List[DocItem]) -> str:
        """Generate code documentation"""
        content = "# Code Reference\n\n"
        
        if classes:
            content += "## Classes\n\n"
            
            for cls in sorted(classes, key=lambda x: x.name):
                content += f"### {cls.name}\n\n"
                content += f"**File:** `{cls.source_file}:{cls.line_number}`\n\n"
                
                if cls.description:
                    content += f"{cls.description}\n\n"
                
                if cls.docstring and cls.docstring != cls.description:
                    content += f"``````\n\n"
                
                # Methods
                if cls.parameters:  # Parameters contain methods for classes
                    content += "**Methods:**\n\n"
                    for method in cls.parameters:
                        content += f"- `{method['name']}()` - {method.get('docstring', '')}\n"
                    content += "\n"
                
                content += "---\n\n"
        
        if functions:
            content += "## Functions\n\n"
            
            for func in sorted(functions, key=lambda x: x.name):
                content += f"### {func.name}\n\n"
                content += f"**File:** `{func.source_file}:{func.line_number}`\n\n"
                
                if func.description:
                    content += f"{func.description}\n\n"
                
                # Parameters
                if func.parameters:
                    content += "**Parameters:**\n\n"
                    for param in func.parameters:
                        content += f"- `{param['name']}` ({param.get('type', 'Any')}): {param.get('description', '')}\n"
                    content += "\n"
                
                # Returns
                if func.returns:
                    content += f"**Returns:** {func.returns}\n\n"
                
                if func.docstring and func.docstring != func.description:
                    content += f"``````\n\n"
                
                content += "---\n\n"
        
        return content
    
    def _generate_modules_docs(self, modules: List[DocItem]) -> str:
        """Generate modules documentation"""
        content = "# Modules Reference\n\n"
        
        for module in sorted(modules, key=lambda x: x.name):
            content += f"## {module.name}\n\n"
            content += f"**File:** `{module.source_file}`\n\n"
            
            if module.description:
                content += f"{module.description}\n\n"
            
            if module.docstring and module.docstring != module.description:
                content += f"``````\n\n"
            
            content += "---\n\n"
        
        return content

class HTMLGenerator:
    """Generate HTML documentation"""
    
    def __init__(self, config: DocsConfig):
        self.config = config
    
    def generate_documentation(self, markdown_files: List[str]) -> List[str]:
        """Convert Markdown files to HTML"""
        if not HAS_MARKDOWN:
            logger.warning("Markdown library not available for HTML generation")
            return []
        
        logger.info("Generating HTML documentation...")
        
        html_files = []
        
        # Setup markdown processor
        md = markdown.Markdown(
            extensions=['codehilite', 'toc', 'tables', 'fenced_code'],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': True
                },
                'toc': {
                    'permalink': True
                }
            }
        )
        
        for md_file in markdown_files:
            try:
                # Read markdown content
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # Convert to HTML
                html_content = md.convert(md_content)
                
                # Wrap in HTML template
                full_html = self._create_html_template(html_content, Path(md_file).stem)
                
                # Write HTML file
                html_file = Path(md_file).with_suffix('.html')
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(full_html)
                
                html_files.append(str(html_file))
                logger.info(f"Generated HTML: {html_file}")
                
            except Exception as e:
                logger.error(f"Failed to convert {md_file} to HTML: {e}")
        
        return html_files
    
    def _create_html_template(self, content: str, title: str) -> str:
        """Create HTML template"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {self.config.project_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 2em;
        }}
        
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        
        h2 {{
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }}
        
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        
        .highlight {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        
        .toc {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 2em;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 15px;
            font-style: italic;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .navigation {{
            background-color: #2c3e50;
            color: white;
            padding: 10px 0;
            margin: -20px -20px 20px -20px;
            padding-left: 20px;
        }}
        
        .navigation a {{
            color: white;
            margin-right: 20px;
        }}
    </style>
</head>
<body>
    <div class="navigation">
        <a href="README.html">Home</a>
        <a href="modules.html">Modules</a>
        <a href="code.html">Code</a>
        <a href="api.html">API</a>
    </div>
    
    {content}
    
    <footer style="margin-top: 3em; padding-top: 2em; border-top: 1px solid #ecf0f1; text-align: center; color: #7f8c8d;">
        <p>Generated by {self.config.project_name} Documentation System</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>"""

class DocsOrchestrator:
    """Main orchestrator for documentation generation"""
    
    def __init__(self, config: DocsConfig):
        self.config = config
        self.code_analyzer = CodeAnalyzer(config)
        self.api_analyzer = APIAnalyzer(config)
        self.markdown_generator = MarkdownGenerator(config)
        self.html_generator = HTMLGenerator(config)
    
    def generate_documentation(self) -> DocsGenerationResult:
        """Generate comprehensive documentation"""
        logger.info("🚀 Starting documentation generation...")
        start_time = datetime.now()
        
        try:
            # Clean output directory
            if OUTPUT_DIR.exists():
                import shutil
                shutil.rmtree(OUTPUT_DIR)
            OUTPUT_DIR.mkdir(parents=True)
            
            # Collect all documentation items
            all_doc_items = []
            warnings = []
            errors = []
            
            # Analyze code
            if self.config.include_code_docs:
                try:
                    code_items = self.code_analyzer.analyze_project()
                    all_doc_items.extend(code_items)
                    logger.info(f"Found {len(code_items)} code documentation items")
                except Exception as e:
                    error_msg = f"Code analysis failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Analyze API
            if self.config.include_api_docs:
                try:
                    api_items = self.api_analyzer.analyze_api_endpoints()
                    all_doc_items.extend(api_items)
                    logger.info(f"Found {len(api_items)} API documentation items")
                except Exception as e:
                    error_msg = f"API analysis failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Generate documentation in requested formats
            generated_files = []
            
            # Generate Markdown
            if 'markdown' in self.config.output_formats:
                try:
                    md_files = self.markdown_generator.generate_documentation(all_doc_items)
                    generated_files.extend(md_files)
                    logger.info(f"Generated {len(md_files)} Markdown files")
                except Exception as e:
                    error_msg = f"Markdown generation failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Generate HTML
            if 'html' in self.config.output_formats:
                try:
                    # Find markdown files to convert
                    md_files = [f for f in generated_files if f.endswith('.md')]
                    html_files = self.html_generator.generate_documentation(md_files)
                    generated_files.extend(html_files)
                    logger.info(f"Generated {len(html_files)} HTML files")
                except Exception as e:
                    error_msg = f"HTML generation failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Calculate statistics
            modules_count = len([item for item in all_doc_items if item.type == 'module'])
            functions_count = len([item for item in all_doc_items if item.type == 'function'])
            classes_count = len([item for item in all_doc_items if item.type == 'class'])
            endpoints_count = len([item for item in all_doc_items if item.type == 'endpoint'])
            
            # Create result
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            result = DocsGenerationResult(
                generation_timestamp=start_time.isoformat(),
                output_formats=self.config.output_formats,
                files_generated=generated_files,
                modules_documented=modules_count,
                functions_documented=functions_count,
                classes_documented=classes_count,
                api_endpoints_documented=endpoints_count,
                generation_time=generation_time,
                warnings=warnings,
                errors=errors
            )
            
            # Save generation report
            self._save_generation_report(result)
            
            # Print summary
            self._print_generation_summary(result)
            
            logger.info("✅ Documentation generation completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Documentation generation failed: {e}")
            
            return DocsGenerationResult(
                generation_timestamp=start_time.isoformat(),
                output_formats=self.config.output_formats,
                files_generated=[],
                modules_documented=0,
                functions_documented=0,
                classes_documented=0,
                api_endpoints_documented=0,
                generation_time=(datetime.now() - start_time).total_seconds(),
                errors=[str(e)]
            )
    
    def _save_generation_report(self, result: DocsGenerationResult) -> None:
        """Save generation report"""
        try:
            report_file = OUTPUT_DIR / 'generation_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info(f"Generation report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save generation report: {e}")
    
    def _print_generation_summary(self, result: DocsGenerationResult) -> None:
        """Print generation summary"""
        print("\n" + "="*60)
        print("DOCUMENTATION GENERATION SUMMARY")
        print("="*60)
        print(f"Generation Time: {result.generation_time:.1f} seconds")
        print(f"Output Formats: {', '.join(result.output_formats)}")
        
        print(f"\nDocumentation Items:")
        print(f"  📚 Modules: {result.modules_documented}")
        print(f"  🏗️  Classes: {result.classes_documented}")
        print(f"  🔧 Functions: {result.functions_documented}")
        print(f"  🌐 API Endpoints: {result.api_endpoints_documented}")
        
        print(f"\nGenerated Files ({len(result.files_generated)}):")
        for file_path in result.files_generated:
            print(f"  ✅ {file_path}")
        
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  ⚠️ {warning}")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  ❌ {error}")
        
        print(f"\n📁 Documentation available in: {OUTPUT_DIR}")

def load_config_from_file(config_path: str) -> DocsConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return DocsConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return DocsConfig()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate documentation for StockPredictionPro')
    parser.add_argument('--config', help='Path to documentation configuration JSON file')
    parser.add_argument('--formats', nargs='+', choices=['html', 'markdown'],
                       default=['html', 'markdown'], help='Output formats')
    parser.add_argument('--output-dir', default='docs/generated', help='Output directory')
    parser.add_argument('--project-name', default='StockPredictionPro', help='Project name')
    parser.add_argument('--version', default='1.0.0', help='Project version')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--no-api', action='store_true', help='Skip API documentation')
    parser.add_argument('--no-code', action='store_true', help='Skip code documentation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = DocsConfig()
    
    # Override config with command line arguments
    config.output_formats = args.formats
    config.output_directory = args.output_dir
    config.project_name = args.project_name
    config.project_version = args.version
    config.api_base_url = args.api_url
    
    if args.no_api:
        config.include_api_docs = False
    if args.no_code:
        config.include_code_docs = False
    
    try:
        # Generate documentation
        orchestrator = DocsOrchestrator(config)
        result = orchestrator.generate_documentation()
        
        if result.errors:
            print(f"\n❌ Documentation generation completed with {len(result.errors)} errors")
            sys.exit(1)
        elif result.warnings:
            print(f"\n⚠️ Documentation generation completed with {len(result.warnings)} warnings")
            sys.exit(0)
        else:
            print(f"\n✅ Documentation generation completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Documentation generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Documentation generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()