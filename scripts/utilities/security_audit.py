"""
scripts/utilities/security_audit.py

Comprehensive security auditing system for StockPredictionPro.
Performs vulnerability scanning, code analysis, dependency checks,
configuration audits, and generates detailed security reports.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import subprocess
import hashlib
import re
import ast
import socket
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import tempfile
from urllib.parse import urlparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Security scanning libraries
try:
    import bandit
    from bandit.core import manager as bandit_manager
    HAS_BANDIT = True
except ImportError:
    HAS_BANDIT = False

try:
    import safety
    HAS_SAFETY = True
except ImportError:
    HAS_SAFETY = False

try:
    import semgrep
    HAS_SEMGREP = True
except ImportError:
    HAS_SEMGREP = False

try:
    import cryptography
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'security_audit_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.SecurityAudit')

# Directory configuration
PROJECT_ROOT = Path('.')
AUDIT_DIR = PROJECT_ROOT / 'security_audit'
REPORTS_DIR = AUDIT_DIR / 'reports'
CONFIGS_DIR = AUDIT_DIR / 'configs'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
API_DIR = PROJECT_ROOT / 'api'
CONFIG_DIR = PROJECT_ROOT / 'config'

# Ensure directories exist
for dir_path in [AUDIT_DIR, REPORTS_DIR, CONFIGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class SecurityConfig:
    """Configuration for security audit"""
    # Scan types
    enable_code_scan: bool = True
    enable_dependency_scan: bool = True
    enable_config_scan: bool = True
    enable_network_scan: bool = True
    enable_file_permissions_scan: bool = True
    enable_secrets_scan: bool = True
    
    # Scan directories
    scan_directories: List[str] = None
    exclude_directories: List[str] = None
    exclude_files: List[str] = None
    
    # Severity levels
    min_severity: str = 'LOW'  # LOW, MEDIUM, HIGH, CRITICAL
    fail_on_high: bool = True
    fail_on_critical: bool = True
    
    # Network scanning
    scan_ports: List[int] = None
    scan_urls: List[str] = None
    check_ssl_certificates: bool = True
    
    # Secrets detection
    secrets_patterns: List[Dict[str, str]] = None
    
    # Reporting
    generate_html_report: bool = True
    generate_json_report: bool = True
    generate_sarif_report: bool = False
    
    # Third-party tools
    use_bandit: bool = True
    use_safety: bool = True
    use_semgrep: bool = False
    
    def __post_init__(self):
        if self.scan_directories is None:
            self.scan_directories = ['scripts', 'api', 'config']
        if self.exclude_directories is None:
            self.exclude_directories = [
                '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
                'node_modules', '.tox', 'htmlcov'
            ]
        if self.exclude_files is None:
            self.exclude_files = ['*.pyc', '*.pyo', '*.log', '*.tmp']
        if self.scan_ports is None:
            self.scan_ports = [22, 80, 443, 5432, 6379, 8000, 8080]
        if self.scan_urls is None:
            self.scan_urls = ['http://localhost:8000']
        if self.secrets_patterns is None:
            self.secrets_patterns = [
                {'name': 'API Key', 'pattern': r'api[_-]?key["\']?\s*[:=]\s*["\']([a-zA-Z0-9]{20,})["\']'},
                {'name': 'Secret Key', 'pattern': r'secret[_-]?key["\']?\s*[:=]\s*["\']([a-zA-Z0-9]{20,})["\']'},
                {'name': 'Password', 'pattern': r'password["\']?\s*[:=]\s*["\']([^"\']{8,})["\']'},
                {'name': 'JWT Token', 'pattern': r'jwt[_-]?token["\']?\s*[:=]\s*["\']([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)["\']'},
                {'name': 'Private Key', 'pattern': r'-----BEGIN (RSA )?PRIVATE KEY-----'},
                {'name': 'AWS Access Key', 'pattern': r'AKIA[0-9A-Z]{16}'},
                {'name': 'Database URL', 'pattern': r'(postgresql|mysql|mongodb)://[^:]+:[^@]+@[^/]+'}
            ]

@dataclass
class SecurityFinding:
    """Individual security finding"""
    id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # CODE, DEPENDENCY, CONFIG, NETWORK, SECRETS, PERMISSIONS
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    evidence: Optional[str] = None
    remediation: Optional[str] = None
    references: List[str] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []

@dataclass
class SecurityReport:
    """Complete security audit report"""
    audit_id: str
    audit_timestamp: str
    config_used: SecurityConfig
    
    # Findings summary
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    info_findings: int
    
    # Findings by category
    findings: List[SecurityFinding]
    
    # Scan results
    code_scan_results: Dict[str, Any] = None
    dependency_scan_results: Dict[str, Any] = None
    network_scan_results: Dict[str, Any] = None
    config_scan_results: Dict[str, Any] = None
    
    # Overall assessment
    security_score: float = 0.0
    risk_level: str = 'UNKNOWN'
    passed_checks: int = 0
    failed_checks: int = 0
    
    # Recommendations
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ============================================
# CODE SECURITY SCANNER
# ============================================

class CodeSecurityScanner:
    """Scan code for security vulnerabilities"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def scan(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Perform code security scan"""
        logger.info("Starting code security scan...")
        
        findings = []
        scan_results = {
            'bandit_results': None,
            'semgrep_results': None,
            'manual_scan_results': None
        }
        
        # Bandit scan
        if self.config.use_bandit and HAS_BANDIT:
            try:
                bandit_findings, bandit_results = self._run_bandit_scan()
                findings.extend(bandit_findings)
                scan_results['bandit_results'] = bandit_results
            except Exception as e:
                logger.error(f"Bandit scan failed: {e}")
        
        # Semgrep scan
        if self.config.use_semgrep and HAS_SEMGREP:
            try:
                semgrep_findings = self._run_semgrep_scan()
                findings.extend(semgrep_findings)
            except Exception as e:
                logger.error(f"Semgrep scan failed: {e}")
        
        # Manual code analysis
        try:
            manual_findings = self._run_manual_code_analysis()
            findings.extend(manual_findings)
            scan_results['manual_scan_results'] = {'findings_count': len(manual_findings)}
        except Exception as e:
            logger.error(f"Manual code analysis failed: {e}")
        
        logger.info(f"Code scan completed: {len(findings)} findings")
        return findings, scan_results
    
    def _run_bandit_scan(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Run Bandit security scanner"""
        findings = []
        
        try:
            # Run bandit via subprocess for better control
            cmd = ['bandit', '-r', '-f', 'json']
            
            # Add scan directories
            for directory in self.config.scan_directories:
                if (PROJECT_ROOT / directory).exists():
                    cmd.append(str(PROJECT_ROOT / directory))
            
            # Add exclusions
            if self.config.exclude_directories:
                exclude_paths = ','.join([f"*/{d}/*" for d in self.config.exclude_directories])
                cmd.extend(['--exclude', exclude_paths])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                # Process bandit results
                for issue in bandit_data.get('results', []):
                    severity_map = {
                        'HIGH': 'HIGH',
                        'MEDIUM': 'MEDIUM', 
                        'LOW': 'LOW'
                    }
                    
                    finding = SecurityFinding(
                        id=f"BANDIT-{issue['issue_cwe']['id']}-{hash(issue['filename'] + str(issue['line_number']))}",
                        severity=severity_map.get(issue['issue_severity'], 'MEDIUM'),
                        category='CODE',
                        title=issue['issue_text'],
                        description=f"{issue['issue_text']}\n\nConfidence: {issue['issue_confidence']}",
                        file_path=issue['filename'],
                        line_number=issue['line_number'],
                        evidence=issue['code'],
                        remediation=f"Review the code at line {issue['line_number']} and apply appropriate security measures.",
                        references=[f"CWE-{issue['issue_cwe']['id']}", issue['more_info']] if issue.get('more_info') else []
                    )
                    findings.append(finding)
                
                return findings, {
                    'total_issues': len(bandit_data.get('results', [])),
                    'skipped_tests': bandit_data.get('skipped', []),
                    'generated_at': bandit_data.get('generated_at')
                }
        
        except subprocess.TimeoutExpired:
            logger.error("Bandit scan timed out")
        except json.JSONDecodeError:
            logger.error("Failed to parse Bandit output")
        except Exception as e:
            logger.error(f"Bandit scan error: {e}")
        
        return findings, {}
    
    def _run_semgrep_scan(self) -> List[SecurityFinding]:
        """Run Semgrep security scanner"""
        findings = []
        
        try:
            # Use common security rulesets
            rulesets = [
                'p/security-audit',
                'p/python',
                'p/owasp-top-ten'
            ]
            
            for ruleset in rulesets:
                cmd = [
                    'semgrep', '--config', ruleset, '--json',
                    '--exclude', ','.join(self.config.exclude_directories)
                ]
                
                # Add scan directories
                for directory in self.config.scan_directories:
                    if (PROJECT_ROOT / directory).exists():
                        cmd.append(str(PROJECT_ROOT / directory))
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.stdout:
                    semgrep_data = json.loads(result.stdout)
                    
                    for issue in semgrep_data.get('results', []):
                        severity_map = {
                            'ERROR': 'HIGH',
                            'WARNING': 'MEDIUM',
                            'INFO': 'LOW'
                        }
                        
                        finding = SecurityFinding(
                            id=f"SEMGREP-{issue['check_id']}-{hash(issue['path'] + str(issue['start']['line']))}",
                            severity=severity_map.get(issue['extra']['severity'], 'MEDIUM'),
                            category='CODE',
                            title=issue['extra']['message'],
                            description=issue['extra']['message'],
                            file_path=issue['path'],
                            line_number=issue['start']['line'],
                            column_number=issue['start']['col'],
                            evidence=issue['extra'].get('lines', ''),
                            references=[issue['check_id']]
                        )
                        findings.append(finding)
        
        except Exception as e:
            logger.error(f"Semgrep scan error: {e}")
        
        return findings
    
    def _run_manual_code_analysis(self) -> List[SecurityFinding]:
        """Run manual code security analysis"""
        findings = []
        
        try:
            # Security patterns to look for
            security_patterns = [
                {
                    'name': 'SQL Injection Risk',
                    'pattern': r'\.execute\s*\(\s*["\'].*%.*["\']',
                    'severity': 'HIGH',
                    'description': 'Potential SQL injection vulnerability - string formatting in SQL queries'
                },
                {
                    'name': 'Command Injection Risk',
                    'pattern': r'os\.system\s*\(\s*.*\+.*\)',
                    'severity': 'HIGH', 
                    'description': 'Potential command injection - user input in os.system()'
                },
                {
                    'name': 'Hardcoded Secret',
                    'pattern': r'(password|secret|key)\s*=\s*["\'][^"\']{8,}["\']',
                    'severity': 'MEDIUM',
                    'description': 'Hardcoded secrets in source code'
                },
                {
                    'name': 'Debug Mode Enabled',
                    'pattern': r'debug\s*=\s*True',
                    'severity': 'MEDIUM',
                    'description': 'Debug mode enabled - should be disabled in production'
                },
                {
                    'name': 'Insecure Random',
                    'pattern': r'random\.(random|randint)',
                    'severity': 'LOW',
                    'description': 'Using insecure random generator - use secrets module for cryptographic purposes'
                }
            ]
            
            # Scan Python files
            for directory in self.config.scan_directories:
                dir_path = PROJECT_ROOT / directory
                if not dir_path.exists():
                    continue
                
                for python_file in dir_path.rglob('*.py'):
                    if any(exclude in str(python_file) for exclude in self.config.exclude_directories):
                        continue
                    
                    try:
                        with open(python_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = content.split('\n')
                        
                        for pattern_info in security_patterns:
                            pattern = re.compile(pattern_info['pattern'], re.IGNORECASE)
                            
                            for line_num, line in enumerate(lines, 1):
                                if pattern.search(line):
                                    finding = SecurityFinding(
                                        id=f"MANUAL-{pattern_info['name'].replace(' ', '_').upper()}-{hash(str(python_file) + str(line_num))}",
                                        severity=pattern_info['severity'],
                                        category='CODE',
                                        title=pattern_info['name'],
                                        description=pattern_info['description'],
                                        file_path=str(python_file),
                                        line_number=line_num,
                                        evidence=line.strip(),
                                        remediation=f"Review line {line_num} and apply secure coding practices"
                                    )
                                    findings.append(finding)
                    
                    except Exception as e:
                        logger.warning(f"Failed to analyze {python_file}: {e}")
        
        except Exception as e:
            logger.error(f"Manual code analysis error: {e}")
        
        return findings

class DependencyScanner:
    """Scan dependencies for vulnerabilities"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def scan(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Scan dependencies for vulnerabilities"""
        logger.info("Starting dependency vulnerability scan...")
        
        findings = []
        scan_results = {
            'safety_results': None,
            'outdated_packages': None
        }
        
        # Safety scan
        if self.config.use_safety:
            try:
                safety_findings, safety_results = self._run_safety_scan()
                findings.extend(safety_findings)
                scan_results['safety_results'] = safety_results
            except Exception as e:
                logger.error(f"Safety scan failed: {e}")
        
        # Check for outdated packages
        try:
            outdated_findings, outdated_results = self._check_outdated_packages()
            findings.extend(outdated_findings)
            scan_results['outdated_packages'] = outdated_results
        except Exception as e:
            logger.error(f"Outdated packages check failed: {e}")
        
        logger.info(f"Dependency scan completed: {len(findings)} findings")
        return findings, scan_results
    
    def _run_safety_scan(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Run Safety vulnerability scanner"""
        findings = []
        
        try:
            # Run safety check
            cmd = ['safety', 'check', '--json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    
                    for vuln in safety_data:
                        severity = 'HIGH' if vuln.get('vulnerability_id', '').startswith('CVE') else 'MEDIUM'
                        
                        finding = SecurityFinding(
                            id=f"SAFETY-{vuln.get('vulnerability_id', 'UNKNOWN')}",
                            severity=severity,
                            category='DEPENDENCY',
                            title=f"Vulnerable dependency: {vuln.get('package_name')}",
                            description=vuln.get('advisory', ''),
                            evidence=f"Package: {vuln.get('package_name')} {vuln.get('installed_version')}",
                            remediation=f"Update to version {vuln.get('spec', 'latest')} or higher",
                            cve_id=vuln.get('vulnerability_id') if vuln.get('vulnerability_id', '').startswith('CVE') else None,
                            references=[vuln.get('more_info_url')] if vuln.get('more_info_url') else []
                        )
                        findings.append(finding)
                    
                    return findings, {
                        'total_vulnerabilities': len(safety_data),
                        'scan_timestamp': datetime.now().isoformat()
                    }
                
                except json.JSONDecodeError:
                    # Safety might return plain text for some outputs
                    if 'vulnerabilities found' in result.stdout.lower():
                        return findings, {'message': 'No vulnerabilities found'}
        
        except subprocess.TimeoutExpired:
            logger.error("Safety scan timed out")
        except Exception as e:
            logger.error(f"Safety scan error: {e}")
        
        return findings, {}
    
    def _check_outdated_packages(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Check for outdated packages"""
        findings = []
        
        try:
            # Get list of outdated packages
            cmd = ['pip', 'list', '--outdated', '--format=json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.stdout:
                outdated_packages = json.loads(result.stdout)
                
                for package in outdated_packages:
                    # Consider packages more than 6 months behind as medium risk
                    finding = SecurityFinding(
                        id=f"OUTDATED-{package['name']}",
                        severity='LOW',
                        category='DEPENDENCY',
                        title=f"Outdated package: {package['name']}",
                        description=f"Package {package['name']} is outdated",
                        evidence=f"Current: {package['version']}, Latest: {package['latest_version']}",
                        remediation=f"Update package: pip install --upgrade {package['name']}"
                    )
                    findings.append(finding)
                
                return findings, {
                    'total_outdated': len(outdated_packages),
                    'packages': outdated_packages
                }
        
        except Exception as e:
            logger.error(f"Outdated packages check error: {e}")
        
        return findings, {}

class SecretsScanner:
    """Scan for exposed secrets and credentials"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def scan(self) -> List[SecurityFinding]:
        """Scan for secrets in code and configuration files"""
        logger.info("Starting secrets scan...")
        
        findings = []
        
        try:
            # Scan for hardcoded secrets
            for directory in self.config.scan_directories:
                dir_path = PROJECT_ROOT / directory
                if not dir_path.exists():
                    continue
                
                # Scan various file types
                file_patterns = ['*.py', '*.yml', '*.yaml', '*.json', '*.env', '*.ini', '*.cfg']
                
                for pattern in file_patterns:
                    for file_path in dir_path.rglob(pattern):
                        if any(exclude in str(file_path) for exclude in self.config.exclude_directories):
                            continue
                        
                        try:
                            secrets_found = self._scan_file_for_secrets(file_path)
                            findings.extend(secrets_found)
                        except Exception as e:
                            logger.warning(f"Failed to scan {file_path} for secrets: {e}")
        
        except Exception as e:
            logger.error(f"Secrets scan error: {e}")
        
        logger.info(f"Secrets scan completed: {len(findings)} findings")
        return findings
    
    def _scan_file_for_secrets(self, file_path: Path) -> List[SecurityFinding]:
        """Scan individual file for secrets"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            for pattern_info in self.config.secrets_patterns:
                pattern = re.compile(pattern_info['pattern'], re.IGNORECASE)
                
                for line_num, line in enumerate(lines, 1):
                    matches = pattern.finditer(line)
                    
                    for match in matches:
                        # Skip common false positives
                        if self._is_false_positive(line, match.group()):
                            continue
                        
                        finding = SecurityFinding(
                            id=f"SECRET-{pattern_info['name'].replace(' ', '_').upper()}-{hash(str(file_path) + str(line_num))}",
                            severity='HIGH' if 'key' in pattern_info['name'].lower() else 'MEDIUM',
                            category='SECRETS',
                            title=f"Exposed {pattern_info['name']}",
                            description=f"Potential {pattern_info['name']} found in source code",
                            file_path=str(file_path),
                            line_number=line_num,
                            evidence=line.strip()[:100] + '...' if len(line.strip()) > 100 else line.strip(),
                            remediation="Move secrets to environment variables or secure configuration management"
                        )
                        findings.append(finding)
        
        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")
        
        return findings
    
    def _is_false_positive(self, line: str, match: str) -> bool:
        """Check if match is likely a false positive"""
        false_positive_indicators = [
            'example', 'placeholder', 'dummy', 'test', 'sample',
            'your_key_here', 'insert_key', 'replace_with',
            'xxx', '***', '...', 'todo', 'fixme'
        ]
        
        line_lower = line.lower()
        match_lower = match.lower()
        
        for indicator in false_positive_indicators:
            if indicator in line_lower or indicator in match_lower:
                return True
        
        # Check for common placeholder patterns
        if re.match(r'^[x*.-]{8,}$', match):
            return True
        
        return False

class NetworkScanner:
    """Scan network configuration and endpoints"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def scan(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Perform network security scan"""
        logger.info("Starting network security scan...")
        
        findings = []
        scan_results = {
            'port_scan_results': {},
            'ssl_scan_results': {},
            'endpoint_scan_results': {}
        }
        
        # Port scanning
        try:
            port_findings, port_results = self._scan_ports()
            findings.extend(port_findings)
            scan_results['port_scan_results'] = port_results
        except Exception as e:
            logger.error(f"Port scan failed: {e}")
        
        # SSL/TLS scanning
        if self.config.check_ssl_certificates:
            try:
                ssl_findings, ssl_results = self._scan_ssl_certificates()
                findings.extend(ssl_findings)
                scan_results['ssl_scan_results'] = ssl_results
            except Exception as e:
                logger.error(f"SSL scan failed: {e}")
        
        # API endpoint scanning
        try:
            endpoint_findings, endpoint_results = self._scan_api_endpoints()
            findings.extend(endpoint_findings)
            scan_results['endpoint_scan_results'] = endpoint_results
        except Exception as e:
            logger.error(f"Endpoint scan failed: {e}")
        
        logger.info(f"Network scan completed: {len(findings)} findings")
        return findings, scan_results
    
    def _scan_ports(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Scan for open ports"""
        findings = []
        results = {}
        
        try:
            for port in self.config.scan_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    if result == 0:
                        results[port] = 'open'
                        
                        # Assess risk based on port
                        severity = 'LOW'
                        description = f"Port {port} is open"
                        
                        if port == 22:  # SSH
                            severity = 'MEDIUM'
                            description = "SSH port is open - ensure strong authentication"
                        elif port in [80, 8000, 8080]:  # HTTP ports
                            severity = 'LOW'
                            description = "HTTP port is open - ensure HTTPS is used for sensitive data"
                        elif port == 5432:  # PostgreSQL
                            severity = 'HIGH'
                            description = "Database port is open - should not be exposed externally"
                        elif port == 6379:  # Redis
                            severity = 'HIGH'
                            description = "Redis port is open - should not be exposed externally"
                        
                        finding = SecurityFinding(
                            id=f"NETWORK-PORT-{port}",
                            severity=severity,
                            category='NETWORK',
                            title=f"Open port: {port}",
                            description=description,
                            evidence=f"Port {port} is accessible on localhost",
                            remediation="Review port exposure and implement appropriate firewall rules"
                        )
                        findings.append(finding)
                    else:
                        results[port] = 'closed'
                
                except Exception as e:
                    results[port] = f'error: {e}'
        
        except Exception as e:
            logger.error(f"Port scanning error: {e}")
        
        return findings, results
    
    def _scan_ssl_certificates(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Scan SSL certificates"""
        findings = []
        results = {}
        
        if not HAS_CRYPTOGRAPHY:
            logger.warning("Cryptography library not available for SSL scanning")
            return findings, results
        
        try:
            for url in self.config.scan_urls:
                parsed_url = urlparse(url)
                
                if parsed_url.scheme == 'https':
                    try:
                        hostname = parsed_url.hostname
                        port = parsed_url.port or 443
                        
                        # Get SSL certificate
                        context = ssl.create_default_context()
                        with socket.create_connection((hostname, port), timeout=10) as sock:
                            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                                cert_der = ssock.getpeercert(binary_form=True)
                                cert = x509.load_der_x509_certificate(cert_der)
                        
                        # Check certificate validity
                        now = datetime.utcnow()
                        
                        if cert.not_valid_after < now:
                            finding = SecurityFinding(
                                id=f"SSL-EXPIRED-{hostname}",
                                severity='HIGH',
                                category='NETWORK',
                                title=f"Expired SSL certificate: {hostname}",
                                description="SSL certificate has expired",
                                evidence=f"Certificate expired on {cert.not_valid_after}",
                                remediation="Renew SSL certificate immediately"
                            )
                            findings.append(finding)
                        
                        elif cert.not_valid_after < now + timedelta(days=30):
                            finding = SecurityFinding(
                                id=f"SSL-EXPIRING-{hostname}",
                                severity='MEDIUM',
                                category='NETWORK',
                                title=f"SSL certificate expiring soon: {hostname}",
                                description="SSL certificate expires within 30 days",
                                evidence=f"Certificate expires on {cert.not_valid_after}",
                                remediation="Plan SSL certificate renewal"
                            )
                            findings.append(finding)
                        
                        results[url] = {
                            'hostname': hostname,
                            'issuer': cert.issuer.rfc4514_string(),
                            'subject': cert.subject.rfc4514_string(),
                            'not_valid_before': cert.not_valid_before.isoformat(),
                            'not_valid_after': cert.not_valid_after.isoformat(),
                            'serial_number': str(cert.serial_number)
                        }
                    
                    except Exception as e:
                        results[url] = f'error: {e}'
                        
                        finding = SecurityFinding(
                            id=f"SSL-ERROR-{parsed_url.hostname}",
                            severity='MEDIUM',
                            category='NETWORK',
                            title=f"SSL certificate error: {parsed_url.hostname}",
                            description=f"Failed to verify SSL certificate: {e}",
                            remediation="Check SSL certificate configuration"
                        )
                        findings.append(finding)
        
        except Exception as e:
            logger.error(f"SSL scanning error: {e}")
        
        return findings, results
    
    def _scan_api_endpoints(self) -> Tuple[List[SecurityFinding], Dict[str, Any]]:
        """Scan API endpoints for security issues"""
        findings = []
        results = {}
        
        try:
            for url in self.config.scan_urls:
                try:
                    # Check for common security headers
                    response = requests.get(url, timeout=10)
                    
                    security_headers = {
                        'X-Content-Type-Options': 'nosniff',
                        'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                        'X-XSS-Protection': '1; mode=block',
                        'Strict-Transport-Security': 'max-age=',
                        'Content-Security-Policy': 'default-src'
                    }
                    
                    missing_headers = []
                    
                    for header, expected in security_headers.items():
                        actual_value = response.headers.get(header)
                        
                        if not actual_value:
                            missing_headers.append(header)
                        elif isinstance(expected, list):
                            if not any(exp in actual_value for exp in expected):
                                missing_headers.append(header)
                        elif isinstance(expected, str) and expected not in actual_value:
                            missing_headers.append(header)
                    
                    if missing_headers:
                        finding = SecurityFinding(
                            id=f"HEADERS-MISSING-{urlparse(url).hostname}",
                            severity='MEDIUM',
                            category='NETWORK',
                            title=f"Missing security headers: {url}",
                            description=f"Missing security headers: {', '.join(missing_headers)}",
                            evidence=f"Response headers: {dict(response.headers)}",
                            remediation="Implement missing security headers in web server configuration"
                        )
                        findings.append(finding)
                    
                    results[url] = {
                        'status_code': response.status_code,
                        'headers': dict(response.headers),
                        'missing_security_headers': missing_headers
                    }
                
                except requests.RequestException as e:
                    results[url] = f'error: {e}'
        
        except Exception as e:
            logger.error(f"API endpoint scanning error: {e}")
        
        return findings, results

# ============================================
# MAIN SECURITY AUDITOR
# ============================================

class SecurityAuditor:
    """Main security auditing orchestrator"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.code_scanner = CodeSecurityScanner(config)
        self.dependency_scanner = DependencyScanner(config)
        self.secrets_scanner = SecretsScanner(config)
        self.network_scanner = NetworkScanner(config)
    
    def run_audit(self) -> SecurityReport:
        """Run complete security audit"""
        logger.info("🔐 Starting comprehensive security audit...")
        start_time = datetime.now()
        
        audit_id = f"audit_{start_time.strftime('%Y%m%d_%H%M%S')}"
        all_findings = []
        
        # Initialize scan results
        code_scan_results = None
        dependency_scan_results = None
        network_scan_results = None
        
        # Code security scan
        if self.config.enable_code_scan:
            try:
                logger.info("Running code security scan...")
                code_findings, code_scan_results = self.code_scanner.scan()
                all_findings.extend(code_findings)
                logger.info(f"Code scan found {len(code_findings)} issues")
            except Exception as e:
                logger.error(f"Code security scan failed: {e}")
        
        # Dependency vulnerability scan
        if self.config.enable_dependency_scan:
            try:
                logger.info("Running dependency vulnerability scan...")
                dep_findings, dependency_scan_results = self.dependency_scanner.scan()
                all_findings.extend(dep_findings)
                logger.info(f"Dependency scan found {len(dep_findings)} issues")
            except Exception as e:
                logger.error(f"Dependency scan failed: {e}")
        
        # Secrets scan
        if self.config.enable_secrets_scan:
            try:
                logger.info("Running secrets scan...")
                secrets_findings = self.secrets_scanner.scan()
                all_findings.extend(secrets_findings)
                logger.info(f"Secrets scan found {len(secrets_findings)} issues")
            except Exception as e:
                logger.error(f"Secrets scan failed: {e}")
        
        # Network security scan
        if self.config.enable_network_scan:
            try:
                logger.info("Running network security scan...")
                network_findings, network_scan_results = self.network_scanner.scan()
                all_findings.extend(network_findings)
                logger.info(f"Network scan found {len(network_findings)} issues")
            except Exception as e:
                logger.error(f"Network scan failed: {e}")
        
        # Filter findings by severity
        filtered_findings = self._filter_findings_by_severity(all_findings)
        
        # Calculate statistics
        severity_counts = self._calculate_severity_counts(filtered_findings)
        
        # Calculate security score
        security_score = self._calculate_security_score(severity_counts, len(filtered_findings))
        
        # Determine risk level
        risk_level = self._determine_risk_level(severity_counts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(filtered_findings, severity_counts)
        
        # Create report
        report = SecurityReport(
            audit_id=audit_id,
            audit_timestamp=start_time.isoformat(),
            config_used=self.config,
            total_findings=len(filtered_findings),
            critical_findings=severity_counts['CRITICAL'],
            high_findings=severity_counts['HIGH'],
            medium_findings=severity_counts['MEDIUM'],
            low_findings=severity_counts['LOW'],
            info_findings=severity_counts['INFO'],
            findings=filtered_findings,
            code_scan_results=code_scan_results,
            dependency_scan_results=dependency_scan_results,
            network_scan_results=network_scan_results,
            security_score=security_score,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
        # Save report
        if self.config.generate_json_report:
            self._save_json_report(report)
        
        if self.config.generate_html_report:
            self._save_html_report(report)
        
        logger.info(f"🔐 Security audit completed: {len(filtered_findings)} findings")
        return report
    
    def _filter_findings_by_severity(self, findings: List[SecurityFinding]) -> List[SecurityFinding]:
        """Filter findings by minimum severity level"""
        severity_order = {'INFO': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        min_level = severity_order.get(self.config.min_severity, 0)
        
        return [f for f in findings if severity_order.get(f.severity, 0) >= min_level]
    
    def _calculate_severity_counts(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Calculate findings count by severity"""
        counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0}
        
        for finding in findings:
            if finding.severity in counts:
                counts[finding.severity] += 1
        
        return counts
    
    def _calculate_security_score(self, severity_counts: Dict[str, int], total_findings: int) -> float:
        """Calculate overall security score (0-100)"""
        if total_findings == 0:
            return 100.0
        
        # Weight severities differently
        weights = {'CRITICAL': 10, 'HIGH': 5, 'MEDIUM': 2, 'LOW': 1, 'INFO': 0.5}
        
        total_weight = sum(severity_counts[sev] * weights[sev] for sev in weights)
        
        # Normalize to 0-100 scale (lower is worse)
        max_possible_weight = total_findings * weights['CRITICAL']
        
        if max_possible_weight == 0:
            return 100.0
        
        score = max(0, 100 - (total_weight / max_possible_weight * 100))
        return round(score, 1)
    
    def _determine_risk_level(self, severity_counts: Dict[str, int]) -> str:
        """Determine overall risk level"""
        if severity_counts['CRITICAL'] > 0:
            return 'CRITICAL'
        elif severity_counts['HIGH'] > 5:
            return 'HIGH'
        elif severity_counts['HIGH'] > 0 or severity_counts['MEDIUM'] > 10:
            return 'MEDIUM'
        elif severity_counts['MEDIUM'] > 0 or severity_counts['LOW'] > 20:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _generate_recommendations(self, findings: List[SecurityFinding], 
                                severity_counts: Dict[str, int]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Critical and high severity recommendations
        if severity_counts['CRITICAL'] > 0:
            recommendations.append(f"🚨 URGENT: Address {severity_counts['CRITICAL']} critical security issues immediately")
        
        if severity_counts['HIGH'] > 0:
            recommendations.append(f"⚠️ Address {severity_counts['HIGH']} high-severity security issues as priority")
        
        # Category-specific recommendations
        categories = {}
        for finding in findings:
            if finding.category not in categories:
                categories[finding.category] = 0
            categories[finding.category] += 1
        
        if categories.get('CODE', 0) > 0:
            recommendations.append("🔧 Review and remediate code security issues")
        
        if categories.get('DEPENDENCY', 0) > 0:
            recommendations.append("📦 Update vulnerable dependencies")
        
        if categories.get('SECRETS', 0) > 0:
            recommendations.append("🔑 Remove hardcoded secrets and use secure configuration management")
        
        if categories.get('NETWORK', 0) > 0:
            recommendations.append("🌐 Review network security configuration")
        
        # General recommendations
        recommendations.extend([
            "🛡️ Implement automated security scanning in CI/CD pipeline",
            "📋 Establish security code review process",
            "🔄 Schedule regular security audits",
            "📚 Provide security training for development team"
        ])
        
        return recommendations
    
    def _save_json_report(self, report: SecurityReport) -> None:
        """Save JSON security report"""
        try:
            report_file = REPORTS_DIR / f"{report.audit_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            logger.info(f"JSON report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
    
    def _save_html_report(self, report: SecurityReport) -> None:
        """Save HTML security report"""
        try:
            html_content = self._generate_html_report_content(report)
            report_file = REPORTS_DIR / f"{report.audit_id}.html"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")
    
    def _generate_html_report_content(self, report: SecurityReport) -> str:
        """Generate HTML report content"""
        # Get severity colors
        severity_colors = {
            'CRITICAL': '#dc3545',
            'HIGH': '#fd7e14', 
            'MEDIUM': '#ffc107',
            'LOW': '#28a745',
            'INFO': '#17a2b8'
        }
        
        # Generate findings HTML
        findings_html = ""
        for finding in report.findings:
            color = severity_colors.get(finding.severity, '#6c757d')
            
            findings_html += f"""
            <div class="finding-item">
                <div class="finding-header">
                    <span class="severity-badge" style="background-color: {color};">{finding.severity}</span>
                    <span class="category-badge">{finding.category}</span>
                    <h4>{finding.title}</h4>
                </div>
                <p class="description">{finding.description}</p>
                {"<p><strong>File:</strong> " + finding.file_path + ":" + str(finding.line_number) + "</p>" if finding.file_path else ""}
                {"<p><strong>Evidence:</strong> <code>" + finding.evidence + "</code></p>" if finding.evidence else ""}
                {"<p><strong>Remediation:</strong> " + finding.remediation + "</p>" if finding.remediation else ""}
            </div>
            """
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Audit Report - {report.audit_id}</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; 
            margin: 0; 
            padding: 20px; 
            background-color: #f8f9fa;
        }}
        
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{ 
            text-align: center; 
            margin-bottom: 30px; 
            padding-bottom: 20px; 
            border-bottom: 2px solid #e9ecef;
        }}
        
        .summary {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }}
        
        .summary-card {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 6px; 
            text-align: center;
        }}
        
        .summary-card h3 {{ 
            margin: 0 0 10px 0; 
            color: #495057;
        }}
        
        .summary-card .number {{ 
            font-size: 2em; 
            font-weight: bold; 
            color: #007bff;
        }}
        
        .risk-level {{ 
            padding: 10px 20px; 
            border-radius: 20px; 
            color: white; 
            font-weight: bold; 
            display: inline-block;
        }}
        
        .severity-badge {{ 
            padding: 4px 8px; 
            border-radius: 4px; 
            color: white; 
            font-size: 0.85em; 
            font-weight: bold;
        }}
        
        .category-badge {{ 
            padding: 4px 8px; 
            border-radius: 4px; 
            background-color: #6c757d; 
            color: white; 
            font-size: 0.85em; 
            margin-left: 10px;
        }}
        
        .finding-item {{ 
            margin-bottom: 25px; 
            padding: 20px; 
            border: 1px solid #dee2e6; 
            border-radius: 6px; 
            background: #fff;
        }}
        
        .finding-header {{ 
            margin-bottom: 15px;
        }}
        
        .finding-header h4 {{ 
            margin: 10px 0 5px 0; 
            color: #495057;
        }}
        
        .description {{ 
            color: #6c757d; 
            margin-bottom: 15px;
        }}
        
        code {{ 
            background-color: #f8f9fa; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-family: Monaco, 'Courier New', monospace;
        }}
        
        .recommendations {{ 
            background: #e7f3ff; 
            padding: 20px; 
            border-radius: 6px; 
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Security Audit Report</h1>
            <p><strong>Audit ID:</strong> {report.audit_id}</p>
            <p><strong>Generated:</strong> {report.audit_timestamp}</p>
            <p><strong>Risk Level:</strong> 
                <span class="risk-level" style="background-color: {severity_colors.get(report.risk_level, '#6c757d')};">
                    {report.risk_level}
                </span>
            </p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Security Score</h3>
                <div class="number">{report.security_score}/100</div>
            </div>
            <div class="summary-card">
                <h3>Total Findings</h3>
                <div class="number">{report.total_findings}</div>
            </div>
            <div class="summary-card">
                <h3>Critical</h3>
                <div class="number" style="color: {severity_colors['CRITICAL']};">{report.critical_findings}</div>
            </div>
            <div class="summary-card">
                <h3>High</h3>
                <div class="number" style="color: {severity_colors['HIGH']};">{report.high_findings}</div>
            </div>
            <div class="summary-card">
                <h3>Medium</h3>
                <div class="number" style="color: {severity_colors['MEDIUM']};">{report.medium_findings}</div>
            </div>
            <div class="summary-card">
                <h3>Low</h3>
                <div class="number" style="color: {severity_colors['LOW']};">{report.low_findings}</div>
            </div>
        </div>
        
        <h2>🔍 Security Findings</h2>
        <div class="findings">
            {findings_html}
        </div>
        
        <div class="recommendations">
            <h2>📋 Recommendations</h2>
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in report.recommendations)}
            </ul>
        </div>
    </div>
</body>
</html>"""

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security audit for StockPredictionPro')
    parser.add_argument('--config', help='Path to security audit configuration JSON file')
    parser.add_argument('--code', action='store_true', default=True, help='Enable code security scan')
    parser.add_argument('--deps', action='store_true', default=True, help='Enable dependency scan')
    parser.add_argument('--secrets', action='store_true', default=True, help='Enable secrets scan')
    parser.add_argument('--network', action='store_true', help='Enable network scan')
    parser.add_argument('--severity', choices=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                       default='LOW', help='Minimum severity level')
    parser.add_argument('--output-dir', default='security_audit/reports', help='Output directory')
    parser.add_argument('--html', action='store_true', default=True, help='Generate HTML report')
    parser.add_argument('--json', action='store_true', default=True, help='Generate JSON report')
    parser.add_argument('--fail-on-high', action='store_true', help='Exit with error on high severity findings')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = SecurityConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Could not load config from {args.config}: {e}")
            config = SecurityConfig()
    else:
        config = SecurityConfig()
    
    # Override with CLI arguments
    config.enable_code_scan = args.code
    config.enable_dependency_scan = args.deps
    config.enable_secrets_scan = args.secrets
    config.enable_network_scan = args.network
    config.min_severity = args.severity
    config.generate_html_report = args.html
    config.generate_json_report = args.json
    config.fail_on_high = args.fail_on_high
    
    try:
        # Run security audit
        auditor = SecurityAuditor(config)
        report = auditor.run_audit()
        
        # Print summary
        print(f"\n{'='*60}")
        print("SECURITY AUDIT SUMMARY")
        print(f"{'='*60}")
        print(f"Audit ID: {report.audit_id}")
        print(f"Security Score: {report.security_score}/100")
        print(f"Risk Level: {report.risk_level}")
        print(f"Total Findings: {report.total_findings}")
        
        if report.total_findings > 0:
            print(f"\nFindings by Severity:")
            print(f"  🔴 Critical: {report.critical_findings}")
            print(f"  🟠 High: {report.high_findings}")
            print(f"  🟡 Medium: {report.medium_findings}")
            print(f"  🟢 Low: {report.low_findings}")
            print(f"  ℹ️  Info: {report.info_findings}")
            
            print(f"\nTop Issues:")
            for finding in report.findings[:5]:
                print(f"  • [{finding.severity}] {finding.title}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📄 Reports saved in: {REPORTS_DIR}")
        
        # Exit with appropriate code
        if config.fail_on_high and (report.high_findings > 0 or report.critical_findings > 0):
            print(f"\n❌ Audit failed due to high/critical severity findings")
            sys.exit(1)
        elif report.critical_findings > 0:
            print(f"\n⚠️ Critical security issues found - immediate attention required")
            sys.exit(0)
        else:
            print(f"\n✅ Security audit completed")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Security audit interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Security audit failed: {e}")
        print(f"❌ Security audit failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
