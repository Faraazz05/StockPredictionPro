"""
setup/install_dependencies.py

Advanced dependency management script for StockPredictionPro.
Handles virtual environments, package installation, version checking, and system validation.
Supports multiple Python environments and comprehensive error handling.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import subprocess
import logging
import json
import pkg_resources
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from packaging import version

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DependencyInstaller')

# ============================================
# SYSTEM REQUIREMENTS AND VALIDATION
# ============================================

class SystemRequirements:
    """System requirements validation"""
    
    MINIMUM_PYTHON_VERSION = "3.8.0"
    RECOMMENDED_PYTHON_VERSION = "3.11.0"
    SUPPORTED_PLATFORMS = ["Windows", "Linux", "Darwin"]  # Darwin = macOS
    
    CRITICAL_PACKAGES = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "pyyaml>=5.4.0"
    ]
    
    OPTIONAL_PACKAGES = [
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "plotly>=5.0.0",
        "dash>=2.0.0"
    ]
    
    ML_PACKAGES = [
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "catboost>=1.0.0",
        "jax>=0.3.0",
        "flax>=0.6.0",
        "optax>=0.1.0"
    ]
    
    DATABASE_PACKAGES = [
        "sqlalchemy>=1.4.0",
        "sqlalchemy-utils>=0.37.0",
        "psycopg2-binary>=2.9.0",  # PostgreSQL adapter
        "alembic>=1.7.0"  # Database migrations
    ]
    
    API_PACKAGES = [
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "httpx>=0.24.0"
    ]

class DependencyInstaller:
    """Advanced dependency installation and management"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.requirements_dir = self.project_root / "requirements"
        self.venv_path = self.project_root / "venv"
        
        # Requirements files
        self.requirements_files = {
            'base': self.project_root / 'requirements.txt',
            'dev': self.requirements_dir / 'requirements-dev.txt',
            'test': self.requirements_dir / 'requirements-test.txt',
            'prod': self.requirements_dir / 'requirements-prod.txt',
            'ml': self.requirements_dir / 'requirements-ml.txt',
            'api': self.requirements_dir / 'requirements-api.txt'
        }
        
        self.installed_packages = {}
        self.failed_packages = []
    
    def validate_system(self) -> bool:
        """Validate system requirements"""
        logger.info("🔍 Validating system requirements...")
        
        # Check Python version
        current_python = platform.python_version()
        min_version = SystemRequirements.MINIMUM_PYTHON_VERSION
        
        if version.parse(current_python) < version.parse(min_version):
            logger.error(f"❌ Python {min_version}+ required, found {current_python}")
            return False
        
        logger.info(f"✅ Python version: {current_python}")
        
        # Check platform
        current_platform = platform.system()
        if current_platform not in SystemRequirements.SUPPORTED_PLATFORMS:
            logger.warning(f"⚠️ Platform {current_platform} not officially supported")
        else:
            logger.info(f"✅ Platform: {current_platform}")
        
        # Check pip availability
        try:
            subprocess.check_output([sys.executable, '-m', 'pip', '--version'])
            logger.info("✅ pip is available")
        except subprocess.CalledProcessError:
            logger.error("❌ pip is not available")
            return False
        
        return True
    
    def detect_virtual_environment(self) -> Dict[str, any]:
        """Detect and analyze virtual environment"""
        logger.info("🔍 Checking virtual environment...")
        
        venv_info = {
            'active': False,
            'type': None,
            'path': None,
            'python_executable': sys.executable
        }
        
        # Check for active virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_info['active'] = True
            venv_info['path'] = sys.prefix
            
            # Detect virtual environment type
            if 'conda' in sys.executable or 'anaconda' in sys.executable:
                venv_info['type'] = 'conda'
            elif 'venv' in sys.executable or '.venv' in sys.executable:
                venv_info['type'] = 'venv'
            elif 'virtualenv' in sys.executable:
                venv_info['type'] = 'virtualenv'
            else:
                venv_info['type'] = 'unknown'
            
            logger.info(f"✅ Virtual environment detected: {venv_info['type']} at {venv_info['path']}")
        else:
            logger.warning("⚠️ No virtual environment detected - installation will use system Python")
        
        return venv_info
    
    def create_virtual_environment(self, python_version: str = None) -> bool:
        """Create virtual environment if it doesn't exist"""
        if self.venv_path.exists():
            logger.info(f"✅ Virtual environment already exists at: {self.venv_path}")
            return True
        
        logger.info(f"📦 Creating virtual environment at: {self.venv_path}")
        
        try:
            # Use specific Python version if provided
            python_cmd = python_version if python_version else sys.executable
            
            subprocess.check_call([
                python_cmd, '-m', 'venv', str(self.venv_path)
            ])
            
            logger.info("✅ Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"❌ Python executable not found: {python_version}")
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        logger.info("🔄 Upgrading pip...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])
            logger.info("✅ pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to upgrade pip: {e}")
            return False
    
    def install_package(self, package_spec: str, upgrade: bool = False) -> bool:
        """Install single package with error handling"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install']
            
            if upgrade:
                cmd.append('--upgrade')
            
            cmd.append(package_spec)
            
            logger.info(f"📦 Installing: {package_spec}")
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
            
            # Extract package name for tracking
            package_name = package_spec.split('>=')[0].split('==')[0].split('[')[0]
            self.installed_packages[package_name] = package_spec
            
            logger.info(f"✅ Installed: {package_spec}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package_spec}: {e}")
            self.failed_packages.append(package_spec)
            return False
    
    def install_from_requirements(self, requirements_file: Path, 
                                 upgrade: bool = False) -> bool:
        """Install packages from requirements file"""
        if not requirements_file.exists():
            logger.warning(f"⚠️ Requirements file not found: {requirements_file}")
            return False
        
        logger.info(f"📋 Installing from: {requirements_file}")
        
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            
            if upgrade:
                cmd.append('--upgrade')
            
            subprocess.check_call(cmd)
            logger.info(f"✅ Successfully installed from: {requirements_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install from {requirements_file}: {e}")
            return False
    
    def check_package_versions(self, packages: List[str]) -> Dict[str, str]:
        """Check versions of installed packages"""
        logger.info("🔍 Checking package versions...")
        
        versions = {}
        
        for package_spec in packages:
            package_name = package_spec.split('>=')[0].split('==')[0].split('[')[0]
            
            try:
                dist = pkg_resources.get_distribution(package_name)
                versions[package_name] = dist.version
                logger.debug(f"✅ {package_name}: {dist.version}")
            except pkg_resources.DistributionNotFound:
                versions[package_name] = "Not installed"
                logger.debug(f"❌ {package_name}: Not installed")
        
        return versions
    
    def generate_requirements_files(self) -> bool:
        """Generate comprehensive requirements files"""
        logger.info("📝 Generating requirements files...")
        
        try:
            # Create requirements directory
            self.requirements_dir.mkdir(exist_ok=True)
            
            # Base requirements (essential packages)
            base_requirements = (
                SystemRequirements.CRITICAL_PACKAGES + 
                SystemRequirements.DATABASE_PACKAGES
            )
            
            # Development requirements
            dev_requirements = [
                "pytest>=6.0.0",
                "pytest-cov>=3.0.0",
                "black>=22.0.0",
                "flake8>=4.0.0",
                "mypy>=0.900",
                "pre-commit>=2.15.0",
                "jupyter>=1.0.0",
                "notebook>=6.4.0",
                "ipykernel>=6.0.0"
            ]
            
            # Testing requirements
            test_requirements = [
                "pytest>=6.0.0",
                "pytest-cov>=3.0.0",
                "pytest-mock>=3.6.0",
                "pytest-xdist>=2.4.0",
                "factory-boy>=3.2.0",
                "faker>=8.0.0"
            ]
            
            # Production requirements
            prod_requirements = base_requirements + SystemRequirements.API_PACKAGES
            
            # ML requirements
            ml_requirements = SystemRequirements.ML_PACKAGES
            
            # API requirements
            api_requirements = SystemRequirements.API_PACKAGES
            
            # Write requirements files
            requirements_content = {
                'base': base_requirements,
                'dev': dev_requirements,
                'test': test_requirements,
                'prod': prod_requirements,
                'ml': ml_requirements,
                'api': api_requirements
            }
            
            for req_type, packages in requirements_content.items():
                if req_type == 'base':
                    file_path = self.requirements_files['base']
                else:
                    file_path = self.requirements_files[req_type]
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {req_type.upper()} requirements for StockPredictionPro\n")
                    f.write(f"# Generated on {platform.python_version()} Python\n")
                    f.write(f"# Platform: {platform.system()} {platform.release()}\n\n")
                    
                    for package in sorted(packages):
                        f.write(f"{package}\n")
                
                logger.info(f"✅ Created: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate requirements files: {e}")
            return False
    
    def install_environment(self, env_type: str = 'base', 
                           upgrade: bool = False,
                           create_venv: bool = False) -> bool:
        """Install complete environment based on type"""
        logger.info("=" * 60)
        logger.info(f"INSTALLING {env_type.upper()} ENVIRONMENT")
        logger.info("=" * 60)
        
        # Validate system first
        if not self.validate_system():
            return False
        
        # Create virtual environment if requested
        if create_venv and not self.detect_virtual_environment()['active']:
            if not self.create_virtual_environment():
                logger.error("❌ Failed to create virtual environment")
                return False
        
        # Check virtual environment
        venv_info = self.detect_virtual_environment()
        
        # Upgrade pip
        if not self.upgrade_pip():
            logger.warning("⚠️ Could not upgrade pip, continuing...")
        
        # Generate requirements files if they don't exist
        if not all(f.exists() for f in self.requirements_files.values()):
            if not self.generate_requirements_files():
                logger.error("❌ Failed to generate requirements files")
                return False
        
        # Install based on environment type
        success = True
        
        if env_type in ['base', 'all']:
            success &= self.install_from_requirements(self.requirements_files['base'], upgrade)
        
        if env_type in ['dev', 'all']:
            success &= self.install_from_requirements(self.requirements_files['dev'], upgrade)
        
        if env_type in ['ml', 'all']:
            success &= self.install_from_requirements(self.requirements_files['ml'], upgrade)
        
        if env_type in ['api', 'all']:
            success &= self.install_from_requirements(self.requirements_files['api'], upgrade)
        
        if env_type == 'prod':
            success &= self.install_from_requirements(self.requirements_files['prod'], upgrade)
        
        # Install test requirements for dev/all
        if env_type in ['dev', 'test', 'all']:
            success &= self.install_from_requirements(self.requirements_files['test'], upgrade)
        
        # Summary
        self._print_installation_summary(venv_info, success)
        
        return success
    
    def _print_installation_summary(self, venv_info: Dict, success: bool) -> None:
        """Print installation summary"""
        logger.info("=" * 60)
        logger.info("INSTALLATION SUMMARY")
        logger.info("=" * 60)
        
        # Environment info
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"Platform: {platform.system()} {platform.release()}")
        logger.info(f"Virtual Environment: {'✅ Active' if venv_info['active'] else '❌ Not Active'}")
        
        if venv_info['active']:
            logger.info(f"Environment Type: {venv_info['type']}")
            logger.info(f"Environment Path: {venv_info['path']}")
        
        # Package summary
        if self.installed_packages:
            logger.info(f"✅ Successfully installed: {len(self.installed_packages)} packages")
        
        if self.failed_packages:
            logger.warning(f"❌ Failed to install: {len(self.failed_packages)} packages")
            for package in self.failed_packages:
                logger.warning(f"   - {package}")
        
        # Final status
        if success and not self.failed_packages:
            logger.info("🎉 Environment setup completed successfully!")
        elif success and self.failed_packages:
            logger.warning("⚠️ Environment setup completed with some failures")
        else:
            logger.error("❌ Environment setup failed!")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Install StockPredictionPro Dependencies')
    parser.add_argument('--env-type', '-e',
                       choices=['base', 'dev', 'ml', 'api', 'prod', 'test', 'all'],
                       default='base',
                       help='Environment type to install')
    parser.add_argument('--upgrade', '-u',
                       action='store_true',
                       help='Upgrade existing packages')
    parser.add_argument('--create-venv', '-v',
                       action='store_true',
                       help='Create virtual environment if not active')
    parser.add_argument('--generate-requirements', '-g',
                       action='store_true',
                       help='Generate requirements files only')
    parser.add_argument('--check-versions', '-c',
                       action='store_true',
                       help='Check package versions only')
    parser.add_argument('--project-root', '-p',
                       type=Path,
                       help='Project root directory')
    
    args = parser.parse_args()
    
    # Initialize installer
    installer = DependencyInstaller(project_root=args.project_root)
    
    # Handle different operations
    if args.generate_requirements:
        success = installer.generate_requirements_files()
        print("✅ Requirements files generated" if success else "❌ Failed to generate requirements files")
        return
    
    if args.check_versions:
        all_packages = (
            SystemRequirements.CRITICAL_PACKAGES +
            SystemRequirements.ML_PACKAGES +
            SystemRequirements.API_PACKAGES
        )
        versions = installer.check_package_versions(all_packages)
        
        print("\n📦 Package Versions:")
        print("-" * 40)
        for package, pkg_version in versions.items():
            print(f"{package:<20}: {pkg_version}")
        return
    
    # Install environment
    success = installer.install_environment(
        env_type=args.env_type,
        upgrade=args.upgrade,
        create_venv=args.create_venv
    )
    
    exit(0 if success else 1)

if __name__ == '__main__':
    main()
