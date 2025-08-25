"""
setup/setup_directories.py

Comprehensive directory structure setup for StockPredictionPro.
Creates all required folders, sets permissions, and adds documentation files.
Ensures consistent project structure across development and production environments.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import stat
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DirectorySetup')

# ============================================
# DIRECTORY STRUCTURE DEFINITION
# ============================================

class DirectoryStructure:
    """Define complete directory structure for StockPredictionPro"""
    
    # Core data directories
    DATA_DIRECTORIES = {
        'data/raw': 'Raw market data files (OHLCV, fundamentals)',
        'data/processed': 'Cleaned and processed datasets',
        'data/cache': 'Cached API responses and intermediate data',
        'data/external': 'Third-party datasets and reference data',
        'data/interim': 'Intermediate data transformation steps',
        'data/features': 'Engineered features for ML models',
        'data/backup': 'Data backups and archives'
    }
    
    # Model storage directories
    MODEL_DIRECTORIES = {
        'models/trained': 'Trained model artifacts (pickle, joblib, etc.)',
        'models/checkpoints': 'Model training checkpoints',
        'models/experiments': 'Experimental models and variations',
        'models/production': 'Production-ready deployed models',
        'models/metadata': 'Model metadata, configs, and documentation'
    }
    
    # Output and reporting directories
    OUTPUT_DIRECTORIES = {
        'outputs/predictions': 'Model predictions and forecasts',
        'outputs/reports': 'Generated analysis reports',
        'outputs/visualizations': 'Charts, plots, and visualization files',
        'outputs/backtest': 'Backtesting results and analysis',
        'outputs/metrics': 'Performance metrics and evaluations',
        'outputs/exports': 'Exported data and results for external use'
    }
    
    # System and operational directories
    SYSTEM_DIRECTORIES = {
        'logs': 'Application logs and system monitoring',
        'logs/application': 'Main application logs',
        'logs/models': 'Model training and inference logs',
        'logs/data': 'Data processing and ingestion logs',
        'logs/errors': 'Error logs and stack traces',
        'logs/audit': 'Audit trail and system events'
    }
    
    # Configuration and secrets
    CONFIG_DIRECTORIES = {
        'config': 'Configuration files for different environments',
        'config/environments': 'Environment-specific configurations',
        'config/models': 'Model-specific configuration files',
        'config/strategies': 'Trading strategy configurations'
    }
    
    # Development and testing directories
    DEV_DIRECTORIES = {
        'temp': 'Temporary files and working directory',
        'temp/downloads': 'Temporary download files',
        'temp/processing': 'Temporary processing files',
        'temp/uploads': 'Temporary upload staging area',
        'docs/generated': 'Auto-generated documentation',
        'tests/fixtures': 'Test data and fixtures',
        'tests/outputs': 'Test output files and results'
    }
    
    # Additional operational directories
    OPERATIONAL_DIRECTORIES = {
        'cache/redis': 'Redis cache files (if using file-based cache)',
        'cache/http': 'HTTP response cache',
        'cache/models': 'Model prediction cache',
        'monitoring/metrics': 'Monitoring metrics storage',
        'monitoring/alerts': 'Alert configurations and logs',
        'security/keys': 'SSL certificates and key files (if applicable)',
        'deployment/scripts': 'Deployment-specific scripts and configs'
    }

class DirectoryManager:
    """Manage directory creation, permissions, and documentation"""
    
    def __init__(self, base_path: Path = None, environment: str = 'development'):
        self.base_path = base_path or Path.cwd()
        self.environment = environment
        self.created_dirs = []
        self.failed_dirs = []
        
        # Combine all directory definitions
        self.all_directories = {
            **DirectoryStructure.DATA_DIRECTORIES,
            **DirectoryStructure.MODEL_DIRECTORIES,
            **DirectoryStructure.OUTPUT_DIRECTORIES,
            **DirectoryStructure.SYSTEM_DIRECTORIES,
            **DirectoryStructure.CONFIG_DIRECTORIES,
            **DirectoryStructure.DEV_DIRECTORIES,
            **DirectoryStructure.OPERATIONAL_DIRECTORIES
        }
    
    def create_directory(self, dir_path: str, description: str = "") -> bool:
        """Create a single directory with error handling"""
        try:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Set appropriate permissions (readable/writable by owner and group)
            if not os.name == 'nt':  # Unix-like systems
                full_path.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
            
            self.created_dirs.append(str(full_path))
            logger.info(f"✅ Created: {full_path}")
            
            # Create README file with description
            if description:
                self._create_readme(full_path, dir_path, description)
            
            return True
            
        except PermissionError:
            logger.error(f"❌ Permission denied: {dir_path}")
            self.failed_dirs.append(dir_path)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create {dir_path}: {e}")
            self.failed_dirs.append(dir_path)
            return False
    
    def _create_readme(self, dir_path: Path, relative_path: str, description: str) -> None:
        """Create README file in directory with description"""
        readme_file = dir_path / 'README.md'
        
        # Don't overwrite existing README files
        if readme_file.exists():
            return
        
        readme_content = f"""# {relative_path.replace('/', ' > ').title()}

{description}

## Purpose
This directory is part of the StockPredictionPro project structure.

## Contents
- Store files related to: {description.lower()}
- Maintain organized structure for easy access and management

## Usage Guidelines
- Keep files organized with clear naming conventions
- Regular cleanup of temporary or outdated files
- Follow project conventions for file formats and structure

---
*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Environment: {self.environment}*
"""
        
        try:
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.debug(f"📝 Created README: {readme_file}")
        except Exception as e:
            logger.debug(f"⚠️ Could not create README for {relative_path}: {e}")
    
    def create_gitkeep_files(self, empty_dirs: List[str]) -> None:
        """Create .gitkeep files in empty directories to preserve structure in Git"""
        for dir_path in empty_dirs:
            try:
                full_path = self.base_path / dir_path
                gitkeep_file = full_path / '.gitkeep'
                
                if not gitkeep_file.exists():
                    gitkeep_file.touch()
                    logger.debug(f"📌 Created .gitkeep: {gitkeep_file}")
                    
            except Exception as e:
                logger.debug(f"⚠️ Could not create .gitkeep in {dir_path}: {e}")
    
    def create_gitignore_patterns(self) -> None:
        """Create .gitignore files with appropriate patterns for specific directories"""
        gitignore_patterns = {
            'logs': [
                '*.log',
                '*.log.*',
                'log_*',
                '*.tmp'
            ],
            'temp': [
                '*',
                '!.gitkeep',
                '!README.md'
            ],
            'cache': [
                '*.cache',
                '*.tmp',
                'cached_*',
                '*.pickle'
            ],
            'data/raw': [
                '*.csv',
                '*.json',
                '*.parquet',
                '!sample_*',
                '!README.md'
            ],
            'models/trained': [
                '*.pkl',
                '*.joblib',
                '*.h5',
                '*.pt',
                '*.pth',
                '!README.md'
            ],
            'config': [
                '.secrets.yaml',
                'local.yaml',
                '*.key',
                '*.env.local'
            ]
        }
        
        for dir_path, patterns in gitignore_patterns.items():
            try:
                full_path = self.base_path / dir_path
                if full_path.exists():
                    gitignore_file = full_path / '.gitignore'
                    
                    # Don't overwrite existing .gitignore
                    if gitignore_file.exists():
                        continue
                    
                    with open(gitignore_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Auto-generated .gitignore for {dir_path}\n")
                        f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        for pattern in patterns:
                            f.write(f"{pattern}\n")
                    
                    logger.debug(f"🚫 Created .gitignore: {gitignore_file}")
                    
            except Exception as e:
                logger.debug(f"⚠️ Could not create .gitignore for {dir_path}: {e}")
    
    def create_all_directories(self, create_readmes: bool = True, 
                              create_gitkeep: bool = True,
                              create_gitignore: bool = True) -> bool:
        """Create all directories in the structure"""
        logger.info("🚀 Creating StockPredictionPro directory structure...")
        
        total_dirs = len(self.all_directories)
        success_count = 0
        
        # Create directories
        for dir_path, description in self.all_directories.items():
            if self.create_directory(dir_path, description if create_readmes else ""):
                success_count += 1
        
        # Create .gitkeep files for empty directories
        if create_gitkeep:
            empty_dirs = [
                'temp/downloads', 'temp/processing', 'temp/uploads',
                'logs/errors', 'logs/audit', 'cache/http',
                'tests/fixtures', 'tests/outputs'
            ]
            self.create_gitkeep_files(empty_dirs)
        
        # Create .gitignore files
        if create_gitignore:
            self.create_gitignore_patterns()
        
        # Summary
        logger.info("=" * 50)
        logger.info("DIRECTORY SETUP SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Successfully created: {success_count}/{total_dirs} directories")
        
        if self.failed_dirs:
            logger.warning(f"❌ Failed to create: {len(self.failed_dirs)} directories")
            for failed_dir in self.failed_dirs:
                logger.warning(f"   - {failed_dir}")
        
        return len(self.failed_dirs) == 0
    
    def verify_structure(self) -> Dict[str, bool]:
        """Verify that all required directories exist"""
        logger.info("🔍 Verifying directory structure...")
        
        verification_results = {}
        missing_dirs = []
        
        for dir_path in self.all_directories.keys():
            full_path = self.base_path / dir_path
            exists = full_path.exists() and full_path.is_dir()
            verification_results[dir_path] = exists
            
            if not exists:
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            logger.warning(f"⚠️ Missing directories: {len(missing_dirs)}")
            for missing_dir in missing_dirs:
                logger.warning(f"   - {missing_dir}")
        else:
            logger.info("✅ All required directories exist")
        
        return verification_results
    
    def print_structure_tree(self) -> None:
        """Print directory structure as a tree"""
        print("\n📁 StockPredictionPro Directory Structure:")
        print("=" * 50)
        
        # Group directories by category
        categories = {
            'Data Storage': DirectoryStructure.DATA_DIRECTORIES,
            'Model Storage': DirectoryStructure.MODEL_DIRECTORIES,
            'Outputs & Reports': DirectoryStructure.OUTPUT_DIRECTORIES,
            'System & Logs': DirectoryStructure.SYSTEM_DIRECTORIES,
            'Configuration': DirectoryStructure.CONFIG_DIRECTORIES,
            'Development': DirectoryStructure.DEV_DIRECTORIES,
            'Operations': DirectoryStructure.OPERATIONAL_DIRECTORIES
        }
        
        for category, dirs in categories.items():
            print(f"\n📂 {category}:")
            for dir_path, description in dirs.items():
                status = "✅" if (self.base_path / dir_path).exists() else "❌"
                print(f"  {status} {dir_path:<30} - {description}")

def setup_directories(base_path: Path = None, 
                     environment: str = 'development',
                     create_readmes: bool = True,
                     verify: bool = True) -> bool:
    """
    Main function to setup directory structure
    
    Args:
        base_path: Base directory path (defaults to current working directory)
        environment: Environment name for documentation
        create_readmes: Whether to create README files
        verify: Whether to verify structure after creation
        
    Returns:
        True if setup successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STOCKPREDICTIONPRO DIRECTORY SETUP")
    logger.info("=" * 60)
    
    try:
        # Initialize directory manager
        dir_manager = DirectoryManager(base_path, environment)
        
        # Create all directories
        success = dir_manager.create_all_directories(
            create_readmes=create_readmes,
            create_gitkeep=True,
            create_gitignore=True
        )
        
        # Verify structure if requested
        if verify:
            verification_results = dir_manager.verify_structure()
            if not all(verification_results.values()):
                logger.warning("⚠️ Some directories could not be verified")
        
        # Print structure tree
        dir_manager.print_structure_tree()
        
        if success:
            logger.info("🎉 Directory setup completed successfully!")
            return True
        else:
            logger.error("❌ Directory setup completed with errors")
            return False
            
    except Exception as e:
        logger.error(f"❌ Directory setup failed: {e}")
        return False

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup StockPredictionPro Directory Structure')
    parser.add_argument('--path', '-p',
                       type=Path,
                       help='Base path for directory creation (default: current directory)')
    parser.add_argument('--environment', '-e',
                       choices=['development', 'testing', 'production'],
                       default='development',
                       help='Environment for documentation')
    parser.add_argument('--no-readme',
                       action='store_true',
                       help='Skip creating README files')
    parser.add_argument('--no-verify',
                       action='store_true',
                       help='Skip verification step')
    parser.add_argument('--show-tree', '-t',
                       action='store_true',
                       help='Show directory tree only (no creation)')
    
    args = parser.parse_args()
    
    if args.show_tree:
        # Just show the structure without creating
        dir_manager = DirectoryManager(args.path, args.environment)
        dir_manager.print_structure_tree()
        return
    
    # Setup directories
    success = setup_directories(
        base_path=args.path,
        environment=args.environment,
        create_readmes=not args.no_readme,
        verify=not args.no_verify
    )
    
    if success:
        logger.info("✅ Directory setup completed successfully!")
        exit(0)
    else:
        logger.error("❌ Directory setup failed!")
        exit(1)

if __name__ == '__main__':
    main()
