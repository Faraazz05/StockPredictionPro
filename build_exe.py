# ============================================
# StockPredictionPro - build_exe.py
# Advanced PyInstaller executable builder
# ============================================

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import PyInstaller.__main__
from datetime import datetime

class ExecutableBuilder:
    """Advanced executable builder for StockPredictionPro"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / "src"
        self.app_path = self.project_root / "app"
        self.dist_path = self.project_root / "dist"
        self.build_path = self.project_root / "build"
        self.spec_file = self.project_root / "stockpred.spec"
        
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        
        print(f"🔨 Building StockPredictionPro Executable")
        print(f"   Platform: {self.platform} ({self.architecture})")
        print(f"   Python: {sys.version}")
        print(f"   PyInstaller: {PyInstaller.__version__}")
        print("=" * 60)
    
    def clean_previous_builds(self):
        """Clean previous build artifacts"""
        print("🧹 Cleaning previous builds...")
        
        paths_to_clean = [
            self.dist_path,
            self.build_path,
            self.project_root / "*.spec.bak"
        ]
        
        for path in paths_to_clean:
            if isinstance(path, Path) and path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"   Removed directory: {path}")
                else:
                    path.unlink()
                    print(f"   Removed file: {path}")
            elif str(path).endswith("*.spec.bak"):
                # Handle glob pattern
                for spec_bak in self.project_root.glob("*.spec.bak"):
                    spec_bak.unlink()
                    print(f"   Removed backup: {spec_bak}")
        
        print("✅ Build cleanup complete\n")
    
    def verify_dependencies(self):
        """Verify all required dependencies are installed"""
        print("🔍 Verifying dependencies...")
        
        required_packages = [
            "streamlit", "pandas", "numpy", "scikit-learn", 
            "plotly", "yfinance", "ta", "pandas_ta",
            "requests", "python-dotenv", "PyYAML", 
            "joblib", "tqdm", "loguru", "optuna"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"   ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"   ❌ {package} - MISSING")
        
        if missing_packages:
            print(f"\n❌ Missing packages: {missing_packages}")
            print("   Install with: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies verified\n")
        return True
    
    def prepare_build_environment(self):
        """Prepare the build environment"""
        print("⚙️  Preparing build environment...")
        
        # Ensure required directories exist
        directories = [
            "data", "logs", "config", "data/raw", "data/processed",
            "data/models", "data/predictions", "data/backtests", 
            "logs/audit", "logs/audit/runs", "logs/audit/models"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files for empty directories
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists() and not any(dir_path.iterdir()):
                gitkeep.touch()
        
        # Ensure config files exist
        self._ensure_config_files()
        
        print("✅ Build environment ready\n")
    
    def _ensure_config_files(self):
        """Ensure all required config files exist"""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        default_configs = {
            "app_config.yaml": {
                "app": {
                    "name": "StockPredictionPro",
                    "version": "1.0.0",
                    "environment": "production"
                },
                "data": {
                    "default_symbols": ["AAPL", "MSFT", "INFY.NS"],
                    "cache_ttl": 3600
                }
            }
        }
        
        for config_file, config_content in default_configs.items():
            config_path = config_dir / config_file
            if not config_path.exists():
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(config_content, f, default_flow_style=False)
                print(f"   Created: {config_file}")
    
    def build_executable(self, build_type="onefile"):
        """Build the executable using PyInstaller"""
        print(f"🚀 Building executable ({build_type})...")
        
        # Base PyInstaller arguments
        args = [
            "--name=StockPredPro",
            "--clean",
            "--noconfirm",
        ]
        
        # Build type configuration
        if build_type == "onefile":
            args.append("--onefile")
            print("   Type: Single executable file")
        else:
            args.append("--onedir")
            print("   Type: Directory distribution")
        
        # Platform-specific settings
        if self.platform == "windows":
            args.extend([
                "--console",  # Keep console for debugging
                "--icon=assets/icon.ico" if (self.project_root / "assets/icon.ico").exists() else "",
            ])
        elif self.platform == "darwin":  # macOS
            args.extend([
                "--windowed",
                "--icon=assets/icon.icns" if (self.project_root / "assets/icon.icns").exists() else "",
            ])
        else:  # Linux
            args.extend([
                "--console",
            ])
        
        # Remove empty icon arguments
        args = [arg for arg in args if arg]
        
        # Add data files
        data_args = [
            "--add-data=config;config",
            "--add-data=app;app", 
            "--add-data=src;src",
            "--add-data=.env.example;.",
            "--add-data=README.md;.",
            "--add-data=LICENSE;.",
        ]
        
        # Platform-specific data separator
        if self.platform == "windows":
            data_args = [arg.replace(";", ":") if ":" not in arg else arg for arg in data_args]
        
        args.extend(data_args)
        
        # Hidden imports for Streamlit and dependencies
        hidden_imports = [
            # Streamlit core
            "streamlit", "streamlit.web.cli", "streamlit.runtime.scriptrunner",
            "streamlit.components.v1", "streamlit.runtime.caching",
            
            # Scientific computing
            "numpy", "pandas", "scipy", "sklearn", "sklearn.ensemble",
            "sklearn.linear_model", "sklearn.svm", "sklearn.preprocessing",
            "sklearn.model_selection", "sklearn.metrics",
            
            # Visualization
            "plotly", "plotly.graph_objects", "plotly.express", "plotly.subplots",
            
            # Financial data
            "yfinance", "requests", "httpx",
            
            # Technical analysis
            "ta", "pandas_ta",
            
            # Utilities
            "yaml", "dotenv", "joblib", "tqdm", "loguru", "optuna",
            
            # Project modules
            "src", "src.utils", "src.data", "src.features", "src.models",
            "src.evaluation", "src.trading", "app", "app.components", "app.pages"
        ]
        
        for import_name in hidden_imports:
            args.append(f"--hidden-import={import_name}")
        
        # Exclude unnecessary modules
        excludes = [
            "tkinter", "matplotlib", "IPython", "jupyter", 
            "notebook", "pytest", "test", "tests", "unittest"
        ]
        
        for exclude in excludes:
            args.append(f"--exclude-module={exclude}")
        
        # Entry point
        args.append("app/streamlit_app.py")
        
        print(f"   Command: pyinstaller {' '.join(args[:5])}...")
        
        # Run PyInstaller
        try:
            PyInstaller.__main__.run(args)
            print("✅ Executable build completed\n")
            return True
        except Exception as e:
            print(f"❌ Build failed: {str(e)}")
            return False
    
    def create_installer_package(self):
        """Create installation package with launcher and documentation"""
        print("📦 Creating installer package...")
        
        # Determine executable name
        if self.platform == "windows":
            exe_name = "StockPredPro.exe"
        else:
            exe_name = "StockPredPro"
        
        exe_path = self.dist_path / exe_name
        
        if not exe_path.exists():
            print(f"❌ Executable not found: {exe_path}")
            return False
        
        # Create installer directory
        installer_dir = self.dist_path / "installer"
        installer_dir.mkdir(exist_ok=True)
        
        # Copy executable
        shutil.copy2(exe_path, installer_dir / exe_name)
        print(f"   Copied: {exe_name}")
        
        # Create launcher script
        self._create_launcher_script(installer_dir, exe_name)
        
        # Create documentation
        self._create_user_documentation(installer_dir)
        
        # Create uninstaller (optional)
        self._create_uninstaller(installer_dir)
        
        # Calculate package size
        total_size = sum(f.stat().st_size for f in installer_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        print(f"✅ Installer package created: {installer_dir}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Files: {len(list(installer_dir.rglob('*')))}")
        
        return True
    
    def _create_launcher_script(self, installer_dir, exe_name):
        """Create platform-specific launcher script"""
        if self.platform == "windows":
            launcher_content = f"""@echo off
title StockPredictionPro Launcher
echo.
echo 🚀 Starting StockPredictionPro...
echo.
echo The application will open in your web browser at:
echo http://localhost:8501
echo.
echo To stop the application, close this window or press Ctrl+C
echo.
echo ⏳ Please wait while the application loads...
echo.
"{exe_name}"
if errorlevel 1 (
    echo.
    echo ❌ Application failed to start
    echo Check that port 8501 is available
    echo.
    pause
)
"""
            launcher_path = installer_dir / "Start_StockPredPro.bat"
            
        else:  # Linux/macOS
            launcher_content = f"""#!/bin/bash
echo "🚀 Starting StockPredictionPro..."
echo ""
echo "The application will open in your web browser at:"
echo "http://localhost:8501"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""
echo "⏳ Please wait while the application loads..."
echo ""

# Make executable if needed
chmod +x "{exe_name}"

# Run the application
./{exe_name}

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application failed to start"
    echo "Check that port 8501 is available"
    echo ""
    read -p "Press Enter to exit..."
fi
"""
            launcher_path = installer_dir / "start_stockpred.sh"
        
        with open(launcher_path, 'w', newline='\n') as f:
            f.write(launcher_content)
        
        # Make executable on Unix systems
        if self.platform != "windows":
            launcher_path.chmod(0o755)
        
        print(f"   Created: {launcher_path.name}")
    
    def _create_user_documentation(self, installer_dir):
        """Create user documentation"""
        readme_content = f"""
StockPredictionPro - Standalone Application
==========================================

Version: 1.0.0
Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Platform: {self.platform} ({self.architecture})

🚀 QUICK START
==============
1. Run the launcher script:
   • Windows: Double-click "Start_StockPredPro.bat"
   • macOS/Linux: Run "./start_stockpred.sh" in terminal

2. The application will start and open your web browser automatically
   • URL: http://localhost:8501
   • If browser doesn't open, navigate to the URL manually

3. To stop the application:
   • Close the launcher window, or
   • Press Ctrl+C in the terminal

📊 USING THE APPLICATION
========================
1. Start with the "Overview" page for introduction
2. Go to "Data & Indicators" to fetch stock data
3. Use "Regression Models" to train prediction models
4. View results in "Backtest & Performance"
5. Export reports from "Exports & Reports"

🔧 SYSTEM REQUIREMENTS
======================
• Operating System: {platform.system()} {platform.release()}
• RAM: 4GB minimum (8GB recommended)
• Disk Space: 500MB for application + data storage
• Internet: Required for fetching stock data
• Port 8501: Must be available (used by the application)

📁 DATA STORAGE
===============
The application creates these folders in the installation directory:
• data/     - Stock data and processed datasets
• logs/     - Application logs and audit trails
• config/   - Configuration files (can be modified)

⚠️  IMPORTANT NOTES
===================
• This application is for educational purposes only
• Not financial advice - do your own research
• Past performance does not guarantee future results
• Use at your own risk for any trading decisions

🆘 TROUBLESHOOTING
==================
Problem: Application won't start
• Check that port 8501 is not used by another program
• Try restarting your computer
• Run as administrator (Windows) or with sudo (Linux/macOS)

Problem: No data loads
• Check your internet connection
• Ensure firewall allows the application
• Try a different stock symbol (e.g., AAPL, MSFT)

Problem: Charts don't display
• Make sure you have a modern web browser
• Try refreshing the browser page
• Clear your browser cache

📧 SUPPORT
==========
• GitHub: https://github.com/Faraazz05/StockPredictionPro
• Issues: https://github.com/Faraazz05/StockPredictionPro/issues
• Documentation: https://github.com/Faraazz05/StockPredictionPro/wiki

Built with ❤️ using Python, Streamlit, and scikit-learn
"""
        
        readme_path = installer_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"   Created: README.txt")
    
    def _create_uninstaller(self, installer_dir):
        """Create simple uninstaller"""
        if self.platform == "windows":
            uninstaller_content = """@echo off
echo StockPredictionPro Uninstaller
echo ================================
echo.
echo This will remove the application and all data.
echo.
set /p confirm="Are you sure? (y/N): "
if /i "%confirm%"=="y" (
    echo.
    echo Removing application files...
    cd ..
    rmdir /s /q installer
    echo.
    echo ✅ StockPredictionPro has been removed.
    echo.
) else (
    echo Uninstall cancelled.
)
pause
"""
            uninstaller_path = installer_dir / "Uninstall.bat"
            
        else:
            uninstaller_content = """#!/bin/bash
echo "StockPredictionPro Uninstaller"
echo "=============================="
echo ""
echo "This will remove the application and all data."
echo ""
read -p "Are you sure? (y/N): " confirm
if [[ $confirm == [yY] ]]; then
    echo ""
    echo "Removing application files..."
    cd ..
    rm -rf installer
    echo ""
    echo "✅ StockPredictionPro has been removed."
    echo ""
else
    echo "Uninstall cancelled."
fi
"""
            uninstaller_path = installer_dir / "uninstall.sh"
        
        with open(uninstaller_path, 'w', newline='\n') as f:
            f.write(uninstaller_content)
        
        if self.platform != "windows":
            uninstaller_path.chmod(0o755)
        
        print(f"   Created: {uninstaller_path.name}")
    
    def build_report(self):
        """Generate build report"""
        print("📋 Build Report")
        print("=" * 40)
        
        # Check what was created
        exe_files = list(self.dist_path.glob("StockPred*"))
        installer_dir = self.dist_path / "installer"
        
        if exe_files:
            for exe_file in exe_files:
                if exe_file.is_file():
                    size_mb = exe_file.stat().st_size / (1024 * 1024)
                    print(f"📦 Executable: {exe_file.name} ({size_mb:.1f} MB)")
        
        if installer_dir.exists():
            installer_files = list(installer_dir.iterdir())
            print(f"📁 Installer Package: {len(installer_files)} files")
            
            for file in sorted(installer_files):
                if file.is_file():
                    size_kb = file.stat().st_size / 1024
                    print(f"   • {file.name} ({size_kb:.1f} KB)")
        
        print("\n✅ Build completed successfully!")
        print("\n🚀 To test the executable:")
        if self.platform == "windows":
            print("   1. Navigate to dist/installer/")
            print("   2. Double-click 'Start_StockPredPro.bat'")
        else:
            print("   1. cd dist/installer/")
            print("   2. ./start_stockpred.sh")
        
        print("\n📤 To distribute:")
        print("   • Zip the 'installer' directory")
        print("   • Share the zip file with users")
        print("   • Include README.txt for instructions")

def main():
    """Main build function"""
    builder = ExecutableBuilder()
    
    try:
        # Build process
        builder.clean_previous_builds()
        
        if not builder.verify_dependencies():
            return False
        
        builder.prepare_build_environment()
        
        # Build executable
        if not builder.build_executable(build_type="onefile"):
            return False
        
        # Create installer package
        if not builder.create_installer_package():
            return False
        
        # Generate report
        builder.build_report()
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Build cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Build failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
