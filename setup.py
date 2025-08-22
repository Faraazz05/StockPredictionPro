# ============================================
# StockPredictionPro - setup.py
# Modern Python packaging setup
# ============================================

import os
from pathlib import Path
from setuptools import setup, find_packages

# Read version from pyproject.toml or use default
def get_version():
    """Get version from pyproject.toml or fallback to default"""
    try:
        import tomli
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject = tomli.load(f)
                return pyproject["project"]["version"]
    except (ImportError, KeyError):
        pass
    return "1.0.0"

# Read README for long description
def get_long_description():
    """Get long description from README.md"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Advanced stock prediction system with ML models"

# Read requirements.txt
def get_requirements():
    """Parse requirements.txt for dependencies"""
    requirements_path = Path(__file__).parent / "requirements.txt"
    requirements = []
    
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    # Handle inline comments
                    if "#" in line:
                        line = line.split("#")[0].strip()
                    # Skip conditional dependencies for now
                    if not line.startswith("-"):
                        requirements.append(line)
    
    return requirements

# Development dependencies
def get_dev_requirements():
    """Get development dependencies"""
    return [
        "black>=24.8.0",
        "isort>=5.13.2", 
        "flake8>=7.1.1",
        "mypy>=1.11.2",
        "pre-commit>=3.8.0",
        "pytest>=8.3.2",
        "pytest-cov>=5.0.0",
        "pytest-mock>=3.14.0",
        "pytest-asyncio>=0.24.0"
    ]

# Optional dependencies
extras_require = {
    "api": [
        "fastapi>=0.112.2",
        "uvicorn[standard]>=0.30.6", 
        "pydantic>=2.8.2"
    ],
    "dev": get_dev_requirements(),
    "build": [
        "pyinstaller>=6.10.0",
        "docker>=7.1.0"
    ],
    "ml-extra": [
        "xgboost>=2.1.1",
        "lightgbm>=4.5.0",
        "shap>=0.46.0"
    ],
    "monitoring": [
        "sentry-sdk>=2.13.0",
        "prometheus-client>=0.20.0",
        "structlog>=24.4.0"
    ]
}

# Add 'all' extra that includes everything
extras_require["all"] = [
    dep for deps in extras_require.values() for dep in deps
]

# Package configuration
setup(
    name="stockpredictionpro",
    version=get_version(),
    author="Mohd Faraz",
    author_email="sp_mohdfaraz@outlook.com",
    description="Advanced stock prediction system with regression and classification models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Faraazz05/StockPredictionPro",
    project_urls={
        "Bug Tracker": "https://github.com/Faraazz05/StockPredictionPro/issues",
        "Documentation": "https://github.com/Faraazz05/StockPredictionPro/wiki",
        "Source Code": "https://github.com/Faraazz05/StockPredictionPro",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv", "*.md", "*.txt"],
    },
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=get_requirements(),
    extras_require=extras_require,
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "stockpred=src.cli:main",
            "stockpred-train=scripts.models.train_all_models:main",
            "stockpred-serve=src.api.main:serve",
            "stockpred-streamlit=app.streamlit_app:main",
        ],
    },
    
    # Classification metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", 
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "stock-prediction", "machine-learning", "financial-analysis",
        "regression", "classification", "streamlit", "technical-indicators",
        "trading", "finance", "ml", "data-science"
    ],
    
    # Licensing
    license="MIT",
    
    # Zip safe
    zip_safe=False,
)

# Post-installation setup
if __name__ == "__main__":
    print("🚀 StockPredictionPro Setup Complete!")
    print("=" * 50)
    print("📦 Package: stockpredictionpro")
    print(f"📋 Version: {get_version()}")
    print("🐍 Python: >=3.10")
    print()
    print("🏁 Quick Start:")
    print("   pip install -e .                    # Development install")
    print("   pip install -e .[dev]              # With dev tools")
    print("   pip install -e .[all]              # All features")
    print()
    print("🚀 Run Commands:")
    print("   stockpred-streamlit                 # Launch Streamlit app")  
    print("   stockpred-train                     # Train models")
    print("   stockpred-serve                     # Start API server")
    print()
    print("📁 Next Steps:")
    print("   1. Set up config files: python scripts/setup/setup_config.py")
    print("   2. Create directories: python scripts/setup/setup_directories.py") 
    print("   3. Run Streamlit: streamlit run app/streamlit_app.py")
    print("=" * 50)
