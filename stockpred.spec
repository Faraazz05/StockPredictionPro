# ============================================
# StockPredictionPro - stockpred.spec
# PyInstaller specification for building executable
# ============================================

# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(SPECPATH).resolve()
src_path = project_root / "src"
app_path = project_root / "app"
config_path = project_root / "config"

# Add src to Python path for imports
sys.path.insert(0, str(src_path))

# Analysis configuration
a = Analysis(
    # Entry point - Streamlit app
    ['app/streamlit_app.py'],
    
    # Additional paths for Python imports
    pathex=[
        str(project_root),
        str(src_path),
        str(app_path),
    ],
    
    # Binary dependencies (empty for now, add if needed)
    binaries=[],
    
    # Data files to include in the executable
    datas=[
        # Configuration files
        ('config/*.yaml', 'config/'),
        ('config/*.yml', 'config/'),
        
        # Streamlit app files
        ('app/pages/*.py', 'app/pages/'),
        ('app/components/*.py', 'app/components/'),
        ('app/styles/*.css', 'app/styles/'),
        ('app/styles/*.py', 'app/styles/'),
        ('app/utils/*.py', 'app/utils/'),
        
        # Source code modules
        ('src/', 'src/'),
        
        # Environment and setup files
        ('.env.example', '.'),
        ('README.md', '.'),
        ('LICENSE', '.'),
        
        # Create empty data directories in executable
        ('data/.gitkeep', 'data/'),
        ('logs/.gitkeep', 'logs/'),
    ],
    
    # Hidden imports (modules not detected automatically)
    hiddenimports=[
        # Streamlit and its dependencies
        'streamlit',
        'streamlit.web',
        'streamlit.web.cli',
        'streamlit.runtime',
        'streamlit.runtime.scriptrunner',
        'streamlit.components',
        'streamlit.components.v1',
        
        # Scientific computing
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.linear_model',
        'sklearn.svm',
        'sklearn.preprocessing',
        'sklearn.model_selection',
        'sklearn.metrics',
        
        # Plotting and visualization
        'plotly',
        'plotly.graph_objects',
        'plotly.express',
        'plotly.subplots',
        
        # Technical analysis
        'ta',
        'pandas_ta',
        
        # Data sources
        'yfinance',
        'requests',
        'httpx',
        
        # Utilities
        'yaml',
        'dotenv',
        'joblib',
        'tqdm',
        'loguru',
        
        # ML optimization
        'optuna',
        'optuna.samplers',
        'optuna.pruners',
        
        # Optional dependencies (uncomment if used)
        # 'fastapi',
        # 'uvicorn',
        # 'pydantic',
        # 'redis',
        # 'sqlalchemy',
        
        # Project modules
        'src',
        'src.utils',
        'src.utils.config_loader',
        'src.utils.logger',
        'src.data',
        'src.data.fetchers',
        'src.data.processors',
        'src.features',
        'src.features.indicators',
        'src.features.transformers',
        'src.models',
        'src.models.regression',
        'src.models.classification',
        'src.evaluation',
        'src.trading',
        
        # App modules
        'app',
        'app.components',
        'app.pages',
        'app.utils',
    ],
    
    # Hooks directory (optional)
    hookspath=[],
    
    # Runtime hooks (optional)
    hooksconfig={},
    
    # Modules to exclude
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'test',
        'tests',
    ],
    
    # Additional options
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Executable configuration
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    
    # Executable name
    name='StockPredPro',
    
    # Debug options
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    
    # Console options
    console=True,  # Set to False for GUI-only app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    
    # Icon (add if you have one)
    # icon='assets/icon.ico',  # Windows
    # icon='assets/icon.icns', # macOS
)

# macOS app bundle (optional)
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='StockPredPro.app',
        icon='assets/icon.icns',  # Add if you have an icon
        bundle_identifier='com.stockpredpro.app',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'StockPred Data',
                    'CFBundleTypeIconFile': 'icon.icns',
                    'LSItemContentTypes': ['public.comma-separated-values-text'],
                    'LSHandlerRank': 'Owner'
                }
            ]
        },
    )

# Collection for distributable directory (alternative to single exe)
# Uncomment if you prefer a directory distribution
"""
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StockPredPro',
)
"""
