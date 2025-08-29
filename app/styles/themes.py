"""
app/styles/themes.py

Advanced theme management and styling for StockPredictionPro.
Provides dynamic theme switching, custom CSS injection, and consistent
UI styling across all application components.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
from typing import Dict, Any, Optional
from .colors import (
    PRIMARY_BLUE, PRIMARY_TEAL, PRIMARY_CORAL, PRIMARY_PURPLE,
    PROFIT_GREEN, LOSS_RED, WARNING_AMBER, NEUTRAL_STEEL,
    BACKGROUND_PRIMARY, BACKGROUND_SECONDARY, BACKGROUND_ACCENT,
    DARK_PRIMARY, DARK_SECONDARY, DARK_ACCENT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_LIGHT, TEXT_INVERSE,
    BUTTON_PRIMARY, BUTTON_SUCCESS, BUTTON_WARNING, BUTTON_DANGER,
    get_theme_colors, GRADIENTS
)

# ============================================
# THEME CONFIGURATIONS
# ============================================

THEME_CONFIGS = {
    "light": {
        "name": "Light Professional",
        "description": "Clean, bright interface perfect for daytime trading",
        "background": BACKGROUND_PRIMARY,
        "background_secondary": BACKGROUND_SECONDARY,
        "background_accent": BACKGROUND_ACCENT,
        "text_primary": TEXT_PRIMARY,
        "text_secondary": TEXT_SECONDARY,
        "primary_color": PRIMARY_BLUE,
        "success_color": PROFIT_GREEN,
        "danger_color": LOSS_RED,
        "warning_color": WARNING_AMBER,
        "border_color": "#e9ecef",
        "shadow": "0 2px 4px rgba(0,0,0,0.1)",
        "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    },
    
    "dark": {
        "name": "Dark Professional",
        "description": "Sleek dark interface for focused analysis",
        "background": DARK_PRIMARY,
        "background_secondary": DARK_SECONDARY,
        "background_accent": DARK_ACCENT,
        "text_primary": TEXT_INVERSE,
        "text_secondary": TEXT_LIGHT,
        "primary_color": "#4dabf7",
        "success_color": "#51cf66",
        "danger_color": "#ff6b6b",
        "warning_color": "#ffd43b",
        "border_color": "#495057",
        "shadow": "0 2px 8px rgba(0,0,0,0.3)",
        "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    },
    
    "financial": {
        "name": "Financial Terminal",
        "description": "Professional trading terminal styling",
        "background": "#fafbfc",
        "background_secondary": "#f1f3f4",
        "background_accent": "#e8eaed",
        "text_primary": "#202124",
        "text_secondary": "#5f6368",
        "primary_color": "#1a73e8",
        "success_color": "#137333",
        "danger_color": "#d93025",
        "warning_color": "#f9ab00",
        "border_color": "#dadce0",
        "shadow": "0 1px 3px rgba(60,64,67,0.3)",
        "font_family": "'Roboto Mono', 'SF Mono', Monaco, Consolas, monospace"
    }
}

# ============================================
# MAIN THEME APPLICATION
# ============================================

def apply_custom_theme(theme_name: str = "financial") -> None:
    """
    Apply comprehensive custom theme to Streamlit app
    
    Args:
        theme_name: Theme to apply ('light', 'dark', 'financial')
    """
    if theme_name not in THEME_CONFIGS:
        st.warning(f"Theme '{theme_name}' not found. Using 'financial' theme.")
        theme_name = "financial"
    
    config = THEME_CONFIGS[theme_name]
    
    # Inject comprehensive CSS
    custom_css = _generate_theme_css(config)
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Store current theme in session state
    st.session_state.current_theme = theme_name

def _generate_theme_css(config: Dict[str, str]) -> str:
    """Generate comprehensive CSS for the theme"""
    
    return f"""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap');
    
    /* Root variables */
    :root {{
        --bg-primary: {config['background']};
        --bg-secondary: {config['background_secondary']};
        --bg-accent: {config['background_accent']};
        --text-primary: {config['text_primary']};
        --text-secondary: {config['text_secondary']};
        --color-primary: {config['primary_color']};
        --color-success: {config['success_color']};
        --color-danger: {config['danger_color']};
        --color-warning: {config['warning_color']};
        --border-color: {config['border_color']};
        --shadow: {config['shadow']};
        --font-family: {config['font_family']};
    }}
    
    /* Main app styling */
    .stApp {{
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: var(--font-family);
    }}
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypc4j {{
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }}
    
    /* Header styling */
    .css-10trblm {{
        background-color: var(--bg-primary);
        border-bottom: 1px solid var(--border-color);
    }}
    
    /* Metric containers */
    [data-testid="metric-container"] {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: all 0.2s ease;
    }}
    
    [data-testid="metric-container"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, var(--color-primary), {PRIMARY_TEAL});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: var(--font-family);
        transition: all 0.2s ease;
        box-shadow: var(--shadow);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        filter: brightness(1.05);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Success button */
    .stButton.success > button {{
        background: linear-gradient(135deg, var(--color-success), #51cf66);
    }}
    
    /* Danger button */
    .stButton.danger > button {{
        background: linear-gradient(135deg, var(--color-danger), #ff6b6b);
    }}
    
    /* Form styling */
    .stSelectbox > div > div, .stMultiSelect > div > div,
    .stTextInput > div > div, .stTextArea textarea {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: var(--font-family);
    }}
    
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within,
    .stTextInput > div > div:focus-within {{
        border-color: var(--color-primary);
        box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
    }}
    
    /* Slider styling */
    .stSlider > div > div > div > div {{
        background-color: var(--color-primary);
    }}
    
    /* Dataframe styling */
    .stDataFrame {{
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }}
    
    /* Chart containers */
    .js-plotly-plot {{
        border-radius: 12px;
        box-shadow: var(--shadow);
        background-color: var(--bg-secondary);
    }}
    
    /* Alert styling */
    .stAlert {{
        border-radius: 8px;
        border: none;
        box-shadow: var(--shadow);
    }}
    
    .stSuccess {{
        background: linear-gradient(135deg, var(--color-success)20, var(--color-success)10);
        border-left: 4px solid var(--color-success);
        color: var(--text-primary);
    }}
    
    .stError {{
        background: linear-gradient(135deg, var(--color-danger)20, var(--color-danger)10);
        border-left: 4px solid var(--color-danger);
        color: var(--text-primary);
    }}
    
    .stWarning {{
        background: linear-gradient(135deg, var(--color-warning)20, var(--color-warning)10);
        border-left: 4px solid var(--color-warning);
        color: var(--text-primary);
    }}
    
    .stInfo {{
        background: linear-gradient(135deg, var(--color-primary)20, var(--color-primary)10);
        border-left: 4px solid var(--color-primary);
        color: var(--text-primary);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: var(--bg-secondary);
        border-radius: 12px;
        padding: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        font-family: var(--font-family);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--color-primary);
        color: white;
        box-shadow: var(--shadow);
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: var(--font-family);
    }}
    
    .streamlit-expanderContent {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }}
    
    /* Progress bar */
    .stProgress .st-bo {{
        background-color: var(--color-primary);
    }}
    
    /* Loading spinner */
    .stSpinner {{
        border-color: var(--color-primary);
    }}
    
    /* Custom classes for financial data */
    .profit {{
        color: var(--color-success);
        font-weight: 600;
    }}
    
    .loss {{
        color: var(--color-danger);
        font-weight: 600;
    }}
    
    .neutral {{
        color: var(--text-secondary);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--bg-accent);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--border-color);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-secondary);
    }}
    
    /* Animation classes */
    .fade-in {{
        animation: fadeIn 0.5s ease-in;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .slide-up {{
        animation: slideUp 0.3s ease-out;
    }}
    
    @keyframes slideUp {{
        from {{ transform: translateY(10px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {{
        [data-testid="metric-container"] {{
            padding: 1rem;
        }}
        
        .stButton > button {{
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }}
    }}
    </style>
    """

# ============================================
# THEME SELECTION WIDGET
# ============================================

def create_theme_selector(key: str = "theme_selector") -> str:
    """
    Create theme selection widget for sidebar
    
    Args:
        key: Unique key for widget
        
    Returns:
        Selected theme name
    """
    with st.sidebar:
        st.subheader("🎨 Theme Selection")
        
        # Theme options with descriptions
        theme_options = [
            f"{config['name']} - {config['description']}" 
            for config in THEME_CONFIGS.values()
        ]
        
        selected_option = st.selectbox(
            "Choose Theme",
            options=theme_options,
            index=2,  # Default to financial theme
            key=key,
            help="Select the visual theme for your application"
        )
        
        # Extract theme name from selection
        theme_name = list(THEME_CONFIGS.keys())[theme_options.index(selected_option)]
        
        # Apply theme immediately
        apply_custom_theme(theme_name)
        
        return theme_name

# ============================================
# COMPONENT-SPECIFIC STYLING
# ============================================

def apply_chart_theme(theme_name: str = "financial") -> Dict[str, Any]:
    """
    Get Plotly theme configuration for charts
    
    Args:
        theme_name: Theme name
        
    Returns:
        Plotly theme configuration
    """
    config = THEME_CONFIGS.get(theme_name, THEME_CONFIGS["financial"])
    
    return {
        "layout": {
            "plot_bgcolor": config["background_secondary"],
            "paper_bgcolor": config["background"],
            "font": {
                "family": config["font_family"],
                "color": config["text_primary"]
            },
            "colorway": [
                config["primary_color"],
                config["success_color"],
                config["danger_color"],
                config["warning_color"],
                PRIMARY_PURPLE,
                PRIMARY_CORAL
            ],
            "template": "plotly_white" if theme_name == "light" else "plotly_dark"
        }
    }

def style_metric_cards() -> None:
    """Apply additional styling to metric cards"""
    st.markdown("""
    <style>
    [data-testid="metric-container"] > div {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    [data-testid="metric-container"] .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }}
    
    [data-testid="metric-container"] .metric-delta {{
        font-size: 0.9rem;
        font-weight: 500;
    }}
    </style>
    """, unsafe_allow_html=True)

def apply_table_styling() -> None:
    """Apply enhanced table styling"""
    st.markdown("""
    <style>
    .dataframe tbody tr:nth-child(even) {{
        background-color: var(--bg-accent);
    }}
    
    .dataframe thead th {{
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        font-weight: 600;
        border-bottom: 2px solid var(--border-color);
    }}
    
    .dataframe tbody td {{
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# INITIALIZATION AND HELPERS
# ============================================

def initialize_theme_system() -> None:
    """Initialize the theme system with default settings"""
    if "current_theme" not in st.session_state:
        st.session_state.current_theme = "financial"
    
    # Apply the current theme
    apply_custom_theme(st.session_state.current_theme)

def get_current_theme() -> str:
    """Get the currently active theme"""
    return st.session_state.get("current_theme", "financial")

def get_theme_config(theme_name: str = None) -> Dict[str, str]:
    """
    Get configuration for specified theme
    
    Args:
        theme_name: Theme name (defaults to current theme)
        
    Returns:
        Theme configuration dictionary
    """
    if theme_name is None:
        theme_name = get_current_theme()
    
    return THEME_CONFIGS.get(theme_name, THEME_CONFIGS["financial"])

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage in your Streamlit app:
    
    # In your main app or pages
    from app.styles.themes import apply_custom_theme, create_theme_selector
    
    # Initialize theme system
    initialize_theme_system()
    
    # Create theme selector in sidebar
    selected_theme = create_theme_selector()
    
    # Apply specific theme
    apply_custom_theme("financial")
    
    # Get chart theme for Plotly
    chart_theme = apply_chart_theme("financial")
    fig.update_layout(**chart_theme["layout"])
    """
    
    st.title("🎨 Theme System Demo")
    
    # Theme selector
    selected_theme = create_theme_selector()
    
    st.write(f"**Current Theme:** {selected_theme}")
    st.write("**Available Themes:**")
    
    for name, config in THEME_CONFIGS.items():
        st.write(f"- **{config['name']}**: {config['description']}")
    
    # Demo components
    st.subheader("Component Demos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Value", "$125,430", "+2.5%")
    with col2:
        st.metric("Daily P&L", "+$1,250", "+1.2%")
    with col3:
        st.metric("Win Rate", "68.5%", "+3.1%")
    
    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Primary Action")
    with col2:
        st.success("Success Message")
    with col3:
        st.error("Error Message")
