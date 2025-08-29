"""
app/styles/colors.py

Vibrant and minimalist color palette for StockPredictionPro.
Modern, clean colors designed for financial data visualization with 
excellent contrast and accessibility.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

# ============================================
# CORE BRAND COLORS
# ============================================

# Primary Colors - Professional & Vibrant
PRIMARY_BLUE = "#0052cc"      # Deep Royal Blue - Main brand color
PRIMARY_TEAL = "#00b894"      # Fresh Teal - Secondary actions
PRIMARY_CORAL = "#fd79a8"     # Vibrant Coral - Highlights & CTAs
PRIMARY_PURPLE = "#6c5ce7"    # Modern Purple - Premium features

# ============================================
# FINANCIAL STATUS COLORS
# ============================================

# Profit & Loss (High Contrast)
PROFIT_GREEN = "#00b894"      # Success Green - Clear profits
LOSS_RED = "#e17055"          # Warm Red - Losses (less harsh)
WARNING_AMBER = "#fdcb6e"     # Amber - Caution states
NEUTRAL_STEEL = "#636e72"     # Steel Gray - Neutral/No change

# ============================================
# BACKGROUND & TEXT COLORS
# ============================================

# Light Theme Colors
BACKGROUND_PRIMARY = "#ffffff"     # Pure white - Main background
BACKGROUND_SECONDARY = "#f8f9fa"   # Soft gray - Cards & sections
BACKGROUND_ACCENT = "#e9ecef"      # Light accent - Borders & dividers

# Dark Theme Colors
DARK_PRIMARY = "#1a1a1a"          # Rich black - Main background
DARK_SECONDARY = "#2d3436"        # Dark gray - Cards & sections
DARK_ACCENT = "#636e72"           # Medium gray - Borders

# Text Colors
TEXT_PRIMARY = "#2d3436"          # Almost black - Main text
TEXT_SECONDARY = "#636e72"        # Medium gray - Secondary text
TEXT_LIGHT = "#b2bec3"            # Light gray - Captions
TEXT_INVERSE = "#ffffff"          # White - Text on dark backgrounds

# ============================================
# CHART & VISUALIZATION COLORS
# ============================================

# Main Chart Palette (10 Colors) - Vibrant & Distinctive
CHART_COLORS = [
    "#0984e3",    # Electric Blue
    "#00b894",    # Mint Green  
    "#fd79a8",    # Pink
    "#fdcb6e",    # Sunny Yellow
    "#e17055",    # Coral Red
    "#a29bfe",    # Lavender
    "#55efc4",    # Aqua
    "#fd79a8",    # Rose
    "#74b9ff",    # Sky Blue
    "#81ecec"     # Pale Turquoise
]

# Technical Indicator Colors
INDICATOR_COLORS = {
    "rsi": "#fd79a8",          # Pink - RSI
    "macd": "#00b894",         # Green - MACD
    "sma": "#0984e3",          # Blue - Simple Moving Average
    "ema": "#e17055",          # Coral - Exponential Moving Average
    "bollinger": "#a29bfe",    # Lavender - Bollinger Bands
    "volume": "#636e72",       # Gray - Volume
    "support": "#55efc4",      # Aqua - Support levels
    "resistance": "#fdcb6e"    # Yellow - Resistance levels
}

# Candlestick Colors
CANDLESTICK_COLORS = {
    "bullish": "#00b894",      # Green - Up candles
    "bearish": "#e17055",      # Red - Down candles
    "neutral": "#636e72",      # Gray - Doji/unchanged
    "wick": "#2d3436"          # Dark - Wicks
}

# ============================================
# UI ELEMENT COLORS
# ============================================

# Interactive Elements
BUTTON_PRIMARY = "#0984e3"        # Blue - Primary buttons
BUTTON_SUCCESS = "#00b894"        # Green - Success actions
BUTTON_WARNING = "#fdcb6e"        # Amber - Warning actions
BUTTON_DANGER = "#e17055"         # Red - Dangerous actions

# Form Elements
INPUT_BORDER = "#ddd6fe"          # Light purple - Input borders
INPUT_FOCUS = "#a29bfe"           # Purple - Focused inputs
INPUT_ERROR = "#e17055"           # Red - Error states
INPUT_SUCCESS = "#00b894"         # Green - Success states

# ============================================
# SEMANTIC COLORS
# ============================================

# Status Colors
SUCCESS = "#00b894"               # Green - Success states
WARNING = "#fdcb6e"               # Amber - Warning states  
ERROR = "#e17055"                 # Red - Error states
INFO = "#74b9ff"                  # Blue - Information

# Market Status Colors
MARKET_OPEN = "#00b894"           # Green - Market open
MARKET_CLOSED = "#636e72"         # Gray - Market closed
MARKET_PRE = "#fdcb6e"            # Amber - Pre-market
MARKET_AFTER = "#a29bfe"          # Purple - After hours

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_chart_palette(count: int = None):
    """
    Get chart color palette
    
    Args:
        count: Number of colors needed (defaults to all)
        
    Returns:
        List of hex color codes
    """
    if count is None:
        return CHART_COLORS
    return CHART_COLORS[:count] * (count // len(CHART_COLORS) + 1)[:count]

def get_pnl_color(value: float):
    """
    Get color based on profit/loss value
    
    Args:
        value: Numeric P&L value
        
    Returns:
        Hex color code
    """
    if value > 0:
        return PROFIT_GREEN
    elif value < 0:
        return LOSS_RED
    else:
        return NEUTRAL_STEEL

def get_theme_colors(theme: str = "light"):
    """
    Get color set for specified theme
    
    Args:
        theme: Theme name ('light' or 'dark')
        
    Returns:
        Dictionary of theme colors
    """
    if theme == "dark":
        return {
            "background": DARK_PRIMARY,
            "background_secondary": DARK_SECONDARY,
            "text": TEXT_INVERSE,
            "text_secondary": TEXT_LIGHT,
            "accent": DARK_ACCENT
        }
    else:  # light theme
        return {
            "background": BACKGROUND_PRIMARY,
            "background_secondary": BACKGROUND_SECONDARY,
            "text": TEXT_PRIMARY,
            "text_secondary": TEXT_SECONDARY,
            "accent": BACKGROUND_ACCENT
        }

def get_indicator_color(indicator: str):
    """
    Get color for specific technical indicator
    
    Args:
        indicator: Indicator name (lowercase)
        
    Returns:
        Hex color code
    """
    return INDICATOR_COLORS.get(indicator.lower(), PRIMARY_BLUE)

# ============================================
# GRADIENT DEFINITIONS
# ============================================

# CSS Gradient Strings
GRADIENTS = {
    "primary": f"linear-gradient(135deg, {PRIMARY_BLUE}, {PRIMARY_TEAL})",
    "success": f"linear-gradient(135deg, {PROFIT_GREEN}, #55efc4)",
    "warning": f"linear-gradient(135deg, {WARNING_AMBER}, #ffeaa7)",
    "danger": f"linear-gradient(135deg, {LOSS_RED}, #fab1a0)",
    "purple": f"linear-gradient(135deg, {PRIMARY_PURPLE}, #a29bfe)",
    "ocean": "linear-gradient(135deg, #0984e3, #74b9ff, #00b894)"
}

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage patterns:
    
    # Basic colors
    primary_color = PRIMARY_BLUE
    success_color = PROFIT_GREEN
    
    # Chart palette
    chart_colors = get_chart_palette(5)
    all_colors = get_chart_palette()
    
    # P&L colors
    profit_color = get_pnl_color(150.50)    # Returns green
    loss_color = get_pnl_color(-75.25)      # Returns red
    
    # Theme colors
    light_theme = get_theme_colors("light")
    dark_theme = get_theme_colors("dark")
    
    # Indicator colors  
    rsi_color = get_indicator_color("rsi")   # Returns pink
    macd_color = get_indicator_color("macd") # Returns green
    """
    
    print("🎨 StockPredictionPro Color Palette")
    print("=====================================")
    print(f"Primary Blue: {PRIMARY_BLUE}")
    print(f"Success Green: {PROFIT_GREEN}")
    print(f"Warning Amber: {WARNING_AMBER}")
    print(f"Chart Colors: {len(CHART_COLORS)} vibrant colors")
    print(f"Indicator Colors: {len(INDICATOR_COLORS)} technical indicators")
    print("=====================================")
    print("Example Chart Palette (5 colors):", get_chart_palette(5))
    print("Example Chart Palette (all colors):", get_chart_palette())