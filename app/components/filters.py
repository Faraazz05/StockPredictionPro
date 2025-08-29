"""
app/components/filters.py

Modular filtering components for StockPredictionPro.
Supports ranged date filtering, symbol and category selectors,
numeric sliders, and domain-specific extensions.

Author: StockPredictionPro Team
Date: 2025
"""

import streamlit as st
from typing import List, Optional, Tuple, Any
from datetime import date, timedelta
import pandas as pd


def filter_date_range(
    days: int = 90,
    min_days: int = 1,
    max_days: int = 365 * 5,
    key: str = "filter_date_range"
) -> Tuple[date, date]:
    """Filter component for selecting a date range"""
    today = date.today()
    default_start = today - timedelta(days=days)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=today - timedelta(days=max_days),
                max_value=today,
                key=f"{key}_start"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=today,
                min_value=today - timedelta(days=max_days),
                max_value=today,
                key=f"{key}_end"
            )
        if start_date > end_date:
            st.error("Start Date must be before End Date.")
    return start_date, end_date


def filter_symbols(
    symbols: List[str],
    title: str = "Symbol(s)",
    multi: bool = True,
    default: Optional[List[str]] = None,
    key: str = "filter_symbols"
) -> Any:
    """Dropdown or multiselect for stock symbols."""
    sorted_symbols = sorted(symbols)
    if multi:
        return st.multiselect(title, options=sorted_symbols, default=default or [sorted_symbols[0]], key=key)
    return st.selectbox(title, options=sorted_symbols, index=0, key=key)


def filter_numeric(
    label: str,
    vmin: float,
    vmax: float,
    step: Optional[float] = None,
    default: Optional[Tuple[float, float]] = None,
    key: str = None
) -> Tuple[float, float]:
    """Numeric slider for quantitative filtering."""
    if step is None:
        step = (vmax - vmin) / 100
    if default is None:
        default = (vmin, vmax)
    return st.slider(
        label,
        min_value=float(vmin),
        max_value=float(vmax),
        value=default,
        step=step,
        key=key or f"{label.lower().replace(' ', '_')}_range"
    )


def filter_categorical(
    label: str,
    options: List[str],
    multi: bool = True,
    default: Optional[List[str]] = None,
    key: str = None
) -> Any:
    """Dropdown or multiselect for categorical filtering."""
    if not options:
        st.warning(f"No options available for {label}.")
        return [] if multi else None
    if multi:
        return st.multiselect(label, options=options, default=default or [], key=key)
    default_idx = 0
    if default and default[0] in options:
        default_idx = options.index(default[0])
    return st.selectbox(label, options=options, index=default_idx, key=key)


# Example usage
if __name__ == "__main__":
    import streamlit as st
    example_symbols = ["AAPL", "MSFT", "GOOG", "TSLA"]
    st.title("Demo: Modular Filters")

    start, end = filter_date_range()
    selected_symbols = filter_symbols(example_symbols)
    numeric_range = filter_numeric("Price", 100.0, 500.0)
    categorical_selection = filter_categorical("Sector", ["Tech", "Finance", "Energy"])

    st.write({
        "date_range": (start, end),
        "selected_symbols": selected_symbols,
        "numeric_range": numeric_range,
        "categorical_selection": categorical_selection
    })
