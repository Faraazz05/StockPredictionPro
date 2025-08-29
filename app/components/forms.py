"""
app/components/forms.py

Form components for StockPredictionPro Streamlit application.
Provides reusable form components for trade entry, user configuration,
model parameters, and generic data collection forms.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from datetime import date, datetime
import pandas as pd


def trade_entry_form(
    symbols: List[str], 
    default_qty: int = 100, 
    default_price: float = 0.0, 
    key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Renders a trade entry form for placing stock trades.
    
    Args:
        symbols: List of available stock symbols
        default_qty: Default quantity for trade
        default_price: Default price for trade (0.0 for market price)
        key: Optional unique key for form
        
    Returns:
        Dict containing trade details if submitted, else None
    """
    form_key = key or "trade_entry_form"
    
    with st.form(form_key):
        st.subheader("📈 Place Trade Order")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.selectbox(
                "Symbol",
                symbols,
                key=f"{form_key}_symbol",
                help="Select the stock symbol to trade"
            )
            
            trade_type = st.selectbox(
                "Trade Type",
                ["Buy", "Sell"],
                key=f"{form_key}_type",
                help="Choose whether to buy or sell the stock"
            )
        
        with col2:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=default_qty,
                step=1,
                key=f"{form_key}_qty",
                help="Number of shares to trade"
            )
            
            order_type = st.selectbox(
                "Order Type",
                ["Market", "Limit"],
                key=f"{form_key}_order_type",
                help="Market order executes immediately, Limit order at specified price"
            )
        
        # Price input (only for limit orders)
        if order_type == "Limit":
            price = st.number_input(
                "Limit Price",
                min_value=0.01,
                value=max(default_price, 0.01),
                step=0.01,
                format="%.2f",
                key=f"{form_key}_price",
                help="Price at which to execute the limit order"
            )
        else:
            price = 0.0  # Market order
        
        # Additional options
        with st.expander("Advanced Options"):
            stop_loss = st.number_input(
                "Stop Loss Price (Optional)",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=f"{form_key}_stop_loss",
                help="Automatic sell order if price drops to this level"
            )
            
            take_profit = st.number_input(
                "Take Profit Price (Optional)",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=f"{form_key}_take_profit",
                help="Automatic sell order if price rises to this level"
            )
        
        # Calculate estimated value
        if order_type == "Market":
            st.info("💡 Market order will execute at current market price")
        else:
            estimated_value = quantity * price
            st.info(f"💰 Estimated Trade Value: ${estimated_value:,.2f}")
        
        submit = st.form_submit_button("🚀 Place Trade", type="primary")
        
        if submit:
            trade_data = {
                "symbol": symbol,
                "trade_type": trade_type,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "stop_loss": stop_loss if stop_loss > 0 else None,
                "take_profit": take_profit if take_profit > 0 else None,
                "timestamp": datetime.now(),
                "estimated_value": quantity * price if price > 0 else None
            }
            
            # Basic validation
            if trade_type == "Sell" and quantity <= 0:
                st.error("❌ Invalid quantity for sell order")
                return None
            
            if order_type == "Limit" and price <= 0:
                st.error("❌ Limit price must be greater than 0")
                return None
            
            return trade_data
    
    return None


def portfolio_settings_form(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Form for configuring portfolio settings.
    
    Args:
        key: Optional unique key for form
        
    Returns:
        Dict containing portfolio settings if submitted, else None
    """
    form_key = key or "portfolio_settings_form"
    
    with st.form(form_key):
        st.subheader("💼 Portfolio Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000,
                key=f"{form_key}_capital",
                help="Starting capital for the portfolio"
            )
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=50,
                value=10,
                key=f"{form_key}_position_size",
                help="Maximum percentage of portfolio in single position"
            )
        
        with col2:
            transaction_cost = st.number_input(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.1,
                step=0.01,
                format="%.3f",
                key=f"{form_key}_transaction_cost",
                help="Cost per trade as percentage"
            )
            
            rebalance_frequency = st.selectbox(
                "Rebalance Frequency",
                ["Daily", "Weekly", "Monthly", "Quarterly", "Never"],
                index=2,
                key=f"{form_key}_rebalance",
                help="How often to rebalance the portfolio"
            )
        
        # Risk management
        st.subheader("⚖️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_drawdown = st.slider(
                "Max Drawdown (%)",
                min_value=1,
                max_value=50,
                value=20,
                key=f"{form_key}_max_drawdown",
                help="Maximum acceptable portfolio drawdown"
            )
        
        with col2:
            stop_loss_level = st.slider(
                "Stop Loss Level (%)",
                min_value=1,
                max_value=30,
                value=10,
                key=f"{form_key}_stop_loss",
                help="Stop loss level for individual positions"
            )
        
        submit = st.form_submit_button("💾 Save Settings", type="primary")
        
        if submit:
            return {
                "initial_capital": initial_capital,
                "max_position_size": max_position_size / 100,
                "transaction_cost": transaction_cost / 100,
                "rebalance_frequency": rebalance_frequency,
                "max_drawdown": max_drawdown / 100,
                "stop_loss_level": stop_loss_level / 100,
                "timestamp": datetime.now()
            }
    
    return None


def model_parameters_form(
    model_types: List[str],
    key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Form for configuring ML model parameters.
    
    Args:
        model_types: List of available model types
        key: Optional unique key for form
        
    Returns:
        Dict containing model parameters if submitted, else None
    """
    form_key = key or "model_parameters_form"
    
    with st.form(form_key):
        st.subheader("🤖 Model Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            model_types,
            key=f"{form_key}_model_type",
            help="Select the machine learning model to use"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Training parameters
            st.subheader("📚 Training Parameters")
            
            train_split = st.slider(
                "Training Split (%)",
                min_value=50,
                max_value=90,
                value=80,
                key=f"{form_key}_train_split",
                help="Percentage of data used for training"
            )
            
            lookback_period = st.slider(
                "Lookback Period (days)",
                min_value=5,
                max_value=100,
                value=30,
                key=f"{form_key}_lookback",
                help="Number of historical days to consider"
            )
        
        with col2:
            # Model-specific parameters
            st.subheader("⚙️ Model Parameters")
            
            if model_type in ["Random Forest", "XGBoost"]:
                n_estimators = st.slider(
                    "Number of Estimators",
                    min_value=50,
                    max_value=500,
                    value=100,
                    key=f"{form_key}_n_estimators"
                )
                
                max_depth = st.slider(
                    "Max Depth",
                    min_value=3,
                    max_value=20,
                    value=10,
                    key=f"{form_key}_max_depth"
                )
            
            elif model_type == "LSTM":
                lstm_units = st.slider(
                    "LSTM Units",
                    min_value=32,
                    max_value=256,
                    value=64,
                    key=f"{form_key}_lstm_units"
                )
                
                epochs = st.slider(
                    "Epochs",
                    min_value=10,
                    max_value=200,
                    value=50,
                    key=f"{form_key}_epochs"
                )
        
        # Feature selection
        st.subheader("📊 Feature Selection")
        
        use_technical_indicators = st.checkbox(
            "Use Technical Indicators",
            value=True,
            key=f"{form_key}_tech_indicators",
            help="Include RSI, MACD, SMA, etc."
        )
        
        use_volume_data = st.checkbox(
            "Use Volume Data",
            value=True,
            key=f"{form_key}_volume_data",
            help="Include trading volume information"
        )
        
        use_sentiment_data = st.checkbox(
            "Use Sentiment Data",
            value=False,
            key=f"{form_key}_sentiment_data",
            help="Include market sentiment indicators"
        )
        
        submit = st.form_submit_button("🎯 Train Model", type="primary")
        
        if submit:
            params = {
                "model_type": model_type,
                "train_split": train_split / 100,
                "lookback_period": lookback_period,
                "use_technical_indicators": use_technical_indicators,
                "use_volume_data": use_volume_data,
                "use_sentiment_data": use_sentiment_data,
                "timestamp": datetime.now()
            }
            
            # Add model-specific parameters
            if model_type in ["Random Forest", "XGBoost"]:
                params.update({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                })
            elif model_type == "LSTM":
                params.update({
                    "lstm_units": lstm_units,
                    "epochs": epochs
                })
            
            return params
    
    return None


def general_form(
    fields: List[Dict[str, Any]], 
    form_title: str = "Form",
    key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generic form builder for custom data collection.
    
    Args:
        fields: List of field definitions
        form_title: Title for the form
        key: Optional unique key for form
        
    Returns:
        Dict containing form data if submitted, else None
    """
    form_key = key or "general_form"
    
    with st.form(form_key):
        st.subheader(f"📝 {form_title}")
        
        responses = {}
        
        for field in fields:
            label = field.get("label", "Field")
            field_type = field.get("type", "text")
            default = field.get("default")
            options = field.get("options", [])
            required = field.get("required", False)
            help_text = field.get("help")
            field_name = field.get("name", label.lower().replace(" ", "_"))
            
            field_key = f"{form_key}_{field_name}"
            
            # Add required indicator to label
            display_label = f"{label} *" if required else label
            
            if field_type == "text":
                responses[field_name] = st.text_input(
                    display_label,
                    value=default or "",
                    key=field_key,
                    help=help_text
                )
            
            elif field_type == "textarea":
                responses[field_name] = st.text_area(
                    display_label,
                    value=default or "",
                    key=field_key,
                    help=help_text
                )
            
            elif field_type == "number":
                responses[field_name] = st.number_input(
                    display_label,
                    value=default or 0,
                    key=field_key,
                    help=help_text
                )
            
            elif field_type == "select":
                index = 0
                if default and default in options:
                    index = options.index(default)
                responses[field_name] = st.selectbox(
                    display_label,
                    options,
                    index=index,
                    key=field_key,
                    help=help_text
                )
            
            elif field_type == "multiselect":
                responses[field_name] = st.multiselect(
                    display_label,
                    options,
                    default=default or [],
                    key=field_key,
                    help=help_text
                )
            
            elif field_type == "checkbox":
                responses[field_name] = st.checkbox(
                    display_label,
                    value=default or False,
                    key=field_key,
                    help=help_text
                )
            
            elif field_type == "slider":
                min_val = field.get("min", 0)
                max_val = field.get("max", 100)
                responses[field_name] = st.slider(
                    display_label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default or min_val,
                    key=field_key,
                    help=help_text
                )
            
            elif field_type == "date":
                responses[field_name] = st.date_input(
                    display_label,
                    value=default,
                    key=field_key,
                    help=help_text
                )
            
            else:
                # Default to text input
                responses[field_name] = st.text_input(
                    display_label,
                    value=default or "",
                    key=field_key,
                    help=help_text
                )
        
        submit = st.form_submit_button("Submit", type="primary")
        
        if submit:
            # Validate required fields
            missing_fields = []
            for field in fields:
                if field.get("required", False):
                    field_name = field.get("name", field.get("label", "").lower().replace(" ", "_"))
                    if not responses.get(field_name):
                        missing_fields.append(field.get("label", field_name))
            
            if missing_fields:
                st.error(f"❌ Please fill in required fields: {', '.join(missing_fields)}")
                return None
            
            responses["timestamp"] = datetime.now()
            return responses
    
    return None


# Example usage and testing
if __name__ == "__main__":
    st.title("📋 Forms Demo - StockPredictionPro")
    
    # Demo data
    demo_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    demo_models = ["Random Forest", "XGBoost", "LSTM", "Linear Regression"]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Trade Entry", "Portfolio Settings", "Model Config", "Custom Form"])
    
    with tab1:
        trade_result = trade_entry_form(demo_symbols, key="demo_trade")
        if trade_result:
            st.success("✅ Trade order submitted!")
            st.json(trade_result)
    
    with tab2:
        portfolio_result = portfolio_settings_form(key="demo_portfolio")
        if portfolio_result:
            st.success("✅ Portfolio settings saved!")
            st.json(portfolio_result)
    
    with tab3:
        model_result = model_parameters_form(demo_models, key="demo_model")
        if model_result:
            st.success("✅ Model configuration saved!")
            st.json(model_result)
    
    with tab4:
        # Custom form example
        custom_fields = [
            {
                "label": "Name",
                "type": "text",
                "required": True,
                "help": "Enter your full name"
            },
            {
                "label": "Experience Level",
                "type": "select",
                "options": ["Beginner", "Intermediate", "Advanced"],
                "default": "Beginner",
                "required": True
            },
            {
                "label": "Preferred Sectors",
                "type": "multiselect",
                "options": ["Technology", "Healthcare", "Finance", "Energy", "Consumer"],
                "help": "Select sectors you're interested in"
            },
            {
                "label": "Risk Tolerance",
                "type": "slider",
                "min": 1,
                "max": 10,
                "default": 5,
                "help": "Rate your risk tolerance from 1 (low) to 10 (high)"
            }
        ]
        
        custom_result = general_form(custom_fields, "User Profile", key="demo_custom")
        if custom_result:
            st.success("✅ Profile created!")
            st.json(custom_result)
