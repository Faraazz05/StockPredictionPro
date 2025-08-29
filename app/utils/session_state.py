"""
app/utils/session_state.py

Session state management utilities for StockPredictionPro Streamlit app.
Provides clean, reusable functions for managing Streamlit session state
and a SessionState class for complex stateful objects.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
from typing import Any, Dict, Optional, Union
import json
from datetime import datetime



# ============================================
# BASIC SESSION STATE FUNCTIONS
# ============================================

def get_state(name: str, default: Any = None) -> Any:
    """
    Get a value from session state or return default
    
    Args:
        name: Key name in session state
        default: Default value if key doesn't exist
    
    Returns:
        Value from session state or default
    """
    return st.session_state.get(name, default)


def set_state(name: str, value: Any) -> None:
    """
    Set a value in session state
    
    Args:
        name: Key name in session state
        value: Value to store
    """
    st.session_state[name] = value


def toggle_state(name: str) -> bool:
    """
    Toggle a boolean value in session state
    
    Args:
        name: Key name in session state
    
    Returns:
        New boolean value after toggle
    """
    current = st.session_state.get(name, False)
    st.session_state[name] = not current
    return not current


def reset_state(name: str) -> None:
    """
    Reset (delete) a value from session state
    
    Args:
        name: Key name to delete
    """
    if name in st.session_state:
        del st.session_state[name]


def clear_all_state() -> None:
    """Clear all session state variables"""
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        del st.session_state[key]


def state_exists(name: str) -> bool:
    """Check if a key exists in session state"""
    return name in st.session_state


def get_state_size() -> int:
    """Get the number of items in session state"""
    return len(st.session_state)


# ============================================
# ADVANCED SESSION STATE MANAGEMENT
# ============================================

class SessionStateManager:
    """
    Advanced session state manager for complex applications
    Provides namespaced state management and persistence
    """
    
    def __init__(self, namespace: str = "stockpro"):
        """
        Initialize session state manager
        
        Args:
            namespace: Namespace prefix for state keys
        """
        self.namespace = namespace
        self._init_base_state()
    
    def _init_base_state(self) -> None:
        """Initialize base session state variables"""
        base_keys = {
            f"{self.namespace}_initialized": True,
            f"{self.namespace}_session_id": self._generate_session_id(),
            f"{self.namespace}_created_at": datetime.now().isoformat(),
            f"{self.namespace}_page_views": 0
        }
        
        for key, default_value in base_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get namespaced value from session state"""
        namespaced_key = f"{self.namespace}_{key}"
        return st.session_state.get(namespaced_key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set namespaced value in session state"""
        namespaced_key = f"{self.namespace}_{key}"
        st.session_state[namespaced_key] = value
    
    def delete(self, key: str) -> None:
        """Delete namespaced key from session state"""
        namespaced_key = f"{self.namespace}_{key}"
        if namespaced_key in st.session_state:
            del st.session_state[namespaced_key]
    
    def exists(self, key: str) -> bool:
        """Check if namespaced key exists"""
        namespaced_key = f"{self.namespace}_{key}"
        return namespaced_key in st.session_state
    
    def increment_page_views(self) -> int:
        """Increment and return page view count"""
        current_views = self.get("page_views", 0)
        new_views = current_views + 1
        self.set("page_views", new_views)
        return new_views
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            "session_id": self.get("session_id"),
            "created_at": self.get("created_at"),
            "page_views": self.get("page_views"),
            "namespace": self.namespace
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export all namespaced state as dictionary"""
        prefix = f"{self.namespace}_"
        exported = {}
        
        for key, value in st.session_state.items():
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                try:
                    # Try to serialize the value
                    json.dumps(value)
                    exported[clean_key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    exported[clean_key] = str(value)
        
        return exported
    
    def import_state(self, state_dict: Dict[str, Any]) -> None:
        """Import state from dictionary"""
        for key, value in state_dict.items():
            self.set(key, value)


# ============================================
# STATEFUL OBJECTS
# ============================================

class SessionState:
    """
    A stateful object that persists across Streamlit reruns
    Stores all attributes in session state automatically
    """
    
    def __init__(self, key: str = "default", **kwargs):
        """
        Initialize SessionState object
        
        Args:
            key: Unique key for this state object
            **kwargs: Initial state values
        """
        self._session_key = f"_session_state_obj_{key}"
        
        # Initialize or load existing state
        if self._session_key not in st.session_state:
            st.session_state[self._session_key] = {}
        
        # Set initial values
        for k, v in kwargs.items():
            if k not in st.session_state[self._session_key]:
                st.session_state[self._session_key][k] = v
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from session state"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        state_dict = st.session_state[self._session_key]
        return state_dict.get(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in session state"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            if not hasattr(self, '_session_key'):
                object.__setattr__(self, name, value)
            else:
                st.session_state[self._session_key][name] = value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return st.session_state[self._session_key].get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting"""
        st.session_state[self._session_key][key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in st.session_state[self._session_key]
    
    def keys(self):
        """Get all keys"""
        return st.session_state[self._session_key].keys()
    
    def values(self):
        """Get all values"""
        return st.session_state[self._session_key].values()
    
    def items(self):
        """Get all key-value pairs"""
        return st.session_state[self._session_key].items()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary"""
        return dict(st.session_state[self._session_key])
    
    def update(self, other: Dict[str, Any]) -> None:
        """Update from dictionary"""
        st.session_state[self._session_key].update(other)
    
    def clear(self) -> None:
        """Clear all state"""
        st.session_state[self._session_key].clear()
    
    def reset(self) -> None:
        """Reset to empty state"""
        self.clear()


# ============================================
# UTILITY FUNCTIONS
# ============================================

def init_session_defaults(**defaults) -> None:
    """
    Initialize default session state values
    
    Args:
        **defaults: Key-value pairs of default values
    """
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_or_create_state(key: str, factory_func, *args, **kwargs) -> Any:
    """
    Get existing state or create using factory function
    
    Args:
        key: Session state key
        factory_func: Function to create initial value
        *args, **kwargs: Arguments for factory function
    
    Returns:
        Existing or newly created value
    """
    if key not in st.session_state:
        st.session_state[key] = factory_func(*args, **kwargs)
    return st.session_state[key]


def batch_set_state(**kwargs) -> None:
    """
    Set multiple session state values at once
    
    Args:
        **kwargs: Key-value pairs to set
    """
    for key, value in kwargs.items():
        st.session_state[key] = value


def debug_session_state() -> None:
    """Debug function to display current session state"""
    st.write("### Current Session State")
    st.write(dict(st.session_state))


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage patterns:
    
    # Basic functions
    set_state('counter', 0)
    counter = get_state('counter', 0)
    toggle_state('show_advanced')
    
    # SessionStateManager
    manager = SessionStateManager("myapp")
    manager.set('user_id', 'user123')
    user_id = manager.get('user_id')
    
    # SessionState object
    app_state = SessionState('main_app', 
                           current_page='home',
                           selected_symbols=[],
                           last_update=None)
    
    app_state.current_page = 'analysis'
    app_state['selected_symbols'] = ['AAPL', 'MSFT']
    
    # Initialize defaults
    init_session_defaults(
        theme='dark',
        auto_refresh=True,
        chart_height=400
    )
    """
    pass
