import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from options_analyzer import OptionsAnalyzer


st.set_page_config(
    page_title="Options Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'analyzer' not in st.session_state:
    st.session_state.analyzer = OptionsAnalyzer()
    print("Initialized OptionsAnalyzer in session state.")


def format_currency(value, currency='USD'):
    """Helper to format currency within Streamlit"""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        # Simplified version for Streamlit, analyzer's version can be used too
        return f"{currency} {value:,.2f}" if currency != 'USD' else f"${value:,.2f}"
    except (TypeError, ValueError):
        return "N/A"
    
    
    



