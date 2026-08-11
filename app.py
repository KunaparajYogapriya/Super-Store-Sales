"""
Superstore Sales Intelligence - Main Streamlit Application Entry Point.
"""

import os
import sys
import pandas as pd
import streamlit as st
import joblib

# Ensure src/ directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, get_dataset_summary

# Page Configuration
st.set_page_config(
    page_title="Superstore Sales Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Cached Data & Model Loaders
@st.cache_data
def get_cached_data():
    try:
        df = load_data()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def load_saved_model(model_filename: str):
    model_path = os.path.join(os.path.dirname(__file__), "models", model_filename)
    if os.path.exists(model_path):
        try:
            artifact = joblib.load(model_path)
            return artifact
        except Exception as e:
            st.error(f"Failed to load model {model_filename}: {e}")
            return None
    return None


def main():
    # Sidebar Header
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 0.5rem 0 1.5rem 0;'>
            <h2 style='margin:0; color:#0f172a; font_weight:700; font-size:1.4rem;'>💼 Superstore AI</h2>
            <p style='margin:0; color:#64748b; font-size:0.85rem;'>Sales & ML Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.divider()
    
    # Load dataset to session state
    df = get_cached_data()
    if df is not None:
        st.session_state['df'] = df
        summary = get_dataset_summary(df)
        st.sidebar.caption(f"**Dataset Active:** {summary['num_rows']:,} Rows | {summary['start_date'][:4]}-{summary['end_date'][:4]}")
    else:
        st.sidebar.error("Dataset inactive or missing.")
        
    st.sidebar.info("Use the navigation menu above to explore interactive analytics, predictions, and forecasting.")
    
    # Redirect to Home page content if run directly
    st.title("Superstore Sales Intelligence")
    st.markdown("### Welcome to the End-to-End Enterprise Analytics & Machine Learning Platform")
    st.write("Please select a module from the sidebar navigation to begin.")

if __name__ == "__main__":
    main()
