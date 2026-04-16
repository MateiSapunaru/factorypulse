from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="FactoryPulse Dashboard",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 FactoryPulse")
st.write(
    """
    Time-Series Anomaly Detection on Synthetic Industrial Data

    Use the sidebar to navigate through:
    - Dataset Overview
    - Anomaly Visualization
    - Model Performance
    - Inference Demo
    """
)