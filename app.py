import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -------------------------------------
# Streamlit App: Omni-Channel Campaign Planner
# -------------------------------------

st.set_page_config(
    page_title="Ogilvy Planner", layout="centered", page_icon="🟥"
)

# Logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.ogilvy.com/sites/g/files/dhpsjz106/files/inline-images/Ogilvy%20Restructures.jpg" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

# Titles
st.title("Omni-Channel Campaign Planner")
st.markdown("Meta, Google & TV Data")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # ----- Meta Section -----
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type=['csv'], key="meta_csv")
    meta_df = None
    meta_freq_val = None
    meta_slider_val = None
    meta_selected_col = None

    if meta_file is not None:
        meta_df = pd.read_csv(meta_file)
        meta_reach_cols = [c for c in meta_df.columns if c.startswith('Reach at ') and c.endswith('frequency')]
        if meta_reach_cols:
            meta_freq_val = st.slider(
                "Meta: Select Frequency (Reach at X+)",
                1, 10, 1, 1, key="meta_freq"
            )
            meta_selected_col = f"Reach at {meta_freq_val}+ frequency"
            if meta_selected_col not in meta_df.columns:
                meta_selected_col = meta_reach_cols[0]

            max_r = meta_df[meta_selected_col].max()
            meta_df['Reach %'] = meta_df[meta_selected_col] / max_r * 100
            min_pct = int(meta_df['Reach %'].min())
            max_pct = int(meta_df['Reach %'].max())
            meta_slider_val = st.slider(
                "Meta: Custom Reach %", min_pct, max_pct, min(70, max_pct), 1, key="meta_slider"
            )

    st.markdown("---")

    # ----- Google Section (fixed) -----
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV or Excel", type=['csv', 'xlsx'], key="google_csv")
        conversion_rate = st.number_input(
        "USD to LKR Conversion Rate", min_value=0.0, value=300.0, step=1.0
    )
