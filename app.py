import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -------------------------------------
# Streamlit App: Omni-Channel Campaign Planner
# -------------------------------------

st.set_page_config(
    page_title="Ogilvy Planner",
    layout="centered",
    page_icon="🟥"
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

st.title("Omni-Channel Campaign Planner")
st.markdown("Meta, Google & TV Data")

# -------------------------------------
# Sidebar: Upload & Settings
# -------------------------------------
with st.sidebar:
    # Meta
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type='csv', key='meta_csv')
    if not meta_file and os.path.exists('/mnt/data/Meta.csv'):
        meta_file = '/mnt/data/Meta.csv'
    meta_opts = {}
    if meta_file:
        df_meta = pd.read_csv(meta_file)
        reach_cols = [c for c in df_meta.columns if c.startswith('Reach at ') and c.endswith('frequency')]
        if reach_cols:
            freq_meta = st.slider("Meta Frequency (X+)", 1, len(reach_cols), 1)
            meta_col = f"Reach at {freq_meta}+ frequency"
            if meta_col not in reach_cols:
                meta_col = reach_cols[0]
            meta_pct = st.slider("Meta: Custom Reach %", 0, 100, 70)
            meta_opts = {'df': df_meta, 'col': meta_col, 'pct': meta_pct}
        else:
            st.error("Meta file missing required reach columns.")

    st.markdown("---")
    # Google
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV/XLSX", type=['csv','xlsx'], key='google_csv')
    if not google_file and os.path.exists('/mnt/data/Google.xlsx'):
        google_file = '/mnt/data/Google.xlsx'
    google_opts = {}
    if google_file:
        # read file
        if isinstance(google_file, str) and google_file.lower().endswith('.xlsx'):
            df_google = pd.read_excel(google_file)
        else:
            df_google = pd.read_csv(google_file)
        reach_cols = [c for c in df_google.columns if 'on-target reach' in c.lower()]
        if reach_cols:
            freq_goog = st.slider("Google Frequency (X+)", 1, len(reach_cols), 1)
            goog_col = f"{freq_goog}+ on-target reach"
            if goog_col not in reach_cols:
                goog_col = reach_cols[0]
            goog_pct = st.slider("Google: Custom Reach %", 0, 100, 70)
            rate = st.number_input("USD → LKR rate", min_value=0.0, value=300.0)
            google_opts = {'df': df_google, 'col': goog_col, 'pct': goog_pct, 'rate': rate}
        else:
            st.error("Google file missing required reach columns.")

    st.markdown("---")
    # TV
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV/XLSX", type=['csv','xlsx'], key='tv_csv')
    if not tv_file and os.path.exists('/mnt/data/tv.xlsx'):
        tv_file = '/mnt/data/tv.xlsx'
    tv_opts = {}
    if tv_file:
        # read file with encoding fallback
        if isinstance(tv_file, str) and tv_file.lower().endswith('.xlsx'):
            df_tv = pd.read_excel(tv_file)
        else:
            try:
                df_tv = pd.read_csv(tv_file)
            except UnicodeDecodeError:
                df_tv = pd.read_csv(tv_file, encoding='latin-1')
        cprp = st.number_input("CPRP (LKR)", min_value=0, value=8000)
        acd = st.number_input("ACD (sec)", min_value=0, value=17)
        uni = st.number_input("Universe (pop) 📊", min_value=0, value=11440000)
        max_r = st.number_input("Max Reach (abs)", min_value=0, value=10296000)
        # frequency slider
        freq_tv = st.slider("TV Frequency (X+)", 1, 10, 1)
        col_tv = f"{freq_tv}+"
        # custom reach slider
        tv_pct = None
        if col_tv in df_tv.columns:
            vals = pd.to_numeric(df_tv[col_tv].astype(str).str.replace(',',''), errors='coerce') / 100 * uni
            pct = vals / max_r * 100
            tv_pct = st.slider("TV: Custom Reach %", int(pct.min()), int(pct.max()), 70)
        else:
            st.error(f"TV column '{col_tv}' not found.")
        tv_opts = {'df': df_tv, 'cprp': cprp, 'acd': acd, 'uni': uni, 'max_reach': max_r, 'col': col_tv, 'pct': tv_pct}

# -------------------------------------
# Diminishing Returns Detector
# -------------------------------------
def find_optimal(df, budget, reach):
    x = df[budget].values
    y = df[reach].values
    dy = np.gradient(y, x)
    if len(dy) > 5:
        dy = savgol_filter(dy, 5, 2)
    thr = 0.1 * np.max(dy)
    idx = np.where(dy < thr)[0]
    return int(idx[0]) if idx.size else int(np.argmin(np.abs(x - x.mean())))

results = []

# -------------------------------------
# Meta Analysis
# -------------------------------------
if meta_opts:
    df = meta_opts['df'].copy()
    col = meta_opts['col']
    df['PrevR'] = df[col].shift(1)
    df['PrevB'] = df['Budget'].shift(1)
    df['IncR'] = df[col] - df['PrevR']
    df['IncB'] = df['Budget'] - df['PrevB']
    df['Eff'] = df['IncR'] / df['IncB']
    idx = find_optimal(df, 'Budget', col)
    b, r, e = df.at[idx,'Budget'], df.at[idx,col], df.at[idx,'Eff']
    st.success(f"Meta optimal: {b:,.0f} LKR | Eff: {e:.2f}")
    # plot
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Reach'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='Efficiency'), secondary_y=True)
    fig.add_trace(go.Scatter(x=[b], y=[r], mode='markers', marker=dict(color='orange',size=12), name='Optimum'), secondary_y=False)
    st.plotly_chart(fig, use_container_width=True)
    results.append(('Meta', b, r, e))

# -------------------------------------
# Google Analysis
# -------------------------------------
if google_opts:
    df = google_opts['df'].copy()
    col = google_opts['col']
    df['Total Budget'] = pd.to_numeric(df['Total Budget'].astype(str).str.replace(',',''), errors='coerce')
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')
    df['Budget'] = df['Total Budget'] * google_opts['rate']
    df['PrevR'] = df[col].shift(1)
    df['PrevB'] = df['Budget'].shift(1)
    df['IncR'] = df[col] - df['PrevR']
    df['IncB'] = df['Budget'] - df['PrevB']
    df['Eff'] = df['IncR'] / df['IncB']
    idx = find_optimal(df, 'Budget', col)
    b, r, e = df.at[idx,'Budget'], df.at[idx,col], df.at[idx,'Eff']
    st.success(f"Google optimal: {b:,.0f} LKR | Eff: {e:.2f}")
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Reach'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='Efficiency'), secondary_y=True)
    fig.add_trace(go.Scatter(x=[b], y=[r], mode='markers', marker=dict(color='red',size=12), name='Optimum'), secondary_y=False)
    st.plotly_chart(fig, use_container_width=True)
    results.append(('Google', b, r, e))

# -------------------------------------
# TV Analysis
# -------------------------------------
if tv_opts:
    df = tv_opts['df'].copy()
    col = tv_opts['col']
    # convert % to absolute reach
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')/100 * tv_opts['uni']
    df['Budget'] = df['GRPs'].astype(float)*tv_opts['cprp']*tv_opts['acd']/30
    df['PrevR'] = df[col].shift(1)
    df['PrevB'] = df['Budget'].shift(1)
    df['IncR'] = df[col]-df['PrevR']
    df['IncB'] = df['Budget']-df['PrevB']
    df['Eff'] = df['IncR']/df['IncB']
    idx = find_optimal(df, 'Budget', col)
    b, r, e = df.at[idx,'Budget'], df.at[idx,col], df.at[idx,'Eff']
    st.success(f"TV optimal: {b:,.0f} LKR | Eff: {e:.2f}")
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Reach'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='Efficiency'), secondary_y=True)
    fig.add_trace(go.Scatter(x=[b], y=[r], mode='markers', marker=dict(color='green',size=12), name='Optimum'), secondary_y=False)
    st.plotly_chart(fig, use_container_width=True)
    results.append(('TV', b, r, e))

# -------------------------------------
# Summary
# -------------------------------------
st.header("Summary")
if results:
    df_sum = pd.DataFrame(results, columns=['Platform','Budget','Reach','Efficiency']).set_index('Platform')
    st.table(df_sum)
else:
    st.info("Upload and select at least one dataset to see results.")
