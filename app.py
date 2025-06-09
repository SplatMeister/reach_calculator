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

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type=['csv'])
    meta_opts = {}
    if meta_file:
        df_meta = pd.read_csv(meta_file)
        reach_cols = [c for c in df_meta.columns if c.startswith('Reach at ') and c.endswith('frequency')]
        freq = st.slider("Meta Frequency (X+)", 1, len(reach_cols), 1)
        col = f"Reach at {freq}+ frequency"
        if col not in reach_cols: col = reach_cols[0]
        pct = st.slider("Meta: Custom Reach %", 0, 100, 70)
        meta_opts = {'df': df_meta, 'col': col, 'pct': pct}

    st.markdown("---")
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV/Excel", type=['csv','xlsx'])
    google_opts = {}
    if google_file:
        df_google = pd.read_excel(google_file) if google_file.name.endswith('.xlsx') else pd.read_csv(google_file)
        reach_cols = [c for c in df_google.columns if 'on-target reach' in c.lower()]
        freq = st.slider("Google Frequency (X+)", 1, len(reach_cols), 1)
        col = f"{freq}+ on-target reach"
        if col not in reach_cols: col = reach_cols[0]
        pct = st.slider("Google: Custom Reach %", 0, 100, 70)
        rate = st.number_input("USD→LKR rate", min_value=0.0, value=300.0)
        google_opts = {'df': df_google, 'col': col, 'pct': pct, 'rate': rate}

    st.markdown("---")
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV/Excel", type=['csv','xlsx'])
    tv_opts = {}
    if tv_file:
        df_tv = pd.read_excel(tv_file) if tv_file.name.endswith('.xlsx') else pd.read_csv(tv_file)
        cprp = st.number_input("CPRP (LKR)", value=8000)
        acd = st.number_input("ACD (sec)", value=17)
        uni = st.number_input("Universe (pop)", value=11440000)
        max_r = st.number_input("Max Reach (abs)", value=10296000)
        freq_choice = st.selectbox("TV Frequency", [f"{i}+" for i in range(1,11)])
        tv_opts = {'df': df_tv, 'cprp': cprp, 'acd': acd, 'uni': uni, 'max_reach': max_r, 'freq': freq_choice}

# --- Utility: find diminishing returns point ---
def find_optimal(df, budget_col, reach_col):
    # smooth reach curve
    x = df[budget_col].values
    y = df[reach_col].values
    # compute derivative
    dy = np.gradient(y, x)
    # smooth derivative
    if len(dy) > 5:
        dy = savgol_filter(dy, 5, 2)
    # threshold: 10% of max slope
    thresh = 0.1 * np.max(dy)
    # find first index where derivative falls below threshold
    idx = np.where(dy < thresh)[0]
    if len(idx) > 0:
        opt_idx = idx[0]
    else:
        opt_idx = np.argmin(np.abs(x - x.mean()))
    return opt_idx

# --- Plot and Analyze ---
results = []

# Meta
if meta_opts:
    df = meta_opts['df'].copy()
    col = meta_opts['col']
    # compute efficiency
    df['PrevR'] = df[col].shift(1)
    df['PrevB'] = df['Budget'].shift(1)
    df['IncR'] = df[col] - df['PrevR']
    df['IncB'] = df['Budget'] - df['PrevB']
    df['Eff'] = df['IncR'] / df['IncB']
    # find optimal
    idx = find_optimal(df, 'Budget', col)
    optB, optR, optE = df.at[idx,'Budget'], df.at[idx,col], df.at[idx,'Eff']
    st.success(f"Meta optimal: {optB:,.0f} LKR, Efficiency {optE:.2f}")
    # custom reach marker
    custom_idx = df[df[col] / df[col].max() * 100 >= meta_opts['pct']].index.min() or None
    # plot
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Meta Reach', line=dict(color='skyblue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='Meta Efficiency', line=dict(color='royalblue', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[optB], y=[optR], mode='markers', name='Meta Opt', marker=dict(color='orange',size=12)), secondary_y=False)
    if custom_idx is not None:
        cb=df.at[custom_idx,'Budget']; cr=df.at[custom_idx,col]
        fig.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Meta Custom', marker=dict(color='purple',size=10)), secondary_y=False)
    fig.update_xaxes(title='Budget'); fig.update_yaxes(title='Reach', secondary_y=False); fig.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig)
    results.append(('Meta', optB, optR, optE))

# Google
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
    optB, optR, optE = df.at[idx,'Budget'], df.at[idx,col], df.at[idx,'Eff']
    st.success(f"Google optimal: {optB:,.0f} LKR, Efficiency {optE:.2f}")
    custom_idx = df[df[col] / df[col].max() * 100 >= google_opts['pct']].index.min() or None
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Google Reach', line=dict(color='lightblue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='Google Efficiency', line=dict(color='orange', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[optB], y=[optR], mode='markers', name='Google Opt', marker=dict(color='red',size=12)), secondary_y=False)
    if custom_idx is not None:
        cb=df.at[custom_idx,'Budget']; cr=df.at[custom_idx,col]
        fig.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Google Custom', marker=dict(color='purple',size=10)), secondary_y=False)
    fig.update_xaxes(title='Budget'); fig.update_yaxes(title='Reach', secondary_y=False); fig.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig)
    results.append(('Google', optB, optR, optE))

# TV
if tv_opts:
    df = tv_opts['df'].copy()
    # convert
    for i in range(1,11):
        col = f"{i}+"
        if col in df: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')/100 * tv_opts['uni']
    idx_num = int(tv_opts['freq'].replace('+',''))
    col = f"{idx_num}+"
    df['Budget'] = df['GRPs'].astype(float) * tv_opts['cprp'] * tv_opts['acd'] / 30
    df['PrevR'] = df[col].shift(1)
    df['PrevB'] = df['Budget'].shift(1)
    df['IncR'] = df[col] - df['PrevR']
    df['IncB'] = df['Budget'] - df['PrevB']
    df['Eff'] = df['IncR'] / df['IncB']
    idx = find_optimal(df, 'Budget', col)
    optB, optR, optE = df.at[idx,'Budget'], df.at[idx,col], df.at[idx,'Eff']
    st.success(f"TV optimal: {optB:,.0f} LKR, Efficiency {optE:.2f}")
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='TV Reach', line=dict(color='cyan')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='TV Efficiency', line=dict(color='magenta', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[optB], y=[optR], mode='markers', name='TV Opt', marker=dict(color='red',size=12)), secondary_y=False)
    fig.update_xaxes(title='Budget'); fig.update_yaxes(title='Reach', secondary_y=False); fig.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig)
    results.append(('TV', optB, optR, optE))

# --- Summary ---
st.header("Summary")
if results:
    summary = pd.DataFrame(results, columns=['Platform','OptBudget','OptReach','Efficiency'])
    st.table(summary.set_index('Platform'))
else:
    st.info("Upload at least one dataset to run analysis.")
