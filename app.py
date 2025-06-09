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
    page_title="Ogilvy Planner",
    layout="centered",
    page_icon="🟥"
)

# Display Logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.ogilvy.com/sites/g/files/dhpsjz106/files/inline-images/Ogilvy%20Restructures.jpg" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

# Main Title
st.title("Omni-Channel Campaign Planner")
st.markdown("Meta, Google & TV Data")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # --- Meta Settings ---
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type=['csv'], key="meta_csv")
    meta_df = None
    meta_col = None
    meta_pct = None

    if meta_file:
        meta_df = pd.read_csv(meta_file)
        reach_cols = [c for c in meta_df.columns if c.startswith('Reach at ') and c.endswith('frequency')]
        if reach_cols:
            freq_meta = st.slider("Meta: Frequency (Reach at X+)", 1, len(reach_cols), 1)
            meta_col = f"Reach at {freq_meta}+ frequency" if f"Reach at {freq_meta}+ frequency" in reach_cols else reach_cols[0]
            # compute % reach for slider
            max_r = meta_df[meta_col].max()
            meta_df['Reach %'] = meta_df[meta_col] / max_r * 100
            min_pct = int(meta_df['Reach %'].min())
            max_pct = int(meta_df['Reach %'].max())
            meta_pct = st.slider("Meta: Custom Reach %", min_pct, max_pct, min(70, max_pct))

    st.markdown("---")

    # --- Google Settings ---
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV/Excel", type=['csv','xlsx'], key="google_csv")
    conv_rate = st.number_input("USD to LKR Rate", min_value=0.0, value=300.0)
    google_df = None
    google_col = None
    google_pct = None

    if google_file:
        google_df = pd.read_excel(google_file) if google_file.name.endswith('.xlsx') else pd.read_csv(google_file)
        reach_cols = [c for c in google_df.columns if 'on-target reach' in c.lower()]
        if reach_cols:
            # coerce numeric columns
            for col in ['Total Budget'] + reach_cols:
                if col in google_df:
                    google_df[col] = pd.to_numeric(google_df[col].astype(str).str.replace(',',''), errors='coerce')
            freq_goog = st.slider("Google: Frequency (X+ on-target reach)", 1, len(reach_cols), 1)
            google_col = f"{freq_goog}+ on-target reach" if f"{freq_goog}+ on-target reach" in reach_cols else reach_cols[0]
            # compute % reach
            max_r = google_df[google_col].max()
            google_df['Reach %'] = google_df[google_col] / max_r * 100
            min_pct = int(google_df['Reach %'].min())
            max_pct = int(google_df['Reach %'].max())
            google_pct = st.slider("Google: Custom Reach %", min_pct, max_pct, min(70, max_pct))

    st.markdown("---")

    # --- TV Settings ---
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV/Excel", type=['csv','xlsx'], key="tv_file")
    cprp = st.number_input("CPRP (LKR)", min_value=0, value=8000, step=1)
    acd  = st.number_input("ACD (sec)", min_value=0, value=17, step=1)
    universe = st.number_input("Universe (pop)", min_value=0, value=11440000, step=1)
    max_tv   = st.number_input("Max Reach (abs)", min_value=0, value=10296000, step=1)
    freq_opts = [f"{i}+" for i in range(1,11)]
    tv_freq = st.selectbox("TV: Frequency", freq_opts)

# ----------------- META ANALYSIS -----------------
st.header("Meta Data")
if meta_df is not None and meta_col:
    df = meta_df.copy()
    df['Prev Reach'] = df[meta_col].shift(1)
    df['Prev Budget'] = df['Budget'].shift(1)
    df['Inc Reach'] = df[meta_col] - df['Prev Reach']
    df['Inc Budget'] = df['Budget'] - df['Prev Budget']
    df['Efficiency'] = df['Inc Reach'] / df['Inc Budget']
    idx_opt = df['Efficiency'].idxmax()
    opt_b, opt_r, opt_e = df.at[idx_opt,'Budget'], df.at[idx_opt,meta_col], df.at[idx_opt,'Efficiency']
    st.success(f"Meta: Opt Budget → {opt_b:,.0f} LKR")
    st.write(f"Meta: Efficiency → {opt_e:.4f} reach/LKR")
    custom = df[df['Reach %']>=meta_pct].iloc[0] if meta_pct else None
    # plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[meta_col], name='Meta Reach', line=dict(color='skyblue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Efficiency'], name='Meta Efficiency', line=dict(color='royalblue', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[opt_b], y=[opt_r], mode='markers', name='Meta Opt Budget', marker=dict(color='orange', size=12)), secondary_y=False)
    fig.add_trace(go.Scatter(x=[opt_b], y=[opt_e], mode='markers', name='Meta Opt Efficiency', marker=dict(color='red', size=12)), secondary_y=True)
    if custom is not None:
        cb, cr = custom['Budget'], custom[meta_col]
        fig.add_vline(x=cb, line_dash='dot', line_color='purple')
        fig.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Meta Custom', marker=dict(color='purple', size=10)), secondary_y=False)
    fig.update_xaxes(title='Budget (LKR)')
    fig.update_yaxes(title='Reach', secondary_y=False)
    fig.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ----------------- GOOGLE ANALYSIS -----------------
st.header("Google Data")
if google_df is not None and google_col:
    dg = google_df.copy()
    dg['Budget LKR'] = dg['Total Budget'] * conv_rate
    dg['Prev Reach'] = dg[google_col].shift(1)
    dg['Prev Budget'] = dg['Budget LKR'].shift(1)
    dg['Inc Reach'] = dg[google_col] - dg['Prev Reach']
    dg['Inc Budget'] = dg['Budget LKR'] - dg['Prev Budget']
    dg['Efficiency'] = dg['Inc Reach'] / dg['Inc Budget']
    idx_opt = dg['Efficiency'].idxmax()
    gob, gor, goe = dg.at[idx_opt,'Budget LKR'], dg.at[idx_opt,google_col], dg.at[idx_opt,'Efficiency']
    st.success(f"Google: Opt Budget → {gob:,.0f} LKR")
    st.write(f"Google: Efficiency → {goe:.4f} reach/LKR")
    custom = dg[dg['Reach %']>=google_pct].iloc[0] if google_pct else None
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=dg['Budget LKR'], y=dg[google_col], name='Google Reach', line=dict(color='lightblue')), secondary_y=False)
    fig2.add_trace(go.Scatter(x=dg['Budget LKR'], y=dg['Efficiency'], name='Google Efficiency', line=dict(color='orange', dash='dash')), secondary_y=True)
    fig2.add_trace(go.Scatter(x=[gob], y=[gor], mode='markers', name='Google Opt Budget', marker=dict(color='red', size=12)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=[gob], y=[goe], mode='markers', name='Google Opt Efficiency', marker=dict(color='green', size=12)), secondary_y=True)
    if custom is not None:
        cb, cr = custom['Budget LKR'], custom[google_col]
        fig2.add_vline(x=cb, line_dash='dot', line_color='purple')
        fig2.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Google Custom', marker=dict(color='purple', size=10)), secondary_y=False)
    fig2.update_xaxes(title='Budget (LKR)')
    fig2.update_yaxes(title='Reach', secondary_y=False)
    fig2.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------- TV ANALYSIS -----------------
st.header("TV Data")
if tv_file:
    dt = pd.read_csv(tv_file) if tv_file.name.endswith('.csv') else pd.read_excel(tv_file)
    dt.columns = [c.strip() for c in dt.columns]
    for i in range(1,11):
        col = f"{i}+"
        if col in dt.columns:
            dt[col] = pd.to_numeric(dt[col], errors='coerce')/100 * universe
    # determine actual column
    actual = int(tv_freq.replace('+',''))
    col = f"{actual}+"
    dt['GRPs'] = pd.to_numeric(dt['GRPs'], errors='coerce')
    dt['Budget'] = dt['GRPs'] * cprp * acd / 30
    dt['Prev Reach'] = dt[col].shift(1)
    dt['Prev Budget'] = dt['Budget'].shift(1)
    dt['Inc Reach'] = dt[col] - dt['Prev Reach']
    dt['Inc Budget'] = dt['Budget'] - dt['Prev Budget']
    dt['Efficiency'] = dt['Inc Reach'] / dt['Inc Budget']
    idx_opt = dt['Efficiency'].idxmax()
    tob, tor, toe = dt.at[idx_opt,'Budget'], dt.at[idx_opt,col], dt.at[idx_opt,'Efficiency']
    st.success(f"TV: Opt Budget → {tob:,.0f} LKR")
    st.write(f"TV: Efficiency → {toe:.4f} reach/LKR")
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=dt['Budget'], y=dt[col], name='TV Reach', line=dict(color='cyan')), secondary_y=False)
    fig3.add_trace(go.Scatter(x=dt['Budget'], y=dt['Efficiency'], name='TV Efficiency', line=dict(color='magenta', dash='dash')), secondary_y=True)
    fig3.add_trace(go.Scatter(x=[tob], y=[tor], mode='markers', name='TV Opt Budget', marker=dict(color='red', size=12)), secondary_y=False)
    fig3.add_trace(go.Scatter(x=[tob], y=[toe], mode='markers', name='TV Opt Efficiency', marker=dict(color='green', size=12)), secondary_y=True)
    fig3.update_xaxes(title='Budget (LKR)')
    fig3.update_yaxes(title='Reach', secondary_y=False)
    fig3.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

# ----------------- SUMMARY -----------------
st.header("Platform Summary")
rows = []
if meta_file and meta_col:
    rows.append({'Platform':'Meta','Opt Budget':opt_b,'Opt Reach':opt_r,'Eff':opt_e})
if google_file and google_col:
    rows.append({'Platform':'Google','Opt Budget':gob,'Opt Reach':gor,'Eff':goe})
if tv_file:
    rows.append({'Platform':'TV','Opt Budget':tob,'Opt Reach':tor,'Eff':toe})
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
else:
    st.info("Upload data to generate summary.")
