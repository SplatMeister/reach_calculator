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

# Display Ogilvy Logo
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
    # Meta Settings
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type=['csv'])
    meta_opts = {}
    if meta_file:
        df_meta = pd.read_csv(meta_file)
        reach_cols = [c for c in df_meta.columns if c.startswith('Reach at ') and c.endswith('frequency')]
        freq_meta = st.slider("Meta Frequency (X+)", 1, len(reach_cols), 1)
        meta_col = f"Reach at {freq_meta}+ frequency"
        if meta_col not in reach_cols:
            meta_col = reach_cols[0]
        meta_pct = st.slider("Meta: Custom Reach %", 0, 100, 70)
        meta_opts = {'df': df_meta, 'col': meta_col, 'pct': meta_pct}

    st.markdown("---")
    # Google Settings
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV or Excel", type=['csv','xlsx'])
    google_opts = {}
    if google_file:
        if google_file.name.endswith('.xlsx'):
            df_google = pd.read_excel(google_file)
        else:
            df_google = pd.read_csv(google_file)
        reach_cols = [c for c in df_google.columns if 'on-target reach' in c.lower()]
        freq_google = st.slider("Google Frequency (X+)", 1, len(reach_cols), 1)
        google_col = f"{freq_google}+ on-target reach"
        if google_col not in reach_cols:
            google_col = reach_cols[0]
        google_pct = st.slider("Google: Custom Reach %", 0, 100, 70)
        rate = st.number_input("USD → LKR rate", min_value=0.0, value=300.0)
        google_opts = {'df': df_google, 'col': google_col, 'pct': google_pct, 'rate': rate}

    st.markdown("---")
    # TV Settings
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV or Excel", type=['csv','xlsx'])
    tv_opts = {}
    if tv_file:
        if tv_file.name.endswith('.xlsx'):
            df_tv = pd.read_excel(tv_file)
        else:
            df_tv = pd.read_csv(tv_file)
        cprp = st.number_input("CPRP (LKR)", min_value=0, value=8000)
        acd = st.number_input("ACD (sec)", min_value=0, value=17)
        uni = st.number_input("Universe (pop)", min_value=0, value=11440000)
        max_reach_val = st.number_input("Max Reach (abs)", min_value=0, value=10296000)
        freq_tv = st.selectbox("TV Frequency (X+)", [f"{i}+" for i in range(1,11)])
        tv_opts = {'df': df_tv, 'cprp': cprp, 'acd': acd, 'uni': uni, 'max_reach': max_reach_val, 'freq': freq_tv}

# -------------------------------------
# Utility: Diminishing Returns Detector
# -------------------------------------
def find_optimal(df, budget_col, reach_col):
    x = df[budget_col].values
    y = df[reach_col].values
    # marginal gain
    dy = np.gradient(y, x)
    # smooth
    if len(dy) > 5:
        dy = savgol_filter(dy, 5, 2)
    # threshold at 10% of peak marginal
    thr = 0.1 * np.max(dy)
    below = np.where(dy < thr)[0]
    if below.size:
        return below[0]
    # fallback to mid-point
    return np.argmin(np.abs(x - x.mean()))

results = []

# -------------------------------------
# Meta Analysis & Plot
# -------------------------------------
if meta_opts:
    df = meta_opts['df'].copy()
    col = meta_opts['col']
    df['PrevR'] = df[col].shift(1)
    df['PrevB'] = df['Budget'].shift(1)
    df['IncR'] = df[col] - df['PrevR']
    df['IncB'] = df['Budget'] - df['PrevB']
    df['Eff'] = df['IncR'] / df['IncB']
    idx_opt = find_optimal(df, 'Budget', col)
    b_opt, r_opt, e_opt = df.at[idx_opt,'Budget'], df.at[idx_opt,col], df.at[idx_opt,'Eff']
    st.success(f"Meta optimal: {b_opt:,.0f} LKR (Eff {e_opt:.2f})")
    # custom reach marker
    cust_ix = df[df[col]/df[col].max()*100 >= meta_opts['pct']].index.min()
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Meta Reach', line=dict(color='skyblue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='Meta Efficiency', line=dict(color='royalblue', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[b_opt], y=[r_opt], mode='markers', name='Meta Opt', marker=dict(color='orange', size=12)), secondary_y=False)
    if pd.notna(cust_ix):
        cb, cr = df.at[cust_ix,'Budget'], df.at[cust_ix,col]
        fig.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Meta Custom', marker=dict(color='purple', size=10)), secondary_y=False)
    fig.update_layout(xaxis_title='Budget (LKR)', template='plotly_dark')
    fig.update_yaxes(title_text='Reach', secondary_y=False)
    fig.update_yaxes(title_text='Efficiency', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    results.append(('Meta', b_opt, r_opt, e_opt))

# -------------------------------------
# Google Analysis & Plot
# -------------------------------------
if google_opts:
    df = google_opts['df'].copy()
    col = google_opts['col']
    # numeric coercion
    df['Total Budget'] = pd.to_numeric(df['Total Budget'].astype(str).str.replace(',',''), errors='coerce')
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')
    df['Budget'] = df['Total Budget'] * google_opts['rate']
    df['PrevR'] = df[col].shift(1)
    df['PrevB'] = df['Budget'].shift(1)
    df['IncR'] = df[col] - df['PrevR']
    df['IncB'] = df['Budget'] - df['PrevB']
    df['Eff'] = df['IncR'] / df['IncB']
    idx_opt = find_optimal(df, 'Budget', col)
    b_opt, r_opt, e_opt = df.at[idx_opt,'Budget'], df.at[idx_opt,col], df.at[idx_opt,'Eff']
    st.success(f"Google optimal: {b_opt:,.0f} LKR (Eff {e_opt:.2f})")
    cust_ix = df[df[col]/df[col].max()*100 >= google_opts['pct']].index.min()
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Google Reach', line=dict(color='lightblue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='Google Efficiency', line=dict(color='orange', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[b_opt], y=[r_opt], mode='markers', name='Google Opt', marker=dict(color='red', size=12)), secondary_y=False)
    if pd.notna(cust_ix):
        cb, cr = df.at[cust_ix,'Budget'], df.at[cust_ix,col]
        fig.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Google Custom', marker=dict(color='purple', size=10)), secondary_y=False)
    fig.update_layout(xaxis_title='Budget (LKR)', template='plotly_dark')
    fig.update_yaxes(title_text='Reach', secondary_y=False)
    fig.update_yaxes(title_text='Efficiency', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    results.append(('Google', b_opt, r_opt, e_opt))

# -------------------------------------
# TV Analysis & Plot
# -------------------------------------
if tv_opts:
    df = tv_opts['df'].copy()
    # convert reach % columns
    for i in range(1,11):
        key = f"{i}+"
        if key in df:
            df[key] = pd.to_numeric(df[key].astype(str).str.replace(',',''), errors='coerce')/100 * tv_opts['uni']
    desired = tv_opts['freq'].replace('+','') + '+'
    actual_col = next((c for c in df.columns if c.replace(' ','') == desired), None)
    if not actual_col:
        st.error(f"TV column '{tv_opts['freq']}' not found.")
    else:
        df['Budget'] = df['GRPs'].astype(float) * tv_opts['cprp'] * tv_opts['acd'] / 30
        df['PrevR'] = df[actual_col].shift(1)
        df['PrevB'] = df['Budget'].shift(1)
        df['IncR'] = df[actual_col] - df['PrevR']
        df['IncB'] = df['Budget'] - df['PrevB']
        df['Eff'] = df['IncR'] / df['IncB']
        idx_opt = find_optimal(df, 'Budget', actual_col)
        b_opt, r_opt, e_opt = df.at[idx_opt,'Budget'], df.at[idx_opt,actual_col], df.at[idx_opt,'Eff']
        st.success(f"TV optimal: {b_opt:,.0f} LKR (Eff {e_opt:.2f})")
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=df['Budget'], y=df[actual_col], name='TV Reach', line=dict(color='cyan')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff'], name='TV Efficiency', line=dict(color='magenta', dash='dash')), secondary_y=True)
        fig.add_trace(go.Scatter(x=[b_opt], y=[r_opt], mode='markers', name='TV Opt', marker=dict(color='red', size=12)), secondary_y=False)
        fig.update_layout(xaxis_title='Budget (LKR)', template='plotly_dark')
        fig.update_yaxes(title_text='Reach', secondary_y=False)
        fig.update_yaxes(title_text='Efficiency', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        results.append(('TV', b_opt, r_opt, e_opt))

# -------------------------------------
# Summary Table
# -------------------------------------
st.header("Platform Summary")
if results:
    summary = pd.DataFrame(results, columns=['Platform','OptBudget','OptReach','Efficiency']).set_index('Platform')
    st.table(summary)
else:
    st.info("Upload data to analyze and see summary.")
