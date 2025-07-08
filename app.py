import os
import streamlit as st
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from scipy.signal import savgol_filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# =============== UTILITY: DATA CLEANING ==============
def clean_data(df, budget_col, reach_col):
    df = df.copy()
    df[budget_col] = pd.to_numeric(df[budget_col], errors='coerce')
    df[reach_col] = pd.to_numeric(df[reach_col], errors='coerce')
    df = df[(df[budget_col] > 0) & (df[reach_col] >= 0)]
    df = df.dropna(subset=[budget_col, reach_col])
    df = df.sort_values(budget_col)
    df = df.loc[df.groupby(budget_col)[reach_col].idxmax()]
    df = df.reset_index(drop=True)
    return df

# =============== EFFICIENCY CALC & SMOOTHING ==============
def calc_efficiency(df, budget_col, reach_col):
    df = df.copy()
    df['PrevR'] = df[reach_col].shift(1)
    df['PrevB'] = df[budget_col].shift(1)
    df['IncR'] = df[reach_col] - df['PrevR']
    df['IncB'] = df[budget_col] - df['PrevB']
    df['Eff'] = df['IncR'] / df['IncB']
    # Drop first row, since .shift(1) gives NaN there
    df = df.dropna(subset=['Eff']).reset_index(drop=True)
    # Smooth efficiency for plotting/optimum (if enough points)
    if len(df) >= 7:
        try:
            df['Eff_smooth'] = savgol_filter(df['Eff'], 7, 3)
        except Exception:
            df['Eff_smooth'] = df['Eff']
    else:
        df['Eff_smooth'] = df['Eff']
    return df

def find_elbow(df, budget, reach, eff_col='Eff_smooth'):
    # Detect max curvature (elbow) using smoothed efficiency
    x = df[budget].values
    y = df[reach].values
    eff = df[eff_col].values
    if len(x) < 5:
        return 0
    # Use Savitzky-Golay again to smooth for curvature, if possible
    try:
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        if len(dy) >= 7 and np.all(np.isfinite(dy)):
            dy = savgol_filter(dy, 7, 3)
            d2y = savgol_filter(d2y, 7, 3)
        curvature = np.abs(d2y) / (1 + dy**2)**1.5
        if not np.any(np.isfinite(curvature)):
            return 0
        return int(np.nanargmax(curvature))
    except Exception:
        return int(np.argmax(eff)) # fallback: pick best efficiency if curvature fails

# -------------------------------------
# Streamlit App: Omni-Channel Campaign Planner
# -------------------------------------
st.set_page_config(page_title="Ogilvy Tensor", layout="centered", page_icon="🟥")

# Display Logo
st.markdown(
    """
    <div style="text-align: center;">
      <img src="https://www.ogilvy.com/sites/g/files/dhpsjz106/files/inline-images/Ogilvy%20Restructures.jpg" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Ogilvy Tensor")
st.markdown("Meta, Google & TV Data")

# =============== FREQUENCY CALCULATOR SECTION ===============

st.subheader("Frequency Calculator")

section_names = ["Brand Parameters", "Ad Parameters", "Other Parameters"]
sections = {
    "Brand Parameters": [
        "Brand Objectives - Maintain Share",
        "Brand Lifecycle - Established Brand",
        "Brand Proposition - Established",
        "Category Involvement Level - High Involvement"
    ],
    "Ad Parameters": [
        "Ad Duration - 45 sec +",
        "Ad Lifecycle - Existing Copy",
        "Ad Stock (Residue of recency) - High",
        "Executions in Campaign - Multiple",
        "Message Tonality - Simple",
        "Role of Ad - Reinforcing Attitude",
    ],
    "Other Parameters": [
        "Media Clutter - Low",
        "Competitor Activity - Low",
        "Marketing Support - Integrated"
    ]
}
all_params = [p for sec in section_names for p in sections[sec]]
for p in all_params:
    st.session_state.setdefault(f"fr_{p}", 3)
    st.session_state.setdefault(f"w_{p}", 3)
for sec in section_names:
    st.subheader(sec)
    for param in sections[sec]:
        fr_key = f"fr_{param}"
        w_key  = f"w_{param}"
        col1, col2 = st.columns([3,1])
        col1.slider(label=param, min_value=1, max_value=5, step=1, key=fr_key)
        col2.slider(label="Weight", min_value=1, max_value=6, step=1, key=w_key)
        col2.markdown(
    """
    <div style="display: flex; justify-content: space-between; font-size:13px; color:#666; margin-bottom:3px;">
        <span>Low (1)</span>
        <span>Medium (3)</span>
        <span>High (6)</span>
    </div>
    """,
    unsafe_allow_html=True
)

        
    st.markdown("---")
ratings = [st.session_state[f"fr_{p}"] for p in all_params]
weights = [st.session_state[f"w_{p}"]  for p in all_params]
scores = [round(r * w / 2, 2) for r, w in zip(ratings, weights)]
recommended_freq = round(sum(scores) / len(scores))
df = pd.DataFrame({
    "Section":        [sec for sec in section_names for _ in sections[sec]],
    "Parameter":      all_params,
    "Factor Rating":  ratings,
    "Weight":         weights,
    "Score":          scores
}).set_index(["Section", "Parameter"])
st.subheader("Results")
st.write(f"*Sum of all Factor Ratings:* {sum(ratings)}")
st.dataframe(df, use_container_width=True)
st.markdown(f"## Recommended Frequency Level: *{recommended_freq}*")
if "recommended_freq" not in st.session_state or st.session_state["recommended_freq"] != recommended_freq:
    st.session_state["recommended_freq"] = recommended_freq

# =============== SIDEBAR: FILE UPLOADS & SETTINGS ===============
#st.title("Omni-Channel Campaign Planner")
st.markdown("Meta, Google & TV Data")
with st.sidebar:
    # Meta
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type=['csv'], key='meta_csv')
    meta_opts = {}
    if meta_file:
        df_meta = pd.read_csv(meta_file)
        reach_cols = [c for c in df_meta.columns if c.startswith('Reach at ') and c.endswith('frequency')]
        if reach_cols:
            freq_init = st.session_state.get("recommended_freq", 1)
            freq_init = int(np.clip(freq_init, 1, len(reach_cols)))
            freq = st.slider("Meta Frequency (X+)", 1, len(reach_cols), freq_init)
            col = f"Reach at {freq}+ frequency"
            if col not in reach_cols:
                col = reach_cols[0]
            pct = st.slider("Meta: Custom Reach %", 0, 100, 70)
            meta_opts = {'df': df_meta, 'col': col, 'pct': pct}
        else:
            st.error("Meta file missing required reach columns.")
    # Google
    st.markdown("---")
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV/XLSX", type=['csv','xlsx'], key='google_csv')
    google_opts = {}
    if google_file:
        if str(google_file.name).endswith('.xlsx'):
            df_google = pd.read_excel(google_file)
        else:
            try:
                df_google = pd.read_csv(google_file)
            except (UnicodeDecodeError, EmptyDataError):
                df_google = pd.read_csv(google_file, encoding='latin-1')
        reach_cols = [c for c in df_google.columns if 'on-target reach' in c.lower()]
        if reach_cols:
            freq_init = st.session_state.get("recommended_freq", 1)
            freq_init = int(np.clip(freq_init, 1, len(reach_cols)))
            freq = st.slider("Google Frequency (X+)", 1, len(reach_cols), freq_init)
            col = f"{freq}+ on-target reach"
            if col not in reach_cols:
                col = reach_cols[0]
            pct = st.slider("Google: Custom Reach %", 0, 100, 70)
            rate = st.number_input("USD → LKR rate", min_value=0.0, value=300.0)
            google_opts = {'df': df_google, 'col': col, 'pct': pct, 'rate': rate}
        else:
            st.error("Google file missing required reach columns.")
    # TV
    st.markdown("---")
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV/XLSX", type=['csv','xlsx'], key='tv_csv')
    tv_opts = {}
    if tv_file:
        if str(tv_file.name).endswith('.xlsx'):
            df_tv = pd.read_excel(tv_file)
        else:
            try:
                df_tv = pd.read_csv(tv_file)
            except (UnicodeDecodeError, EmptyDataError):
                df_tv = pd.read_csv(tv_file, encoding='latin-1')
        cprp = st.number_input("CPRP (LKR)", min_value=0, value=8000)
        acd = st.number_input("ACD (sec)", min_value=0, value=17)
        uni = st.number_input("Universe (pop)", min_value=0, value=11440000)
        max_r = st.number_input("Max Reach (abs)", min_value=0, value=10296000)
        freq_init = st.session_state.get("recommended_freq", 1)
        freq_init = int(np.clip(freq_init, 1, 10))
        freq = st.slider("TV Frequency (X+)", 1, 10, freq_init)
        key = f"{freq}+"
        actual_col = next((c for c in df_tv.columns if c.replace(' ', '') == key), None)
        if not actual_col:
            st.error(f"TV column '{key}' not found.")
        pct = None
        if actual_col:
            vals = pd.to_numeric(df_tv[actual_col].astype(str).str.replace(',',''), errors='coerce') / 100 * uni
            prct = vals / max_r * 100
            pct = st.slider("TV: Custom Reach %", int(prct.min()), int(prct.max()), 70)
        tv_opts = {'df': df_tv, 'col': actual_col, 'cprp': cprp, 'acd': acd, 'uni': uni, 'max_reach': max_r, 'pct': pct}

results = []

# =============== ANALYSIS & PLOTS ===============
if meta_opts:
    df = clean_data(meta_opts['df'], 'Budget', meta_opts['col'])
    if not df.empty:
        df = calc_efficiency(df, 'Budget', meta_opts['col'])
        col, pct = meta_opts['col'], meta_opts['pct']
        idx = find_elbow(df, 'Budget', col)
        b, r, e = df.at[idx, 'Budget'], df.at[idx, col], df.at[idx, 'Eff_smooth']
        st.success(f"Meta optimal: {b:,.0f} LKR | Eff: {e:.2f}")
        max_r = df[col].max()
        custom = df[df[col] / max_r * 100 >= pct].iloc[0] if not df[df[col] / max_r * 100 >= pct].empty else None
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Meta Reach', line=dict(color='#EB3F43')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff_smooth'], name='Meta Smoothed Efficiency', line=dict(color='#F58E8F', dash='dash')), secondary_y=True)
        fig.add_trace(go.Scatter(x=[b], y=[r], mode='markers', name='Meta Optimum', marker=dict(color='orange', size=12)), secondary_y=False)
        if custom is not None:
            fig.add_vline(x=custom['Budget'], line_dash='dot', line_color='purple', annotation_text=f"{pct}%", annotation_position='top left')
            fig.add_trace(go.Scatter(x=[custom['Budget']], y=[custom[col]], mode='markers+text', name='Meta Custom', marker=dict(color='purple', size=10), text=[f"{custom[col]:,.0f}"], textposition='bottom center'), secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)
        results.append(('Meta', b, r, e))

if google_opts:
    df = google_opts['df'].copy()
    col, pct = google_opts['col'], google_opts['pct']
    df['Total Budget'] = pd.to_numeric(df['Total Budget'].astype(str).str.replace(',',''), errors='coerce')
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')
    df['Budget'] = df['Total Budget'] * google_opts['rate']
    df = clean_data(df, 'Budget', col)
    if not df.empty:
        df = calc_efficiency(df, 'Budget', col)
        idx = find_elbow(df, 'Budget', col)
        b, r, e = df.at[idx, 'Budget'], df.at[idx, col], df.at[idx, 'Eff_smooth']
        st.success(f"Google optimal: {b:,.0f} LKR | Eff: {e:.2f}")
        max_r = df[col].max()
        custom = df[df[col] / max_r * 100 >= pct].iloc[0] if not df[df[col] / max_r * 100 >= pct].empty else None
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='Google Reach', line=dict(color='#EB3F43')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff_smooth'], name='Google Smoothed Efficiency', line=dict(color='#F58E8F', dash='dash')), secondary_y=True)
        fig.add_trace(go.Scatter(x=[b], y=[r], mode='markers', name='Google Optimum', marker=dict(color='red', size=12)), secondary_y=False)
        if custom is not None:
            fig.add_vline(x=custom['Budget'], line_dash='dot', line_color='purple', annotation_text=f"{pct}%", annotation_position='top left')
            fig.add_trace(go.Scatter(x=[custom['Budget']], y=[custom[col]], mode='markers+text', name='Google Custom', marker=dict(color='purple', size=10), text=[f"{custom[col]:,.0f}"], textposition='bottom center'), secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)
        results.append(('Google', b, r, e))

if tv_opts:
    df = tv_opts['df'].copy()
    col, pct = tv_opts['col'], tv_opts['pct']
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce') / 100 * tv_opts['uni']
    df['Budget'] = df['GRPs'].astype(float) * tv_opts['cprp'] * tv_opts['acd'] / 30
    df = clean_data(df, 'Budget', col)
    if not df.empty:
        df = calc_efficiency(df, 'Budget', col)
        idx = find_elbow(df, 'Budget', col)
        b, r, e = df.at[idx, 'Budget'], df.at[idx, col], df.at[idx, 'Eff_smooth']
        st.success(f"TV optimal: {b:,.0f} LKR | Eff: {e:.2f}")
        max_r = df[col].max()
        custom = df[df[col] / max_r * 100 >= pct].iloc[0] if pct is not None and not df[df[col] / max_r * 100 >= pct].empty else None
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=df['Budget'], y=df[col], name='TV Reach', line=dict(color='#EB3F43')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Budget'], y=df['Eff_smooth'], name='TV Smoothed Efficiency', line=dict(color='#F58E8F', dash='dash')), secondary_y=True)
        fig.add_trace(go.Scatter(x=[b], y=[r], mode='markers', name='TV Optimum', marker=dict(color='green', size=12)), secondary_y=False)
        if custom is not None:
            fig.add_vline(x=custom['Budget'], line_dash='dot', line_color='purple', annotation_text=f"{pct}%", annotation_position='top left')
            fig.add_trace(go.Scatter(x=[custom['Budget']], y=[custom[col]], mode='markers+text', name='TV Custom', marker=dict(color='purple', size=10), text=[f"{int(custom[col])}"], textposition='bottom center'), secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)
        results.append(('TV', b, r, e))

# =============== SUMMARY TABLE ===============
st.header("Platform Summary")
if results:
    summary = []
    # Meta
    if any(r[0] == 'Meta' for r in results):
        dfm = clean_data(meta_opts['df'], 'Budget', meta_opts['col'])
        if not dfm.empty:
            dfm = calc_efficiency(dfm, 'Budget', meta_opts['col'])
            reach_col = meta_opts['col']
            max_reach = dfm[reach_col].max()
            max_idx = dfm[reach_col].idxmax()
            max_budget = dfm.at[max_idx, 'Budget']
            opt_idx = find_elbow(dfm, 'Budget', reach_col)
            opt_reach = dfm.at[opt_idx, reach_col]
            opt_budget = dfm.at[opt_idx, 'Budget']
            pct = meta_opts['pct']
            cust = dfm[dfm[reach_col] / max_reach * 100 >= pct]
            if not cust.empty:
                cust_reach = cust.iloc[0][reach_col]
                cust_budget = cust.iloc[0]['Budget']
            else:
                cust_reach = np.nan
                cust_budget = np.nan
            summary.append({
                'Platform': 'Meta',
                'Maximum Reach': f"{max_reach:,.0f}",
                'Budget @ Max Reach (LKR)': f"{max_budget:,.0f}",
                'Optimum Reach': f"{opt_reach:,.0f}",
                'Optimum Budget (LKR)': f"{opt_budget:,.0f}",
                'Custom Reach': f"{cust_reach:,.0f}",
                'Custom Budget (LKR)': f"{cust_budget:,.0f}" if not np.isnan(cust_budget) else ""
            })
    # Google
    if any(r[0] == 'Google' for r in results):
        dfg = google_opts['df'].copy()
        col = google_opts['col']
        dfg['Total Budget'] = pd.to_numeric(dfg['Total Budget'].astype(str).str.replace(',',''), errors='coerce')
        dfg[col] = pd.to_numeric(dfg[col].astype(str).str.replace(',',''), errors='coerce')
        dfg['Budget'] = dfg['Total Budget'] * google_opts['rate']
        dfg = clean_data(dfg, 'Budget', col)
        if not dfg.empty:
            dfg = calc_efficiency(dfg, 'Budget', col)
            max_reach = dfg[col].max()
            max_idx = dfg[col].idxmax()
            max_budget = dfg.at[max_idx, 'Budget']
            opt_idx = find_elbow(dfg, 'Budget', col)
            opt_reach = dfg.at[opt_idx, col]
            opt_budget = dfg.at[opt_idx, 'Budget']
            pct = google_opts['pct']
            cust = dfg[dfg[col] / max_reach * 100 >= pct]
            if not cust.empty:
                cust_reach = cust.iloc[0][col]
                cust_budget = cust.iloc[0]['Budget']
            else:
                cust_reach = np.nan
                cust_budget = np.nan
            summary.append({
                'Platform': 'Google',
                'Maximum Reach': f"{max_reach:,.0f}",
                'Budget @ Max Reach (LKR)': f"{max_budget:,.0f}",
                'Optimum Reach': f"{opt_reach:,.0f}",
                'Optimum Budget (LKR)': f"{opt_budget:,.0f}",
                'Custom Reach': f"{cust_reach:,.0f}",
                'Custom Budget (LKR)': f"{cust_budget:,.0f}" if not np.isnan(cust_budget) else ""
            })
    # TV
    if any(r[0] == 'TV' for r in results):
        dft = tv_opts['df'].copy()
        col = tv_opts['col']
        dft[col] = pd.to_numeric(dft[col].astype(str).str.replace(',',''), errors='coerce') / 100 * tv_opts['uni']
        dft['Budget'] = dft['GRPs'].astype(float) * tv_opts['cprp'] * tv_opts['acd'] / 30
        dft = clean_data(dft, 'Budget', col)
        if not dft.empty:
            dft = calc_efficiency(dft, 'Budget', col)
            max_reach = dft[col].max()
            max_idx = dft[col].idxmax()
            max_budget = dft.at[max_idx, 'Budget']
            opt_idx = find_elbow(dft, 'Budget', col)
            opt_reach = dft.at[opt_idx, col]
            opt_budget = dft.at[opt_idx, 'Budget']
            pct = tv_opts['pct']
            cust = dft[dft[col] / max_reach * 100 >= pct] if pct is not None else pd.DataFrame()
            if not cust.empty:
                cust_reach = cust.iloc[0][col]
                cust_budget = cust.iloc[0]['Budget']
            else:
                cust_reach = np.nan
                cust_budget = np.nan
            summary.append({
                'Platform': 'TV',
                'Maximum Reach': f"{max_reach:,.0f}",
                'Budget @ Max Reach (LKR)': f"{max_budget:,.0f}",
                'Optimum Reach': f"{opt_reach:,.0f}",
                'Optimum Budget (LKR)': f"{opt_budget:,.0f}",
                'Custom Reach': f"{cust_reach:,.0f}",
                'Custom Budget (LKR)': f"{cust_budget:,.0f}" if not np.isnan(cust_budget) else ""
            })
    df_sum = pd.DataFrame(summary).set_index('Platform')
    totals = {}
    for col in df_sum.columns:
        vals = []
        for v in df_sum[col]:
            try:
                vals.append(float(str(v).replace(',','')))
            except:
                vals.append(0.0)
        totals[col] = f"{sum(vals):,.0f}"
    df_sum.loc['Total'] = totals
    st.dataframe(df_sum)
else:
    st.info("Upload data and select settings to view summary.")

# =============== Output Visualization ===============

#chart 1
# Prepare data (ensure df_sum is in memory, remove 'Total' row for plotting)
df_plot = df_sum.drop('Total', errors='ignore').copy()
for col in df_plot.columns:
    df_plot[col] = df_plot[col].str.replace(',', '').replace('', '0').astype(float)
platforms = df_plot.index.tolist()

fig = go.Figure()

# Add scatter points for each platform & metric
colors = {'Meta':'#2471A3','Google':'#229954','TV':'#CB4335'}
markers = ['circle', 'square', 'diamond']
metrics = [
    ("Optimum Budget (LKR)", "Optimum Reach"),
    ("Custom Budget (LKR)", "Custom Reach"),
    ("Budget @ Max Reach (LKR)", "Maximum Reach"),
]
metric_names = ["Optimum", "Custom", "Max"]

for i, plat in enumerate(platforms):
    for j, (b_col, r_col) in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=[df_plot.loc[plat, b_col]],
            y=[df_plot.loc[plat, r_col]],
            name=f"{plat} {metric_names[j]}",
            mode='markers+text',
            marker=dict(size=18, color=colors[plat], symbol=markers[j]),
            text=[f"{plat}"],
            textposition='bottom right',
            hovertemplate=(
                f"<b>{plat} {metric_names[j]}</b><br>"
                "Budget: %{x:,.0f} LKR<br>"
                "Reach: %{y:,.0f}<extra></extra>"
            )
        ))

fig.update_layout(
    title="Platform Budget vs Reach",
    xaxis_title="Budget (LKR)",
    yaxis_title="Reach (Absolute)",
    template="plotly_dark",
    legend_title="Platform & Metric",
    hovermode="closest",
    height=600,
    margin=dict(l=40, r=40, t=80, b=40)
)
st.subheader("Platform Budget vs Reach")
st.plotly_chart(fig, use_container_width=True)

#chart 2

df_plot = df_sum.drop('Total', errors='ignore').copy()
for col in df_plot.columns:
    df_plot[col] = df_plot[col].str.replace(',', '').replace('', '0').astype(float)
df_plot = df_plot.reset_index()

fig = px.scatter(
    df_plot,
    x="Optimum Budget (LKR)",
    y="Optimum Reach",
    size="Maximum Reach",
    color="Platform",
    hover_data=["Custom Budget (LKR)", "Custom Reach"],
    text="Platform",
    template="plotly_dark",
    labels={
        "Optimum Budget (LKR)": "Optimum Budget (LKR)",
        "Optimum Reach": "Optimum Reach"
    },
    title="Platform Optimums (Bubble = Max Reach)"
)
fig.update_traces(textposition='top center')
st.subheader("Bubble Chart: Platform Optimums")
st.plotly_chart(fig, use_container_width=True)


