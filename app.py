import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -------------------------
# Streamlit App: Omni-Channel Campaign Planner
# -------------------------

st.set_page_config(page_title="Ogilvy Planner", layout="centered", page_icon="🟥")

# Logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.ogilvy.com/sites/g/files/dhpsjz106/files/inline-images/Ogilvy%20Restructures.jpg" width="300">
    </div>
    """, unsafe_allow_html=True
)

# Title
st.title("Omni-Channel Campaign Planner")
st.markdown("Meta, Google & TV Data")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # --- Meta Settings ---
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type=['csv'], key="meta_csv")
    meta_df = None
    meta_freq = None
    meta_col = None
    meta_pct = None
    if meta_file:
        meta_df = pd.read_csv(meta_file)
        # detect reach columns
        reach_cols = [c for c in meta_df.columns if c.startswith('Reach at ') and c.endswith('frequency')]
        if reach_cols:
            meta_freq = st.slider("Meta: Frequency (Reach at X+)", 1, len(reach_cols), 1, step=1)
            meta_col = f"Reach at {meta_freq}+ frequency" if f"Reach at {meta_freq}+ frequency" in reach_cols else reach_cols[0]
            # custom reach slider
            max_reach = meta_df[meta_col].max()
            meta_df['Reach %'] = meta_df[meta_col] / max_reach * 100
            min_pct, max_pct = int(meta_df['Reach %'].min()), int(meta_df['Reach %'].max())
            meta_pct = st.slider("Meta: Custom Reach %", min_pct, max_pct, min(70, max_pct))
    st.markdown("---")

    # --- Google Settings ---
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV or Excel", type=['csv','xlsx'], key="google_csv")
    conv = st.number_input("USD to LKR Conversion Rate", min_value=0.0, value=300.0, step=1.0)
    google_df = None
    google_freq = None
    google_col = None
    google_pct = None
    if google_file:
        google_df = pd.read_excel(google_file) if google_file.name.endswith('.xlsx') else pd.read_csv(google_file)
        # find on-target reach cols
        reach_cols = [c for c in google_df.columns if 'on-target reach' in c.lower()]
        if reach_cols:
            # coerce numeric
            for col in ['Total Budget'] + reach_cols:
                if col in google_df:
                    google_df[col] = pd.to_numeric(google_df[col].astype(str).str.replace(',',''), errors='coerce')
            google_freq = st.slider("Google: Frequency (X+ on-target reach)", 1, len(reach_cols), 1, step=1)
            google_col = f"{google_freq}+ on-target reach" if f"{google_freq}+ on-target reach" in reach_cols else reach_cols[0]
            # custom reach slider
            max_r = google_df[google_col].max()
            google_df['Reach %'] = google_df[google_col] / max_r * 100
            min_pct, max_pct = int(google_df['Reach %'].min()), int(google_df['Reach %'].max())
            google_pct = st.slider("Google: Custom Reach %", min_pct, max_pct, min(70, max_pct))
    st.markdown("---")

    # --- TV Settings ---
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV or Excel", type=['csv','xlsx'], key="tv_file")
    cprp = st.text_input("CPRP (LKR)", "8000")
    acd  = st.text_input("ACD (sec)", "17")
    universe = st.text_input("Universe (pop)", "11440000")
    max_tv = st.text_input("Max Reach (abs)", "10296000")
    # parse ints
    def to_int(x):
        try: return int(str(x).replace(',',''))
        except: return 0
    cprp, acd, universe, max_tv = to_int(cprp), to_int(acd), to_int(universe), to_int(max_tv)
    freq_opts = [f"{i}+" for i in range(1,11)]
    tv_freq = st.selectbox("TV: Frequency", freq_opts)

# ----------------- META ANALYSIS -----------------
st.header("Meta Data")
if meta_df is not None and meta_col:
    dfm = meta_df.copy()
    # Efficiency = incremental reach / incremental budget
    dfm['Prev Reach'] = dfm[meta_col].shift(1)
    dfm['Prev Budget'] = dfm['Budget'].shift(1)
    dfm['Inc Reach'] = dfm[meta_col] - dfm['Prev Reach']
    dfm['Inc Budget'] = dfm['Budget'] - dfm['Prev Budget']
    dfm['Efficiency'] = dfm['Inc Reach'] / dfm['Inc Budget']
    # optimal = max efficiency
    idx_opt = dfm['Efficiency'].idxmax()
    opt_b, opt_r, opt_e = dfm.at[idx_opt,'Budget'], dfm.at[idx_opt,meta_col], dfm.at[idx_opt,'Efficiency']
    st.success(f"Meta: Opt Budget → {opt_b:,.0f} LKR")
    st.write(f"Meta: Efficiency → {opt_e:.4f} reach/LKR")
    # custom reach row
    custom = dfm[dfm['Reach %']>=meta_pct].iloc[0] if meta_pct else None
    # plot
    fig = make_subplots(specs=[[{{"secondary_y":True}}]])
    fig.add_trace(go.Scatter(x=dfm['Budget'], y=dfm[meta_col], name='Meta Reach', line=dict(color='skyblue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=dfm['Budget'], y=dfm['Efficiency'], name='Meta Efficiency', line=dict(color='royalblue', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[opt_b], y=[opt_r], mode='markers', name='Meta Opt Budget', marker=dict(color='orange', size=12)), secondary_y=False)
    fig.add_trace(go.Scatter(x=[opt_b], y=[opt_e], mode='markers', name='Meta Opt Efficiency', marker=dict(color='red', size=12)), secondary_y=True)
    if custom is not None:
        cb, cr = custom['Budget'], custom[meta_col]
        fig.add_vline(x=cb, line_dash='dot', line_color='purple')
        fig.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Meta Custom'), secondary_y=False)
    fig.update_xaxes(title='Budget (LKR)')
    fig.update_yaxes(title='Reach', secondary_y=False)
    fig.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ----------------- GOOGLE ANALYSIS -----------------
st.header("Google Data")
if google_df is not None and google_col:
    dfg = google_df.copy()
    dfg['Budget_LKR'] = dfg['Total Budget'] * conv
    dfg['Prev Reach'] = dfg[google_col].shift(1)
    dfg['Prev Budg']  = dfg['Budget_LKR'].shift(1)
    dfg['Inc Reach']  = dfg[google_col] - dfg['Prev Reach']
    dfg['Inc Budg']   = dfg['Budget_LKR'] - dfg['Prev Budg']
    dfg['Efficiency'] = dfg['Inc Reach'] / dfg['Inc Budg']
    idx_opt = dfg['Efficiency'].idxmax()
    ob, or_, oe = dfg.at[idx_opt,'Budget_LKR'], dfg.at[idx_opt,google_col], dfg.at[idx_opt,'Efficiency']
    st.success(f"Google: Opt Budget → {ob:,.0f} LKR")
    st.write(f"Google: Efficiency → {oe:.4f} reach/LKR")
    custom = dfg[dfg['Reach %']>=google_pct].iloc[0] if google_pct else None
    fig2 = make_subplots(specs=[[{{"secondary_y":True}}]])
    fig2.add_trace(go.Scatter(x=dfg['Budget_LKR'], y=dfg[google_col], name='Google Reach', line=dict(color='lightblue')), secondary_y=False)
    fig2.add_trace(go.Scatter(x=dfg['Budget_LKR'], y=dfg['Efficiency'], name='Google Efficiency', line=dict(color='orange', dash='dash')), secondary_y=True)
    fig2.add_trace(go.Scatter(x=[ob], y=[or_], mode='markers', name='Google Opt Budget', marker=dict(color='red', size=12)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=[ob], y=[oe], mode='markers', name='Google Opt Efficiency', marker=dict(color='green', size=12)), secondary_y=True)
    if custom is not None:
        cb, cr = custom['Budget_LKR'], custom[google_col]
        fig2.add_vline(x=cb, line_dash='dot', line_color='purple')
        fig2.add_trace(go.Scatter(x=[cb], y=[cr], mode='markers', name='Google Custom'), secondary_y=False)
    fig2.update_xaxes(title='Budget (LKR)')
    fig2.update_yaxes(title='Reach', secondary_y=False)
    fig2.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------- TV ANALYSIS -----------------
st.header("TV Data")
if tv_file:
    dft = pd.read_csv(tv_file) if tv_file.name.endswith('.csv') else pd.read_excel(tv_file)
    dft.columns = [c.strip() for c in dft.columns]
    # percent to absolute
    for i in range(1,11):
        col = f"{i}+"
        if col in dft:
            dft[col] = pd.to_numeric(dft[col], errors='coerce')/100 * universe
    actual = freq_opts.index(tv_freq)+1
    col = f"{actual}+"
    dft['GRPs']   = pd.to_numeric(dft['GRPs'], errors='coerce')
    dft['Budget'] = dft['GRPs'] * cprp * acd / 30
    dft['Prev Reach'] = dft[col].shift(1)
    dft['Prev Budg']  = dft['Budget'].shift(1)
    dft['Inc Reach']  = dft[col] - dft['Prev Reach']
    dft['Inc Budg']   = dft['Budget'] - dft['Prev Budg']
    dft['Efficiency']= dft['Inc Reach']/dft['Inc Budg']
    idx_opt = dft['Efficiency'].idxmax()
    tb, tr, te = dft.at[idx_opt,'Budget'], dft.at[idx_opt,col], dft.at[idx_opt,'Efficiency']
    st.success(f"TV: Opt Budget → {tb:,.0f} LKR")
    st.write(f"TV: Efficiency → {te:.4f} reach/LKR")
    # plot
    fig3 = make_subplots(specs=[[{{"secondary_y":True}}]])
    fig3.add_trace(go.Scatter(x=dft['Budget'], y=dft[col], name='TV Reach', line=dict(color='cyan')), secondary_y=False)
    fig3.add_trace(go.Scatter(x=dft['Budget'], y=dft['Efficiency'], name='TV Efficiency', line=dict(color='magenta', dash='dash')), secondary_y=True)
    fig3.add_trace(go.Scatter(x=[tb], y=[tr], mode='markers', name='TV Opt Budget', marker=dict(color='red', size=12)), secondary_y=False)
    fig3.add_trace(go.Scatter(x=[tb], y=[te], mode='markers', name='TV Opt Efficiency', marker=dict(color='green', size=12)), secondary_y=True)
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
    rows.append({'Platform':'Google','Opt Budget':ob,'Opt Reach':or_,'Eff':oe})
if tv_file:
    rows.append({'Platform':'TV','Opt Budget':tb,'Opt Reach':tr,'Eff':te})
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
else:
    st.info("Upload data to see summary.")
