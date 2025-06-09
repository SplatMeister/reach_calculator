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
    conversion_rate = st.number_input("USD to LKR Conversion Rate", 300.0, 0.0, step=1.0)
    google_df = None
    google_freq_val = None
    google_selected_col = None
    google_slider_val = None

    if google_file is not None:
        # Read file
        try:
            if google_file.name.endswith('.xlsx'):
                google_df = pd.read_excel(google_file)
            else:
                google_df = pd.read_csv(google_file)
        except Exception as e:
            st.error(f"Failed to read Google file: {e}")

        if isinstance(google_df, pd.DataFrame):
            # Identify reach columns
            reach_cols = [c for c in google_df.columns if 'on-target reach' in c.lower()]
            if reach_cols:
                # Convert only numeric columns
                cols_to_num = ['Total Budget'] + reach_cols
                for col in cols_to_num:
                    if col in google_df.columns:
                        google_df[col] = pd.to_numeric(
                            google_df[col].astype(str)
                                        .str.replace(',', '')
                                        .str.strip(),
                            errors='coerce'
                        )
                # Slider for frequency
                freqs = sorted(int(c.split('+')[0]) for c in reach_cols)
                google_freq_val = st.slider(
                    "Google: Select Frequency (X+ on-target reach)",
                    freqs[0], freqs[-1], freqs[0], step=1, key="google_freq"
                )
                google_selected_col = f"{google_freq_val}+ on-target reach"
                if google_selected_col not in google_df.columns:
                    google_selected_col = reach_cols[0]

                # Custom reach % slider
                max_r = google_df[google_selected_col].max()
                google_df['Reach %'] = google_df[google_selected_col] / max_r * 100
                min_pct = int(google_df['Reach %'].min())
                max_pct = int(google_df['Reach %'].max())
                google_slider_val = st.slider(
                    "Google: Custom Reach %", min_pct, max_pct, min(70, max_pct), 1, key="google_slider"
                )
            else:
                st.error("No 'on-target reach' columns found in Google data.")

    st.markdown("---")

    # ----- TV Section -----
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV or Excel", type=['csv', 'xlsx'], key="tv_file")
    cprp_str = st.text_input("CPRP (LKR)", "8,000")
    acd_str  = st.text_input("ACD (sec)", "17")
    univ_str = st.text_input("Universe (pop)", "11,440,000")
    max_tv_str = st.text_input("Max Reach (abs)", "10,296,000")

    def to_int(x, default):
        try:
            return int(str(x).replace(',', ''))
        except:
            return default

    cprp = to_int(cprp_str, 8000)
    acd  = to_int(acd_str, 17)
    universe = to_int(univ_str, 11440000)
    max_tv = to_int(max_tv_str, 10296000)

    freq_opts = [f"{i} +" for i in range(1, 11)]
    freq_selected = st.selectbox("TV: Frequency", freq_opts, index=0)

# --------------- META SECTION ------------------
st.header("Meta Data")
if meta_file and meta_selected_col:
    df = meta_df.copy()
    max_r = df[meta_selected_col].max()
    df['Reach %'] = df[meta_selected_col]/max_r*100
    df['Prev Reach %'] = df['Reach %'].shift(1)
    df['Prev Budget']   = df['Budget'].shift(1)
    df['Efficiency']    = ((df['Reach %']/df['Prev Reach %'])/(df['Budget']/df['Prev Budget']))*100
    df['Efficiency'] = df['Efficiency'].replace([np.inf,-np.inf], np.nan).bfill()

    scaler = MinMaxScaler()
    df['Scaled Eff'] = scaler.fit_transform(df[['Efficiency']])
    df['ΔEff']      = df['Scaled Eff'].diff()
    idx_opt = df['ΔEff'].idxmin()
    opt_b = df.at[idx_opt,'Budget']
    opt_e = df.at[idx_opt,'Efficiency']
    opt_r = df.at[idx_opt,meta_selected_col]

    st.success(f"Meta: Optimal Budget → {opt_b:,.0f} LKR")
    st.write(f"Efficiency: {opt_e:.2f}%")

    # custom row
    sel = df[df['Reach %']>=meta_slider_val].iloc[0] if meta_slider_val else None

    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[meta_selected_col], mode='lines', name=meta_selected_col),secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Efficiency'], mode='lines', name='Efficiency', line=dict(dash='dash')),secondary_y=True)
    fig.add_trace(go.Scatter(x=[opt_b], y=[opt_r], mode='markers+text', name='Opt Budget', marker=dict(color='orange',size=12), text=[f"{opt_b:,.0f}"], textposition='top right'),secondary_y=False)
    fig.add_trace(go.Scatter(x=[opt_b], y=[opt_e], mode='markers+text', name='Opt Eff', marker=dict(color='red',size=12), text=[f"{opt_e:.1f}%"], textposition='bottom left'),secondary_y=True)
    if sel is not None:
        fig.add_vline(x=sel['Budget'], line_dash='dot', annotation_text=f"{meta_slider_val}%", annotation_position='top')
        fig.add_trace(go.Scatter(x=[sel['Budget']], y=[sel[meta_selected_col]], mode='markers+text', name='Custom Reach', marker=dict(color='purple',size=10), text=[f"{sel['Reach %']:.1f}%"], textposition='top center'),secondary_y=False)
    fig.update_layout(template='plotly_white', margin=dict(t=50,b=40))
    fig.update_xaxes(title='Budget (LKR)')
    fig.update_yaxes(title=meta_selected_col, secondary_y=False)
    fig.update_yaxes(title='Efficiency (%)', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --------------- GOOGLE SECTION ------------------
st.header("Google Data")
if google_file and isinstance(google_df,pd.DataFrame) and google_selected_col:
    df1 = google_df.copy()
    # Budget LKR
    df1['Budget_LKR'] = df1['Total Budget'] * conversion_rate
    # Reach %
    max_r = df1[google_selected_col].max()
    df1['Reach %'] = df1[google_selected_col]/max_r*100
    df1['Prev Reach %']    = df1['Reach %'].shift(1)
    df1['Prev Budget']     = df1['Budget_LKR'].shift(1)
    df1['Efficiency']      = ((df1['Reach %']-df1['Prev Reach %'])/(df1['Budget_LKR']-df1['Prev Budget']))*100
    df1['Efficiency'] = df1['Efficiency'].replace([np.inf,-np.inf],0)

    sel = df1[df1['Reach %']>=google_slider_val].iloc[0] if google_slider_val else None
    # elbow
    try:
        from kneed import KneeLocator
        kl = KneeLocator(df1['Budget_LKR'], df1[google_selected_col], curve='concave', direction='increasing')
        knee = kl.knee
        i = (np.abs(df1['Budget_LKR']-knee)).argmin()
        ob = df1.at[i,'Budget_LKR']; or_ = df1.at[i,google_selected_col]; oe = df1.at[i,'Efficiency']
        st.success(f"Google: Opt Budget → {ob:,.0f} LKR")
        st.write(f"Efficiency: {oe:.2f}%")
    except:
        st.error("Install kneed for elbow detection.")
        ob = df1['Budget_LKR'].iloc[0]; or_ = df1[google_selected_col].iloc[0]; oe = df1['Efficiency'].iloc[0]

    fig2 = make_subplots(specs=[[{"secondary_y":True}]])
    fig2.add_trace(go.Scatter(x=df1['Budget_LKR'], y=df1[google_selected_col], mode='lines+markers', name=f"{google_freq_val}+ reach"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df1['Budget_LKR'], y=df1['Efficiency'], mode='lines+markers', name='Efficiency', line=dict(dash='dash')), secondary_y=True)
    fig2.add_trace(go.Scatter(x=[ob], y=[or_], mode='markers+text', name='Opt Budget', marker=dict(color='red',size=12), text=[f"{ob:,.0f}"], textposition='top right'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=[ob], y=[oe], mode='markers+text', name='Opt Eff', marker=dict(color='green',size=12), text=[f"{oe:.1f}%"], textposition='bottom left'), secondary_y=True)
    if sel is not None:
        fig2.add_vline(x=sel['Budget_LKR'], line_dash='dot', annotation_text=f"{google_slider_val}%", annotation_position='top')
        fig2.add_trace(go.Scatter(x=[sel['Budget_LKR']], y=[sel[google_selected_col]], mode='markers+text', name='Custom Reach', marker=dict(color='purple',size=10), text=[f"{sel['Reach %']:.1f}%"], textposition='top center'), secondary_y=False)
    fig2.update_layout(template='plotly_white', margin=dict(t=50,b=40))
    fig2.update_xaxes(title='Budget (LKR)')
    fig2.update_yaxes(title=google_selected_col, secondary_y=False)
    fig2.update_yaxes(title='Efficiency (%)', secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

# --------------- TV SECTION ------------------
st.header("TV Data")
if tv_file:
    df3 = pd.read_csv(tv_file) if tv_file.name.endswith('.csv') else pd.read_excel(tv_file)
    df3.columns = [c.strip() for c in df3.columns]
    for i in range(1,11):
        col = f"{i} +"
        if col in df3.columns:
            df3[col] = pd.to_numeric(df3[col],errors='coerce')/100 * universe
    tgt = freq_selected.replace(" ","")
    actual = next((c for c in df3.columns if c.replace(" ","")==tgt), None)
    if not actual:
        st.error(f"Column {freq_selected} not found.")
    else:
        df3['GRPs'] = pd.to_numeric(df3['GRPs'],errors='coerce')
        df3['Budget'] = df3['GRPs']*cprp*acd/30
        df3['Reach %'] = df3[actual]/max_tv*100
        df3['Prev Reach %'] = df3['Reach %'].shift(1)
        df3['Prev Budget']   = df3['Budget'].shift(1)
        df3['Efficiency']    = ((df3['Reach %']-df3['Prev Reach %'])/(df3['Budget']-df3['Prev Budget']))*100
        df3['Efficiency'] = df3['Efficiency'].replace([np.inf,-np.inf],0)
        mn,mx = int(df3['Reach %'].min()),int(df3['Reach %'].max())
        tv_val = st.sidebar.slider("TV: Custom Reach %", mn,mx,min(70,mx),1, key="tv_slider")
        sel_tv = df3[df3['Reach %']>=tv_val].iloc[0] if tv_val else None
        try:
            from kneed import KneeLocator
            kl = KneeLocator(df3['Budget'],df3[actual],curve='concave',direction='increasing')
            kn = kl.knee
            j = (np.abs(df3['Budget']-kn)).argmin()
            ob3 = df3.at[j,'Budget']; or3 = df3.at[j,actual]; oe3 = df3.at[j,'Efficiency']
            st.success(f"TV: Opt Budget → {ob3:,.0f} LKR")
            st.write(f"Efficiency: {oe3:.2f}%")
        except:
            st.error("Install kneed for elbow detection.")
            ob3 = df3['Budget'].iloc[0]; or3 = df3[actual].iloc[0]; oe3 = df3['Efficiency'].iloc[0]
        fig3 = make_subplots(specs=[[{"secondary_y":True}]])
        fig3.add_trace(go.Scatter(x=df3['Budget'],y=df3[actual],mode='lines+markers',name=f'Reach {freq_selected}'),secondary_y=False)
        fig3.add_trace(go.Scatter(x=df3['Budget'],y=df3['Efficiency'],mode='lines+markers',name='Efficiency',line=dict(dash='dash')),secondary_y=True)
        fig3.add_trace(go.Scatter(x=[ob3],y=[or3],mode='markers+text',name='Opt Budget',marker=dict(color='red',size=12),text=[f"{ob3:,.0f}"],textposition='top right'),secondary_y=False)
        fig3.add_trace(go.Scatter(x=[ob3],y=[oe3],mode='markers+text',name='Opt Eff',marker=dict(color='green',size=12),text=[f"{oe3:.1f}%"],textposition='bottom left'),secondary_y=True)
        if sel_tv is not None:
            fig3.add_vline(x=sel_tv['Budget'],line_dash='dot',annotation_text=f"{tv_val}%",annotation_position='top')
            fig3.add_trace(go.Scatter(x=[sel_tv['Budget']],y=[sel_tv[actual]],mode='markers+text',name='Custom Reach',marker=dict(color='purple',size=10),text=[f"{sel_tv['Reach %']:.1f}%"],textposition='top center'),secondary_y=False)
        fig3.update_layout(template='plotly_white',margin=dict(t=50,b=40))
        fig3.update_xaxes(title='Budget (LKR)')
        fig3.update_yaxes(title=f'Reach {freq_selected}',secondary_y=False)
        fig3.update_yaxes(title='Efficiency (%)',secondary_y=True)
        st.plotly_chart(fig3,use_container_width=True)

# ----------- Summary Table Section -----------
st.header("Platform Comparison Table")
rows = []
if meta_file and meta_selected_col:
    rows.append({
        "Platform":"Meta",
        "Opt Budget (LKR)":f"{opt_b:,.0f}",
        "Opt Reach":f"{opt_r:,.0f}",
        "Custom %":f"{meta_slider_val}%"
    })
if google_file and google_selected_col:
    rows.append({
        "Platform":"Google",
        "Opt Budget (LKR)":f"{ob:,.0f}",
        "Opt Reach":f"{or_:,.0f}",
        "Custom %":f"{google_slider_val}%"
    })
if tv_file and actual:
    rows.append({
        "Platform":"TV",
        "Opt Budget (LKR)":f"{ob3:,.0f}",
        "Opt Reach":f"{or3:,.0f}",
        "Custom %":f"{tv_val}%"
    })
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
else:
    st.info("Upload at least one platform’s data to see summary.")
