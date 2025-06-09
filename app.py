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
    meta_reach_cols = []
    if meta_file is not None:
        meta_df = pd.read_csv(meta_file)
        meta_reach_cols = [col for col in meta_df.columns if col.startswith('Reach at ') and col.endswith('frequency')]
        if meta_reach_cols:
            # slider for frequency
            meta_freq_val = st.slider(
                "Meta: Select Frequency (Reach at X+)",
                min_value=1, max_value=10, value=1, step=1, key="meta_freq"
            )
            meta_selected_col = f"Reach at {meta_freq_val}+ frequency"
            if meta_selected_col not in meta_df.columns:
                meta_selected_col = meta_reach_cols[0]
            # compute percentage range
            temp_max = meta_df[meta_selected_col].max()
            min_pct = int((meta_df[meta_selected_col] / temp_max * 100).min())
            max_pct = int((meta_df[meta_selected_col] / temp_max * 100).max())
            meta_slider_val = st.slider(
                "Meta: Custom Reach Percentage", min_value=min_pct,
                max_value=max_pct, value=min(70, max_pct), step=1, key="meta_slider"
            )
        else:
            meta_selected_col = None
    else:
        meta_selected_col = None

    st.markdown("---")

    # ----- Google Section (modified) -----
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV or Excel", type=['csv', 'xlsx'], key="google_csv")
    conversion_rate = st.number_input(
        "USD to LKR Conversion Rate", value=300.0, min_value=0.0, step=1.0
    )
    google_df = None
    google_freq_val = None
    google_selected_col = None
    google_slider_val = None

    if google_file is not None:
        # try reading
        try:
            if google_file.name.endswith('.xlsx'):
                google_df = pd.read_excel(google_file)
            else:
                google_df = pd.read_csv(google_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
        # process if read
        if google_df is not None:
            # ensure numeric
            google_df = google_df.apply(
                lambda col: pd.to_numeric(
                    col.astype(str).str.replace(',', '').str.strip(), errors='coerce'
                )
            )
            # detect reach columns
            reach_cols = [c for c in google_df.columns if c.strip().endswith('on-target reach')]
            if reach_cols:
                # parse frequencies
                freqs = sorted(int(c.split('+')[0]) for c in reach_cols)
                google_freq_val = st.slider(
                    "Google: Select Frequency (X+ on-target reach)",
                    min_value=freqs[0], max_value=freqs[-1], value=freqs[0], step=1,
                    key="google_freq"
                )
                google_selected_col = f"{google_freq_val}+ on-target reach"
                if google_selected_col not in google_df.columns:
                    google_selected_col = reach_cols[0]
                # percentage range
                max_reach = google_df[google_selected_col].max()
                google_df['Reach Percentage'] = google_df[google_selected_col] / max_reach * 100
                min_pct = int(google_df['Reach Percentage'].min())
                max_pct = int(google_df['Reach Percentage'].max())
                google_slider_val = st.slider(
                    "Google: Custom Reach Percentage", min_value=min_pct,
                    max_value=max_pct, value=min(70, max_pct), step=1,
                    key="google_slider"
                )
            else:
                st.error("No columns matching 'X+ on-target reach'. Check your sheet.")

    st.markdown("---")

    # ----- TV Section -----
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV Excel/CSV", type=['xlsx', 'csv'], key="tv_file")

    def parse_int(val, default):
        try:
            return int(str(val).replace(",", "").replace(" ", ""))
        except:
            st.warning(f"Invalid input: '{val}'. Using default {default}.")
            return default

    cprp_str = st.text_input("CPRP (Cost Per Rating Point (LKR))", value="8,000")
    acd_str = st.text_input("ACD (Average Commercial Duration in Seconds)", value="17")
    tv_universe_str = st.text_input("Universe (Population)", value="11,440,000")
    maximum_reach_tv_str = st.text_input("Maximum Reach (Absolute)", value="10,296,000")

    cprp = parse_int(cprp_str, 8000)
    acd = parse_int(acd_str, 17)
    tv_universe = parse_int(tv_universe_str, 11440000)
    maximum_reach_tv = parse_int(maximum_reach_tv_str, 10296000)

    freq_display = [f"{i} +" for i in range(1, 11)]
    freq_selected = st.selectbox("TV: Select Frequency", options=freq_display, index=0)

# --------------- META SECTION ------------------
st.header("Meta Data")
if meta_file and meta_selected_col:
    df = meta_df.copy()
    max_reach = df[meta_selected_col].max()
    df['Reach Percentage'] = df[meta_selected_col] / max_reach * 100
    df['Previous Reach %'] = df['Reach Percentage'].shift(1)
    df['Previous Budget'] = df['Budget'].shift(1)
    df['Efficiency'] = (
        (df['Reach Percentage'] / df['Previous Reach %']) /
        (df['Budget'] / df['Previous Budget'])
    ) * 100
    df['Efficiency'] = df['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')

    scaler = MinMaxScaler()
    df['Scaled Efficiency'] = scaler.fit_transform(df[['Efficiency']])
    df['Efficiency Change'] = np.diff(df['Scaled Efficiency'], prepend=np.nan)

    opt_idx = df['Efficiency Change'].idxmin()
    opt_budget = df.loc[opt_idx, 'Budget']
    opt_eff = df.loc[opt_idx, 'Efficiency']
    opt_reach = df.loc[opt_idx, meta_selected_col]

    st.success(f"**Meta: Optimal Budget (elbow): {opt_budget:,.2f} LKR**")
    st.write(f"Efficiency at this point: {opt_eff:.2f}")

    sel_row = df[df['Reach Percentage'] >= meta_slider_val].iloc[0] if meta_slider_val else None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Budget'], y=df[meta_selected_col], mode='lines', name=meta_selected_col), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Budget'], y=df['Efficiency'], mode='lines', name='Efficiency', line=dict(dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[opt_budget], y=[opt_reach], mode='markers+text', marker=dict(size=14, color='orange'), text=[f"Optimum Budget:\n{opt_budget:,.0f}"], textposition='middle right'), secondary_y=False)
    fig.add_trace(go.Scatter(x=[opt_budget], y=[opt_eff], mode='markers+text', marker=dict(size=14, color='red'), text=[f"Eff:{opt_eff:.1f}%"], textposition='bottom left'), secondary_y=True)
    if sel_row is not None:
        fig.add_vline(x=sel_row['Budget'], line_dash='dot', annotation_text=f"{meta_slider_val}%", annotation_position='top')
        fig.add_trace(go.Scatter(x=[sel_row['Budget']], y=[sel_row[meta_selected_col]], mode='markers+text', marker=dict(size=12, color='purple'), text=[f"{sel_row['Reach Percentage']:.1f}%"], textposition='top center'), secondary_y=False)
    fig.update_xaxes(title='Budget (LKR)')
    fig.update_yaxes(title=meta_selected_col, secondary_y=False)
    fig.update_yaxes(title='Efficiency', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --------------- GOOGLE SECTION ------------------
st.header("Google Data")
if google_file and isinstance(google_df, pd.DataFrame) and google_selected_col:
    df1 = google_df.copy()
    if 'Total Budget' not in df1.columns or google_selected_col not in df1.columns:
        st.error("Required columns missing.")
    else:
        df1['Budget_LKR'] = df1['Total Budget'] * conversion_rate
        max_reach_g = df1[google_selected_col].max()
        df1['Reach Percentage'] = df1[google_selected_col] / max_reach_g * 100
        df1['Previous Reach %'] = df1['Reach Percentage'].shift(1)
        df1['Previous Budget_LKR'] = df1['Budget_LKR'].shift(1)
        df1['Efficiency'] = ((df1['Reach Percentage'] - df1['Previous Reach %']) / (df1['Budget_LKR'] - df1['Previous Budget_LKR'])) * 100
        df1['Efficiency'] = df1['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # custom slider row
        sel_g = None
        if google_slider_val is not None:
            cond = df1['Reach Percentage'] >= google_slider_val
            sel_g = df1[cond].iloc[0] if cond.any() else None

        # elbow detect
        try:
            from kneed import KneeLocator
            x = df1['Budget_LKR'].values
            y = df1[google_selected_col].values
            kl = KneeLocator(x, y, curve='concave', direction='increasing')
            knee = kl.knee
            idx = (np.abs(df1['Budget_LKR'] - knee)).argmin()
            opt_b = df1.iloc[idx]['Budget_LKR']
            opt_r = df1.iloc[idx][google_selected_col]
            opt_e = df1.iloc[idx]['Efficiency']
            st.success(f"**Google: Optimum Budget ({google_freq_val}+): {opt_b:,.2f} LKR**")
            st.write(f"Efficiency at this point: {opt_e:.2f}")
        except ImportError:
            st.error("Install kneed: pip install kneed")
            opt_b = df1['Budget_LKR'].iloc[0]
            opt_r = df1[google_selected_col].iloc[0]
            opt_e = df1['Efficiency'].iloc[0]

        mask = df1.index != df1.index.min()
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=df1.loc[mask,'Budget_LKR'], y=df1.loc[mask,google_selected_col], mode='lines+markers', name=f"{google_freq_val}+ reach"), secondary_y=False)
        fig1.add_trace(go.Scatter(x=df1.loc[mask,'Budget_LKR'], y=df1.loc[mask,'Efficiency'], mode='lines+markers', name='Efficiency', line=dict(dash='dash')), secondary_y=True)
        fig1.add_trace(go.Scatter(x=[opt_b], y=[opt_r], mode='markers+text', marker=dict(size=12,color='red'), text=[f"Opt Budget:\n{opt_b:,.0f}"], textposition='top right'), secondary_y=False)
        fig1.add_trace(go.Scatter(x=[opt_b], y=[opt_e], mode='markers+text', marker=dict(size=12,color='green'), text=[f"Eff:{opt_e:.1f}%"], textposition='bottom left'), secondary_y=True)
        if sel_g is not None:
            fig1.add_vline(x=sel_g['Budget_LKR'], line_dash='dot', annotation_text=f"{google_slider_val}%", annotation_position='top')
            fig1.add_trace(go.Scatter(x=[sel_g['Budget_LKR']], y=[sel_g[google_selected_col]], mode='markers+text', marker=dict(size=10,color='purple'), text=[f"{sel_g['Reach Percentage']:.1f}%"], textposition='top center'), secondary_y=False)
        fig1.update_xaxes(title='Budget (LKR)')
        fig1.update_yaxes(title=f"{google_freq_val}+ on-target reach", secondary_y=False)
        fig1.update_yaxes(title='Efficiency (%)', secondary_y=True)
        st.plotly_chart(fig1, use_container_width=True)

# --------------- TV SECTION ------------------
st.header("TV Data")
if tv_file:
    if tv_file.name.endswith('.csv'):
        df3 = pd.read_csv(tv_file)
    else:
        df3 = pd.read_excel(tv_file)
    df3.columns = [c.strip() for c in df3.columns]
    freq_cols = [f"{i} +" for i in range(1,11)]
    for col in freq_cols:
        if col in df3.columns:
            df3[col] = pd.to_numeric(df3[col], errors='coerce')/100 * tv_universe
    clean = freq_selected.replace(" ","")
    actual = next((c for c in df3.columns if c.replace(" ","") == clean), None)
    if not actual:
        st.error(f"Column {freq_selected} not found.")
    else:
        df3['GRPs'] = pd.to_numeric(df3['GRPs'], errors='coerce')
        df3['Budget'] = (cprp * df3['GRPs'] * acd/30).round(2)
        df3['Reach Percentage'] = df3[actual]/maximum_reach_tv * 100
        df3['Previous Reach %'] = df3['Reach Percentage'].shift(1)
        df3['Previous Budget'] = df3['Budget'].shift(1)
        df3['Efficiency'] = ((df3['Reach Percentage'] - df3['Previous Reach %'])/(df3['Budget'] - df3['Previous Budget'])).replace([np.inf,-np.inf],np.nan).fillna(0)*100
        # slider
        min_tv = int(df3['Reach Percentage'].min())
        max_tv = int(df3['Reach Percentage'].max())
        tv_slider = st.sidebar.slider("TV: Custom Reach %", min_value=min_tv, max_value=max_tv, value=min(70,max_tv), step=1, key="tv_slider")
        row_tv = df3[df3['Reach Percentage']>=tv_slider].iloc[0] if any(df3['Reach Percentage']>=tv_slider) else None
        # elbow
        try:
            from kneed import KneeLocator
            x3,y3 = df3['Budget'].values, df3[actual].values
            kl3 = KneeLocator(x3,y3,curve='concave',direction='increasing')
            kb = kl3.knee
            idx3 = (np.abs(df3['Budget']-kb)).argmin()
            opt_b3 = df3.iloc[idx3]['Budget']
            opt_r3 = df3.iloc[idx3][actual]
            opt_e3 = df3.iloc[idx3]['Efficiency']
            st.success(f"**TV: Optimum Budget (Elbow): {opt_b3:,.2f} LKR**")
            st.write(f"Efficiency: {opt_e3:.2f}")
        except ImportError:
            st.error("Install kneed for elbow detection.")
        m3 = df3.index!=df3.index.min()
        fig3 = make_subplots(specs=[[{"secondary_y":True}]])
        fig3.add_trace(go.Scatter(x=df3.loc[m3,'Budget'],y=df3.loc[m3,actual],mode='lines+markers',name=f'Reach {freq_selected}'), secondary_y=False)
        fig3.add_trace(go.Scatter(x=df3.loc[m3,'Budget'],y=df3.loc[m3,'Efficiency'],mode='lines+markers',name='Efficiency',line=dict(dash='dash')), secondary_y=True)
        fig3.add_trace(go.Scatter(x=[opt_b3],y=[opt_r3],mode='markers+text',marker=dict(size=12,color='red'),text=[f"Opt Budget:\n{opt_b3:,.0f}"],textposition='top right'), secondary_y=False)
        fig3.add_trace(go.Scatter(x=[opt_b3],y=[opt_e3],mode='markers+text',marker=dict(size=12,color='green'),text=[f"Eff:{opt_e3:.1f}%"],textposition='bottom left'), secondary_y=True)
        if row_tv is not None:
            fig3.add_vline(x=row_tv['Budget'],line_dash='dot',annotation_text=f"{tv_slider}%",annotation_position='top')
            fig3.add_trace(go.Scatter(x=[row_tv['Budget']],y=[row_tv[actual]],mode='markers+text',marker=dict(size=10,color='purple'),text=[f"{row_tv['Reach Percentage']:.1f}%"],textposition='top center'), secondary_y=False)
        fig3.update_xaxes(title='Budget (LKR)')
        fig3.update_yaxes(title=f'Reach {freq_selected}', secondary_y=False)
        fig3.update_yaxes(title='Efficiency (%)', secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

# ----------- Summary Table Section -----------
st.header("Platform Comparison Table")
summary = []
# Meta summary
if meta_file and meta_selected_col:
    row_opt = df.loc[opt_idx]
    row_cust = df[df['Reach Percentage']>=meta_slider_val].iloc[0] if meta_slider_val else None
    summary.append({
        "Platform":"Meta",
        "Optimum Budget (LKR)":f"{row_opt['Budget']:,.0f}",
        "Optimum Reach":f"{row_opt[meta_selected_col]:,.0f}",
        "Custom %":f"{meta_slider_val}%",
        "Budget @ Custom":f"{row_cust['Budget']:,.0f}" if row_cust is not None else ""
    })
# Google summary
if google_file and isinstance(google_df,pd.DataFrame) and google_selected_col:
    row_opt_g = df1.iloc[idx]
    row_cust_g = df1[df1['Reach Percentage']>=google_slider_val].iloc[0] if google_slider_val else None
    summary.append({
        "Platform":"Google",
        "Optimum Budget (LKR)":f"{opt_b:,.0f}",
        "Optimum Reach":f"{opt_r:,.0f}",
        "Custom %":f"{google_slider_val}%",
        "Budget @ Custom":f"{row_cust_g['Budget_LKR']:,.0f}" if row_cust_g is not None else ""
    })
# TV summary
if tv_file and actual:
    row_opt_tv = df3.iloc[idx3]
    row_cust_tv = df3[df3['Reach Percentage']>=tv_slider].iloc[0] if tv_slider else None
    summary.append({
        "Platform":"TV",
        "Optimum Budget (LKR)":f"{row_opt_tv['Budget']:,.0f}",
        "Optimum Reach":f"{row_opt_tv[actual]:,.2f}",
        "Custom %":f"{tv_slider}%",
        "Budget @ Custom":f"{row_cust_tv['Budget']:,.0f}" if row_cust_tv is not None else ""
    })
if summary:
    st.dataframe(pd.DataFrame(summary), hide_index=True)
else:
    st.info("Upload data for at least one platform to see summary.")
