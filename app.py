import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Ogilvy Planner", layout="centered", page_icon="🟥")

# —————————————————————————————————————————————
#  SIDEBAR: Meta, Google & TV Settings
# —————————————————————————————————————————————
with st.sidebar:
    # — Meta Settings —
    st.header("Meta Settings")
    meta_file = st.file_uploader("Upload Meta CSV", type=['csv'], key="meta_csv")
    meta_df = None
    meta_freq_val = None
    meta_slider_val = None
    meta_selected_col = None

    if meta_file is not None:
        meta_df = pd.read_csv(meta_file)
        # pick up all "Reach at X+ frequency" columns
        meta_reach_cols = [
            c for c in meta_df.columns
            if c.startswith("Reach at ") and c.endswith("frequency")
        ]
        if meta_reach_cols:
            # slider for X in "Reach at X+"
            meta_freq_val = st.slider(
                "Meta: Select Frequency (Reach at X+)",
                min_value=1, max_value=10, value=1, step=1, key="meta_freq"
            )
            meta_selected_col = f"Reach at {meta_freq_val}+ frequency"
            if meta_selected_col not in meta_df.columns:
                # fallback to first available
                meta_selected_col = meta_reach_cols[0]

            # compute % for custom slider
            temp_max = meta_df[meta_selected_col].max()
            meta_df['Reach %'] = meta_df[meta_selected_col] / temp_max * 100
            min_pct = int(meta_df['Reach %'].min())
            max_pct = int(meta_df['Reach %'].max())
            meta_slider_val = st.slider(
                "Meta: Custom Reach %",
                min_value=min_pct, max_value=max_pct,
                value=min(70, max_pct), step=1, key="meta_slider"
            )

    st.markdown("---")

    # — Google Settings —
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV / Excel", type=['csv','xlsx'], key="google_csv")
    conversion_rate = st.number_input("USD → LKR", value=300.0, step=1.0)
    google_df = None
    google_freq_val = None
    google_slider_val = None
    google_selected_col = None

    if google_file is not None:
        # read file
        if google_file.name.endswith(".xlsx"):
            google_df = pd.read_excel(google_file)
        else:
            google_df = pd.read_csv(google_file)

        # coerce numeric
        google_df = google_df.apply(lambda col:
            pd.to_numeric(col.astype(str).str.replace(",","").strip(), errors='coerce')
        )

        # find all "X+ on-target reach"
        reach_cols = [c for c in google_df.columns if c.strip().endswith("on-target reach")]
        if reach_cols:
            # slider for X
            freqs = sorted(int(c.split('+')[0]) for c in reach_cols)
            google_freq_val = st.slider(
                "Google: Select Frequency (X+ on-target reach)",
                min_value=freqs[0], max_value=freqs[-1],
                value=freqs[0], step=1, key="google_freq"
            )
            google_selected_col = f"{google_freq_val}+ on-target reach"
            if google_selected_col not in google_df.columns:
                google_selected_col = reach_cols[0]

            # compute % for custom slider
            max_r = google_df[google_selected_col].max()
            google_df['Reach %'] = google_df[google_selected_col] / max_r * 100
            min_pct = int(google_df['Reach %'].min())
            max_pct = int(google_df['Reach %'].max())
            google_slider_val = st.slider(
                "Google: Custom Reach %",
                min_value=min_pct, max_value=max_pct,
                value=min(70, max_pct), step=1, key="google_slider"
            )
        else:
            st.error("No columns like ‘X+ on-target reach’ found.")

    st.markdown("---")

    # — TV Settings —
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV CSV / Excel", type=['csv','xlsx'], key="tv_csv")
    cprp = st.text_input("CPRP (LKR)", value="8,000")
    acd  = st.text_input("ACD (sec)", value="17")
    tv_universe = st.text_input("Universe (pop)", value="11,440,000")
    maximum_reach_tv = st.text_input("Max Reach (abs)", value="10,296,000")

    # helper to parse ints
    def to_int(x, default):
        try:
            return int(str(x).replace(",",""))
        except:
            return default

    cprp = to_int(cprp, 8000)
    acd  = to_int(acd, 17)
    tv_universe    = to_int(tv_universe, 11440000)
    maximum_reach_tv = to_int(maximum_reach_tv, 10296000)

    freq_options = [f"{i} +" for i in range(1,11)]
    freq_selected = st.selectbox("TV: Frequency", options=freq_options, index=0)

# —————————————————————————————————————————————
#  META SECTION  
# —————————————————————————————————————————————
st.header("Meta Data")
st.markdown("Upload your **Meta Reach Planner CSV**. It needs columns `Reach at 1+ frequency`…`Reach at 10+ frequency` in absolute numbers.")
if meta_file is not None and meta_selected_col:
    df = meta_df.copy()
    # % reach
    maximum_reach = df[meta_selected_col].max()
    df['Reach %'] = df[meta_selected_col] / maximum_reach * 100
    df['Prev Reach %']  = df['Reach %'].shift(1)
    df['Prev Budget']   = df['Budget'].shift(1)
    # Efficiency formula
    df['Efficiency'] = ( (df['Reach %']/df['Prev Reach %']) /
                         (df['Budget']/df['Prev Budget']) ) * 100
    df['Efficiency'] = df['Efficiency'].replace([np.inf,-np.inf],np.nan).bfill()

    # find elbow (min change in scaled efficiency)
    scaler = MinMaxScaler()
    df['Scaled Eff']   = scaler.fit_transform(df[['Efficiency']])
    df['Scaled Budg']  = scaler.fit_transform(df[['Budget']])
    df['ΔEff']         = np.diff(df['Scaled Eff'], prepend=np.nan)
    idx_opt = df['ΔEff'].idxmin()
    optimal_budget    = df.at[idx_opt, 'Budget']
    optimal_reach     = df.at[idx_opt, meta_selected_col]
    optimal_efficiency= df.at[idx_opt, 'Efficiency']

    # custom slider row
    slider_row = df[df['Reach %'] >= meta_slider_val].iloc[0] if meta_slider_val else None

    st.success(f"**Meta: Optimum Budget: {optimal_budget:,.0f} LKR**")
    st.write(f"Efficiency there: {optimal_efficiency:.2f}%")

    # plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Reach line
    fig.add_trace(go.Scatter(
        x=df['Budget'], y=df[meta_selected_col],
        mode='lines', name=meta_selected_col,
        line=dict(color='skyblue', width=3)
    ), secondary_y=False)
    # Efficiency line
    fig.add_trace(go.Scatter(
        x=df['Budget'], y=df['Efficiency'],
        mode='lines', name='Efficiency',
        line=dict(color='royalblue', dash='dash', width=3)
    ), secondary_y=True)
    # optimum budget marker
    fig.add_trace(go.Scatter(
        x=[optimal_budget], y=[optimal_reach],
        mode='markers+text', name='Optimum Budget',
        marker=dict(size=14, color='orange', line=dict(width=2, color='black')),
        text=[f"<b>{optimal_budget:,.0f} LKR</b>"], textposition="middle right"
    ), secondary_y=False)
    # optimum efficiency marker
    fig.add_trace(go.Scatter(
        x=[optimal_budget], y=[optimal_efficiency],
        mode='markers+text', name='Optimum Efficiency',
        marker=dict(size=14, color='red', line=dict(width=2, color='black')),
        text=[f"<b>{optimal_efficiency:.1f}%</b>"], textposition="bottom left"
    ), secondary_y=True)
    # custom slider marker
    if slider_row is not None:
        fig.add_vline(x=slider_row['Budget'], line_dash="dot", line_color="purple",
                      annotation_text=f"{meta_slider_val}%", annotation_position="top")
        fig.add_trace(go.Scatter(
            x=[slider_row['Budget']], y=[slider_row[meta_selected_col]],
            mode='markers+text', name='Selected Reach %',
            marker=dict(size=12, color='purple', line=dict(width=2, color='black')),
            text=[f"{slider_row['Reach %']:.1f}%"], textposition="top center"
        ), secondary_y=False)

    fig.update_layout(template="plotly_dark", margin=dict(t=60,b=40))
    fig.update_xaxes(title="Budget (LKR)")
    fig.update_yaxes(title=meta_selected_col,   secondary_y=False)
    fig.update_yaxes(title="Efficiency (%)",    secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# —————————————————————————————————————————————
#  GOOGLE SECTION  
# —————————————————————————————————————————————
st.header("Google Data")
st.markdown("Upload **Total Budget** & **X+ on-target reach** columns.")
if google_file is not None and google_df is not None and google_selected_col:
    df1 = google_df.copy()
    # LKR conversion
    df1['Budget_LKR'] = df1['Total Budget'] * conversion_rate
    # % reach
    max_r = df1[google_selected_col].max()
    df1['Reach %'] = df1[google_selected_col] / max_r * 100
    df1['Prev Reach %']    = df1['Reach %'].shift(1)
    df1['Prev Budget_LKR'] = df1['Budget_LKR'].shift(1)
    df1['Efficiency'] = (
        (df1['Reach %'] - df1['Prev Reach %']) /
        (df1['Budget_LKR'] - df1['Prev Budget_LKR'])
    ) * 100
    df1['Efficiency'] = df1['Efficiency'].replace([np.inf,-np.inf],0)

    # slider pick row
    sel_row = df1[df1['Reach %'] >= google_slider_val].iloc[0] if google_slider_val else None

    # elbow detection
    try:
        from kneed import KneeLocator
        kl = KneeLocator(df1['Budget_LKR'], df1[google_selected_col],
                         curve='concave', direction='increasing')
        knee = kl.knee
        i    = (np.abs(df1['Budget_LKR'] - knee)).argmin()
        optimal_budget_g     = df1.at[i,'Budget_LKR']
        optimal_reach_g      = df1.at[i,google_selected_col]
        optimal_efficiency_g = df1.at[i,'Efficiency']
        st.success(f"**Google: Optimum Budget: {optimal_budget_g:,.0f} LKR**")
        st.write(f"Efficiency there: {optimal_efficiency_g:.2f}%")
    except:
        st.error("Install `kneed` for elbow detection.")
        optimal_budget_g = df1['Budget_LKR'].iloc[0]
        optimal_reach_g  = df1[google_selected_col].iloc[0]
        optimal_efficiency_g = df1['Efficiency'].iloc[0]

    # plot
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(
        x=df1['Budget_LKR'], y=df1[google_selected_col],
        mode='lines+markers', name=f"{google_freq_val}+ reach",
        line=dict(color='lightblue', width=3)
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=df1['Budget_LKR'], y=df1['Efficiency'],
        mode='lines+markers', name='Efficiency',
        line=dict(color='orange', dash='dash', width=3)
    ), secondary_y=True)
    # optimum markers
    fig2.add_trace(go.Scatter(
        x=[optimal_budget_g], y=[optimal_reach_g],
        mode='markers+text', name='Optimum Budget',
        marker=dict(size=14, color='red', line=dict(width=2, color='black')),
        text=[f"{optimal_budget_g:,.0f}"], textposition="top right"
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=[optimal_budget_g], y=[optimal_efficiency_g],
        mode='markers+text', name='Optimum Efficiency',
        marker=dict(size=14, color='green', line=dict(width=2, color='black')),
        text=[f"{optimal_efficiency_g:.1f}%"], textposition="bottom left"
    ), secondary_y=True)
    # custom slider line
    if sel_row is not None:
        fig2.add_vline(x=sel_row['Budget_LKR'], line_dash="dot", line_color="purple",
                       annotation_text=f"{google_slider_val}%", annotation_position="top")
        fig2.add_trace(go.Scatter(
            x=[sel_row['Budget_LKR']], y=[sel_row[google_selected_col]],
            mode='markers+text', name='Selected Reach %',
            marker=dict(size=12, color='purple', line=dict(width=2, color='black')),
            text=[f"{sel_row['Reach %']:.1f}%"], textposition="top center"
        ), secondary_y=False)

    fig2.update_layout(template="plotly_dark", margin=dict(t=60,b=40))
    fig2.update_xaxes(title="Budget (LKR)")
    fig2.update_yaxes(title=google_selected_col, secondary_y=False)
    fig2.update_yaxes(title="Efficiency (%)",    secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

# —————————————————————————————————————————————
#  TV SECTION  
# —————————————————————————————————————————————
st.header("TV Data")
st.markdown("Upload columns `1 +` through `10 +` as % (0–100).")
if tv_file is not None:
    # read TV plan
    df3 = pd.read_csv(tv_file) if tv_file.name.endswith(".csv") else pd.read_excel(tv_file)
    # normalize names
    df3.columns = [c.strip() for c in df3.columns]
    # convert %→absolute
    for i in range(1,11):
        col = f"{i} +"
        if col in df3.columns:
            df3[col] = pd.to_numeric(df3[col],errors='coerce')/100 * tv_universe
    # pick actual col
    target = freq_selected.replace(" ","")
    actual_col = next((c for c in df3.columns if c.replace(" ","")==target), None)
    if actual_col is None:
        st.error(f"Couldn’t find column `{freq_selected}`.")
    else:
        # compute budget + reach %
        df3['GRPs'] = pd.to_numeric(df3['GRPs'],errors='coerce')
        df3['Budget'] = (cprp * df3['GRPs'])*acd/30
        df3['Reach %'] = df3[actual_col] / maximum_reach_tv * 100
        df3['Prev Reach %'] = df3['Reach %'].shift(1)
        df3['Prev Budget']   = df3['Budget'].shift(1)
        df3['Efficiency']    = ((df3['Reach %'] - df3['Prev Reach %'])/
                                (df3['Budget'] - df3['Prev Budget']))*100
        df3['Efficiency'] = df3['Efficiency'].replace([np.inf,-np.inf],0)

        # custom slider
        mn, mx = int(df3['Reach %'].min()), int(df3['Reach %'].max())
        tv_slider_val = st.sidebar.slider("TV: Custom Reach %", min_value=mn, max_value=mx,
                                          value=min(70,mx), step=1, key="tv_slider")
        row_sel = df3[df3['Reach %']>=tv_slider_val].iloc[0] if tv_slider_val else None

        # elbow
        try:
            from kneed import KneeLocator
            kl = KneeLocator(df3['Budget'], df3[actual_col], curve='concave', direction='increasing')
            knee = kl.knee
            j = (np.abs(df3['Budget']-knee)).argmin()
            optimal_budget_tv     = df3.at[j,'Budget']
            optimal_reach_tv      = df3.at[j,actual_col]
            optimal_efficiency_tv = df3.at[j,'Efficiency']
            st.success(f"**TV: Optimum Budget: {optimal_budget_tv:,.0f} LKR**")
            st.write(f"Efficiency there: {optimal_efficiency_tv:.2f}%")
        except:
            st.error("Install `kneed` for elbow detection.")
            optimal_budget_tv     = df3['Budget'].iloc[0]
            optimal_reach_tv      = df3[actual_col].iloc[0]
            optimal_efficiency_tv = df3['Efficiency'].iloc[0]

        # plot
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(
            x=df3['Budget'], y=df3[actual_col],
            mode='lines+markers', name=f"Reach {freq_selected}",
            line=dict(color='cyan', width=3)
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=df3['Budget'], y=df3['Efficiency'],
            mode='lines+markers', name='Efficiency',
            line=dict(color='magenta', dash='dash', width=3)
        ), secondary_y=True)
        # optimum markers
        fig3.add_trace(go.Scatter(
            x=[optimal_budget_tv], y=[optimal_reach_tv],
            mode='markers+text', name='Optimum Budget',
            marker=dict(size=14, color='red', line=dict(width=2, color='black')),
            text=[f"{optimal_budget_tv:,.0f}"], textposition="top right"
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=[optimal_budget_tv], y=[optimal_efficiency_tv],
            mode='markers+text', name='Optimum Efficiency',
            marker=dict(size=14, color='green', line=dict(width=2, color='black')),
            text=[f"{optimal_efficiency_tv:.1f}%"], textposition="bottom left"
        ), secondary_y=True)
        # custom slider marker
        if row_sel is not None:
            fig3.add_vline(x=row_sel['Budget'], line_dash="dot", line_color="purple",
                           annotation_text=f"{tv_slider_val}%", annotation_position="top")
            fig3.add_trace(go.Scatter(
                x=[row_sel['Budget']], y=[row_sel[actual_col]],
                mode='markers+text', name='Selected Reach %',
                marker=dict(size=12, color='purple', line=dict(width=2, color='black')),
                text=[f"{row_sel['Reach %']:.1f}%"], textposition="top center"
            ), secondary_y=False)

        fig3.update_layout(template="plotly_dark", margin=dict(t=60,b=40))
        fig3.update_xaxes(title="Budget (LKR)")
        fig3.update_yaxes(title=f"Reach {freq_selected}", secondary_y=False)
        fig3.update_yaxes(title="Efficiency (%)", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

# —————————————————————————————————————————————
#  SUMMARY TABLE  
# —————————————————————————————————————————————
st.header("Platform Comparison Table")

rows = []

# Meta summary
if meta_file and meta_selected_col:
    rows.append({
        "Platform": "Meta",
        "Max Reach": f"{maximum_reach:,.0f}",
        "Opt Reach": f"{optimal_reach:,.0f}",
        "Opt Budget": f"{optimal_budget:,.0f}",
        "Custom %": f"{meta_slider_val}%",
        "Reach@Custom": f"{slider_row[meta_selected_col]:,.0f}" if slider_row is not None else "–",
        "Budget@Custom": f"{slider_row['Budget']:,.0f}"    if slider_row is not None else "–"
    })

# Google summary
if google_file and google_selected_col:
    rows.append({
        "Platform": "Google",
        "Max Reach": f"{max_r:,.0f}",
        "Opt Reach": f"{optimal_reach_g:,.0f}",
        "Opt Budget": f"{optimal_budget_g:,.0f}",
        "Custom %": f"{google_slider_val}%",
        "Reach@Custom": f"{sel_row[google_selected_col]:,.0f}" if sel_row is not None else "–",
        "Budget@Custom": f"{sel_row['Budget_LKR']:,.0f}"      if sel_row is not None else "–"
    })

# TV summary
if tv_file and actual_col:
    rows.append({
        "Platform": "TV",
        "Max Reach": f"{df3[actual_col].max():,.0f}",
        "Opt Reach": f"{optimal_reach_tv:,.0f}",
        "Opt Budget": f"{optimal_budget_tv:,.0f}",
        "Custom %": f"{tv_slider_val}%",
        "Reach@Custom": f"{row_sel[actual_col]:,.0f}" if row_sel is not None else "–",
        "Budget@Custom": f"{row_sel['Budget']:,.0f}"     if row_sel is not None else "–"
    })

if rows:
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, hide_index=True)
else:
    st.info("Upload at least one platform’s data to see the summary.")
