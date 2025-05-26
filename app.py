import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimum Detection", layout="centered")
st.title("Optimum Budget Detection – Meta, Google & TV Data")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # Meta Section
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
            freq_options = sorted([(int(col.split(' ')[2][0]), col) for col in meta_reach_cols], key=lambda x: x[0])
            meta_freq_val = st.slider(
                "Meta: Select Frequency (Reach at X+)",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="meta_freq"
            )
            meta_selected_col = f"Reach at {meta_freq_val}+ frequency"
            if meta_selected_col not in meta_df.columns:
                meta_selected_col = freq_options[0][1]
            temp_max_reach = meta_df[meta_selected_col].max()
            min_pct = int((meta_df[meta_selected_col] / temp_max_reach * 100).min())
            max_pct = int((meta_df[meta_selected_col] / temp_max_reach * 100).max())
            meta_slider_val = st.slider(
                "Meta: Custom Reach Percentage",
                min_value=min_pct,
                max_value=max_pct,
                value=min(40, max_pct),
                step=1,
                key="meta_slider"
            )
        else:
            meta_selected_col = None
    else:
        meta_selected_col = None

    st.markdown("---")

    # Google Section
    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV", type=['csv'], key="google_csv")
    conversion_rate = st.number_input("USD to LKR Conversion Rate", value=300.0, min_value=0.0, step=1.0)
    google_slider_val = None
    google_df = None
    min_pct_g, max_pct_g = 0, 100
    if google_file is not None:
        google_df = None
        read_error = None
        for try_kwargs in [
            {"sep": ",", "skip_blank_lines": True},
            {"sep": None, "engine": "python", "skip_blank_lines": True},
            {"sep": ",", "skip_blank_lines": True, "skiprows": 1},
            {"sep": None, "engine": "python", "skip_blank_lines": True, "skiprows": 1}
        ]:
            try:
                google_df = pd.read_csv(google_file, **try_kwargs)
                if not google_df.empty and len(google_df.columns) > 1:
                    break
            except Exception as e:
                read_error = e
                continue
        if google_df is None or google_df.empty or len(google_df.columns) < 2:
            st.error(f"Could not read your Google CSV file. Please check the format. Details: {read_error}")
            google_df = None
        else:
            google_df = google_df.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
            google_df = google_df.dropna(subset=['Total Budget', '1+ on-target reach'])
            google_df['Budget_LKR'] = google_df['Total Budget'] * conversion_rate
            maximum_reach_google = google_df["1+ on-target reach"].max()
            google_df['Reach Percentage'] = (google_df['1+ on-target reach'] / maximum_reach_google) * 100
            min_pct_g = int(google_df['Reach Percentage'].min())
            max_pct_g = int(google_df['Reach Percentage'].max())
            google_slider_val = st.slider(
                "Google: Custom Reach Percentage",
                min_value=min_pct_g,
                max_value=max_pct_g,
                value=min(40, max_pct_g),
                step=1,
                key="google_slider"
            )

    st.markdown("---")

    # TV Section (comma separated input)
    st.header("TV Settings")
    tv_file = st.file_uploader("Upload TV Excel/CSV", type=['xlsx', 'csv'], key="tv_file")

    def parse_int(val, default):
        try:
            return int(str(val).replace(",", "").replace(" ", ""))
        except Exception:
            st.warning(f"Invalid input: '{val}'. Using default value {default}.")
            return default

    cprp_str = st.text_input("TV: CPRP (Cost Per Rating Point)", value="8,000")
    acd_str = st.text_input("TV: ACD (Ad Duration in Seconds)", value="17")
    tv_universe_str = st.text_input("TV: Universe (Population)", value="11,440,000")
    maximum_reach_tv_str = st.text_input("TV: Maximum Reach (Absolute)", value="10,296,000")

    cprp = parse_int(cprp_str, 8000)
    acd = parse_int(acd_str, 17)
    tv_universe = parse_int(tv_universe_str, 11440000)
    maximum_reach_tv = parse_int(maximum_reach_tv_str, 10296000)

    freq_display_options = [f"{i} +" for i in range(1, 11)]
    freq_selected = st.selectbox("TV: Select Frequency", options=freq_display_options, index=0)

# --------------- META SECTION ------------------
st.header("Meta Data")
st.write("""
Upload your **Meta Reach Planner CSV**.<br>
Analyze *any* "Reach at X+ frequency" column, and visualize optimum budget and custom reach thresholds.
""", unsafe_allow_html=True)

if meta_file is not None and meta_selected_col is not None:
    df = meta_df.copy()
    maximum_reach = df[meta_selected_col].max()
    df['Reach Percentage'] = (df[meta_selected_col] / maximum_reach) * 100
    df['Previous Reach %'] = df['Reach Percentage'].shift(1)
    df['Previous Budget'] = df['Budget'].shift(1)
    df['Efficiency'] = ((df['Reach Percentage'] / df['Previous Reach %']) /
                        (df['Budget'] / df['Previous Budget'])) * 100
    df['Efficiency'] = df['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')

    scaler = MinMaxScaler()
    df['Scaled Budget'] = scaler.fit_transform(df[['Budget']])
    df['Scaled Efficiency'] = scaler.fit_transform(df[['Efficiency']])
    df['Efficiency Change'] = np.diff(df['Scaled Efficiency'], prepend=np.nan)

    optimal_budget_index = df['Efficiency Change'].idxmin()
    optimal_budget = df.loc[optimal_budget_index, 'Budget']
    optimal_efficiency = df.loc[optimal_budget_index, 'Efficiency']
    optimal_reach = df.loc[optimal_budget_index, meta_selected_col]

    st.success(f"**Meta: Optimal Budget (Change in Efficiency minimum/elbow): {optimal_budget:,.2f} LKR**")
    st.write(f"Meta: Efficiency at this point: {optimal_efficiency:.2f}")

    slider_row = df[df['Reach Percentage'] >= meta_slider_val].iloc[0] if not df[df['Reach Percentage'] >= meta_slider_val].empty else None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
            x=df['Budget'], y=df[meta_selected_col],
            mode='lines', name=meta_selected_col,
            line=dict(color='royalblue', width=3)
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
            x=df['Budget'], y=df['Efficiency'],
            mode='lines', name='Efficiency',
            line=dict(color='seagreen', width=3, dash='dash')
        ), secondary_y=True)
    fig.add_trace(go.Scatter(
            x=[optimal_budget], y=[optimal_reach],
            mode='markers+text',
            marker=dict(size=14, color='orange', line=dict(width=2, color='black')),
            text=[f"<b>Optimum<br>Budget:<br>{optimal_budget:,.0f}</b>"],
            textposition="middle right",
            name='Optimum Point (Reach)'
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
            x=[optimal_budget], y=[optimal_efficiency],
            mode='markers+text',
            marker=dict(size=14, color='red', line=dict(width=2, color='black')),
            text=[f"<b>Efficiency:<br>{optimal_efficiency:.2f}</b>"],
            textposition="bottom left",
            name='Optimum Point (Efficiency)'
        ), secondary_y=True)

    if slider_row is not None:
        fig.add_vline(
            x=slider_row['Budget'],
            line_dash="dot",
            line_color="purple",
            annotation_text=f"{meta_slider_val}%",
            annotation_position="top",
            annotation_font_size=14,
            annotation_font_color="purple"
        )
        fig.add_trace(
            go.Scatter(
                x=[slider_row['Budget']],
                y=[slider_row[meta_selected_col]],
                mode='markers+text',
                marker=dict(size=12, color='purple', line=dict(width=2, color='black')),
                text=[f"{slider_row['Reach Percentage']:.1f}%"],
                textposition="top center",
                name='Selected Reach %'
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title="",
        xaxis=dict(title='Budget (LKR)'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.07,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=100, b=40)
    )
    fig.update_yaxes(title_text=meta_selected_col, color='royalblue', secondary_y=False)
    fig.update_yaxes(title_text='Efficiency', color='seagreen', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --------------- GOOGLE SECTION ------------------
st.header("Google Data")
st.write("""
Upload your **Google Reach CSV** (with `"Total Budget"` and `"1+ on-target reach"` columns).
""")

if google_file is not None and google_df is not None:
    df1 = google_df.copy()
    if df1.empty or 'Total Budget' not in df1.columns or '1+ on-target reach' not in df1.columns:
        st.error("Google CSV file is missing required columns or is empty. Please check the file and try again.")
    else:
        df1['Budget_LKR'] = df1['Total Budget'] * conversion_rate
        maximum_reach_google = df1["1+ on-target reach"].max()
        df1['Reach Percentage'] = (df1['1+ on-target reach'] / maximum_reach_google) * 100

        df1['Previous Reach %'] = df1['Reach Percentage'].shift(1)
        df1['Previous Budget_LKR'] = df1['Budget_LKR'].shift(1)
        df1['Efficiency'] = ((df1['Reach Percentage'] - df1['Previous Reach %']) /
                            (df1['Budget_LKR'] - df1['Previous Budget_LKR'])) * 100
        df1['Efficiency'] = df1['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)

        slider_row_g = None
        if google_slider_val is not None:
            slider_row_g = df1[df1['Reach Percentage'] >= google_slider_val].iloc[0] if not df1[df1['Reach Percentage'] >= google_slider_val].empty else None

        try:
            from kneed import KneeLocator

            x = df1['Budget_LKR'].values
            y = df1['1+ on-target reach'].values

            kl = KneeLocator(x, y, curve='concave', direction='increasing')
            optimal_budget = kl.knee
            optimal_reach = kl.knee_y

            eff_idx = (np.abs(df1['Budget_LKR'] - optimal_budget)).argmin()
            optimal_efficiency = df1.iloc[eff_idx]['Efficiency']
            optimal_budget = df1.iloc[eff_idx]['Budget_LKR']
            optimal_reach = df1.iloc[eff_idx]['1+ on-target reach']

            st.success(f"**Google: Optimum Budget (Kneedle/Elbow): {optimal_budget:,.2f} LKR**")
            st.write(f"Google: Efficiency at this point: {optimal_efficiency:.2f}")

        except ImportError:
            st.error("The 'kneed' library is required for knee/elbow detection. Please run `pip install kneed` in your environment and reload this app.")
            optimal_budget = df1['Budget_LKR'].iloc[0]
            optimal_reach = df1['1+ on-target reach'].iloc[0]
            optimal_efficiency = df1['Efficiency'].iloc[0]

        eff_mask = df1.index != df1.index.min()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
                x=df1.loc[eff_mask, 'Budget_LKR'], y=df1.loc[eff_mask, '1+ on-target reach'],
                mode='lines+markers', name='1+ on-target reach',
                line=dict(color='royalblue', width=3)
            ), secondary_y=False)
        fig.add_trace(go.Scatter(
                x=df1.loc[eff_mask, 'Budget_LKR'], y=df1.loc[eff_mask, 'Efficiency'],
                mode='lines+markers', name='Efficiency',
                line=dict(color='orange', width=3, dash='dash')
            ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=[optimal_budget], y=[optimal_reach],
            mode='markers+text',
            marker=dict(size=14, color='red', line=dict(width=2, color='black')),
            text=[f"<b>Optimum<br>Budget:<br>{optimal_budget:,.0f} LKR</b>"],
            textposition="top right",
            name='Optimum Point (Reach)'
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=[optimal_budget], y=[optimal_efficiency],
            mode='markers+text',
            marker=dict(size=14, color='green', line=dict(width=2, color='black')),
            text=[f"<b>Efficiency:<br>{optimal_efficiency:.2f}</b>"],
            textposition="bottom left",
            name='Optimum Point (Efficiency)'
        ), secondary_y=True)

        if slider_row_g is not None:
            fig.add_vline(
                x=slider_row_g['Budget_LKR'],
                line_dash="dot",
                line_color="purple",
                annotation_text=f"{google_slider_val}%",
                annotation_position="top",
                annotation_font_size=14,
                annotation_font_color="purple"
            )
            fig.add_trace(
                go.Scatter(
                    x=[slider_row_g['Budget_LKR']],
                    y=[slider_row_g['1+ on-target reach']],
                    mode='markers+text',
                    marker=dict(size=12, color='purple', line=dict(width=2, color='black')),
                    text=[f"{slider_row_g['Reach Percentage']:.1f}%"],
                    textposition="top center",
                    name='Selected Reach %'
                ),
                secondary_y=False,
            )

        fig.update_layout(
            title="",
            xaxis=dict(title='Budget (LKR)'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.07,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            ),
            template="plotly_white",
            margin=dict(l=40, r=40, t=100, b=40)
        )
        fig.update_yaxes(title_text="1+ on-target reach", color='royalblue', secondary_y=False)
        fig.update_yaxes(title_text='Efficiency', color='orange', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# --------------- TV SECTION (with comma separated input) ------------------
st.header("TV Data")
st.write("""
Upload your **TV Plan Excel/CSV** (`tv.xlsx` or `.csv` with columns like 'GRPs', '1 +', '2 +', ..., '10 +').<br>
Columns for frequency must be in percentage (%). All calculations and chart will use your input TV universe and max reach.
""", unsafe_allow_html=True)

if tv_file is not None:
    if tv_file.name.endswith('.csv'):
        df3 = pd.read_csv(tv_file)
    else:
        df3 = pd.read_excel(tv_file)

    df3.columns = [col.strip().replace("  ", " ") for col in df3.columns]
    frequency_cols = [f"{i} +" for i in range(1, 11)]
    for col in frequency_cols:
        if col in df3.columns:
            df3[col] = (pd.to_numeric(df3[col], errors='coerce') / 100) * tv_universe

    clean_freq_selected = freq_selected.replace(" ", "")
    actual_col = None
    for col in df3.columns:
        if col.replace(" ", "") == clean_freq_selected:
            actual_col = col
            break

    if actual_col is None:
        st.error(f"Could not find a frequency column matching '{freq_selected}' in your Excel/CSV file. Please check your sheet and column names.")
    else:
        df3[actual_col] = pd.to_numeric(df3[actual_col], errors='coerce')
        df3['GRPs'] = pd.to_numeric(df3['GRPs'], errors='coerce')
        df3['CPRP'] = cprp
        df3['ACD'] = acd
        df3['Budget'] = ((cprp * df3['GRPs']) * acd / 30).round(2)
        df3['Reach Percentage'] = (df3[actual_col] / maximum_reach_tv) * 100
        df3['Previous Reach %'] = df3['Reach Percentage'].shift(1)
        df3['Previous Budget'] = df3['Budget'].shift(1)
        df3['Efficiency'] = ((df3['Reach Percentage'] - df3['Previous Reach %']) /
                             (df3['Budget'] - df3['Previous Budget'])).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        min_tv_pct = int(df3['Reach Percentage'].min())
        max_tv_pct = int(df3['Reach Percentage'].max())
        tv_slider_val = st.sidebar.slider(
            "TV: Custom Reach Percentage",
            min_value=min_tv_pct,
            max_value=max_tv_pct,
            value=min(40, max_tv_pct),
            step=1,
            key="tv_slider"
        )

        tv_slider_row = df3[df3['Reach Percentage'] >= tv_slider_val].iloc[0] if not df3[df3['Reach Percentage'] >= tv_slider_val].empty else None

        plot_mask = df3.index != df3.index.min()

        try:
            from kneed import KneeLocator
            x_tv = df3['Budget'].values
            y_tv = df3[actual_col].values

            kl_tv = KneeLocator(x_tv, y_tv, curve='concave', direction='increasing')
            optimal_budget_tv = kl_tv.knee
            optimal_reach_tv = kl_tv.knee_y

            eff_idx_tv = (np.abs(df3['Budget'] - optimal_budget_tv)).argmin()
            optimal_efficiency_tv = df3.iloc[eff_idx_tv]['Efficiency']
            optimal_budget_tv = df3.iloc[eff_idx_tv]['Budget']
            optimal_reach_tv = df3.iloc[eff_idx_tv][actual_col]
        except ImportError:
            st.error("The 'kneed' library is required for elbow point detection. Please install with `pip install kneed`.")
            optimal_budget_tv = df3['Budget'].iloc[0]
            optimal_reach_tv = df3[actual_col].iloc[0]
            optimal_efficiency_tv = df3['Efficiency'].iloc[0]

        st.success(f"**TV: Optimum Budget (Kneedle/Elbow): {optimal_budget_tv:,.2f} LKR**")
        st.write(f"TV: Efficiency at this point: {optimal_efficiency_tv:.4f}")

        fig_tv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_tv.add_trace(go.Scatter(
                x=df3.loc[plot_mask, 'Budget'], y=df3.loc[plot_mask, actual_col],
                mode='lines+markers', name=f'Reach {freq_selected}',
                line=dict(color='blue', width=3)
            ), secondary_y=False)
        fig_tv.add_trace(go.Scatter(
                x=df3.loc[plot_mask, 'Budget'], y=df3.loc[plot_mask, 'Efficiency'],
                mode='lines+markers', name='Efficiency',
                line=dict(color='orange', width=3, dash='dash')
            ), secondary_y=True)
        fig_tv.add_trace(go.Scatter(
            x=[optimal_budget_tv], y=[optimal_reach_tv],
            mode='markers+text',
            marker=dict(size=14, color='red', line=dict(width=2, color='black')),
            text=[f"<b>Optimum<br>Budget:<br>{optimal_budget_tv:,.0f} LKR</b>"],
            textposition="top right",
            name='Optimum Point (Reach)'
        ), secondary_y=False)
        fig_tv.add_trace(go.Scatter(
            x=[optimal_budget_tv], y=[optimal_efficiency_tv],
            mode='markers+text',
            marker=dict(size=14, color='green', line=dict(width=2, color='black')),
            text=[f"<b>Efficiency:<br>{optimal_efficiency_tv:.2f}</b>"],
            textposition="bottom left",
            name='Optimum Point (Efficiency)'
        ), secondary_y=True)

        if tv_slider_row is not None:
            fig_tv.add_vline(
                x=tv_slider_row['Budget'],
                line_dash="dot",
                line_color="purple",
                annotation_text=f"{tv_slider_val}%",
                annotation_position="top",
                annotation_font_size=14,
                annotation_font_color="purple"
            )
            fig_tv.add_trace(
                go.Scatter(
                    x=[tv_slider_row['Budget']],
                    y=[tv_slider_row[actual_col]],
                    mode='markers+text',
                    marker=dict(size=12, color='purple', line=dict(width=2, color='black')),
                    text=[f"{tv_slider_row['Reach Percentage']:.1f}%"],
                    textposition="top center",
                    name='Selected Reach %'
                ),
                secondary_y=False,
            )

        fig_tv.update_layout(
            title="",
            xaxis=dict(title='Budget (LKR)'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.07,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            ),
            template="plotly_white",
            margin=dict(l=40, r=40, t=100, b=40)
        )
        fig_tv.update_yaxes(title_text=f'Reach {freq_selected}', color='blue', secondary_y=False)
        fig_tv.update_yaxes(title_text='Efficiency', color='orange', secondary_y=True)
        st.plotly_chart(fig_tv, use_container_width=True)

# ----------- Summary Table Section -----------
st.header("Platform Comparison Table")

summary_rows = []

# ----- Meta summary -----
if meta_file is not None and meta_selected_col is not None:
    meta_max_reach = meta_df[meta_selected_col].max()
    meta_opt_row = df.loc[optimal_budget_index]
    meta_opt_reach = meta_opt_row[meta_selected_col]
    meta_opt_budget = meta_opt_row['Budget']
    meta_custom_row = df[df['Reach Percentage'] >= meta_slider_val].iloc[0] if not df[df['Reach Percentage'] >= meta_slider_val].empty else None
    if meta_custom_row is not None:
        meta_custom_reach = meta_custom_row[meta_selected_col]
        meta_custom_budget = meta_custom_row['Budget']
    else:
        meta_custom_reach = np.nan
        meta_custom_budget = np.nan

    summary_rows.append({
        "Platform": "Meta",
        "Maximum Reach": f"{meta_max_reach:,.0f}",
        "Optimum Reach": f"{meta_opt_reach:,.0f}",
        "Optimum Budget (LKR)": f"{meta_opt_budget:,.0f}",
        "Custom % Reach": f"{meta_slider_val}%",
        "Reach @ Custom %": f"{meta_custom_reach:,.0f}",
        "Budget @ Custom % (LKR)": f"{meta_custom_budget:,.0f}"
    })

# ----- Google summary -----
if google_file is not None and google_df is not None:
    google_max_reach = google_df["1+ on-target reach"].max()
    google_opt_reach = optimal_reach
    google_opt_budget = optimal_budget
    google_custom_row = df1[df1['Reach Percentage'] >= google_slider_val].iloc[0] if not df1[df1['Reach Percentage'] >= google_slider_val].empty else None
    if google_custom_row is not None:
        google_custom_reach = google_custom_row['1+ on-target reach']
        google_custom_budget = google_custom_row['Budget_LKR']
    else:
        google_custom_reach = np.nan
        google_custom_budget = np.nan

    summary_rows.append({
        "Platform": "Google",
        "Maximum Reach": f"{google_max_reach:,.0f}",
        "Optimum Reach": f"{google_opt_reach:,.0f}",
        "Optimum Budget (LKR)": f"{google_opt_budget:,.0f}",
        "Custom % Reach": f"{google_slider_val}%",
        "Reach @ Custom %": f"{google_custom_reach:,.0f}",
        "Budget @ Custom % (LKR)": f"{google_custom_budget:,.0f}"
    })

# ----- TV summary -----
if tv_file is not None and 'actual_col' in locals() and actual_col is not None:
    tv_max_reach = df3[actual_col].max()
    tv_opt_reach = optimal_reach_tv
    tv_opt_budget = optimal_budget_tv
    tv_custom_row = df3[df3['Reach Percentage'] >= tv_slider_val].iloc[0] if not df3[df3['Reach Percentage'] >= tv_slider_val].empty else None
    if tv_custom_row is not None:
        tv_custom_reach = tv_custom_row[actual_col]
        tv_custom_budget = tv_custom_row['Budget']
    else:
        tv_custom_reach = np.nan
        tv_custom_budget = np.nan

    summary_rows.append({
        "Platform": "TV",
        "Maximum Reach": f"{tv_max_reach:,.2f}",
        "Optimum Reach": f"{tv_opt_reach:,.2f}",
        "Optimum Budget (LKR)": f"{tv_opt_budget:,.0f}",
        "Custom % Reach": f"{tv_slider_val}%",
        "Reach @ Custom %": f"{tv_custom_reach:,.2f}",
        "Budget @ Custom % (LKR)": f"{tv_custom_budget:,.0f}"
    })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True)
else:
    st.info("Upload data for at least one platform to see the summary table.")
