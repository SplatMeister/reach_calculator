import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimum Detection", layout="centered")
st.title("📊 Optimum Budget Detection – Meta & Google Data")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Meta Analysis Settings")
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

    st.header("Google Settings")
    google_file = st.file_uploader("Upload Google CSV", type=['csv'], key="google_csv")
    conversion_rate = st.number_input("USD to LKR Conversion Rate", value=300.0, min_value=0.0, step=1.0)
    google_slider_val = None
    google_df = None
    min_pct_g, max_pct_g = 0, 100
    if google_file is not None:
        # Try different read_csv strategies
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
        # All budgets shown in LKR
        df1['Budget_LKR'] = df1['Total Budget'] * conversion_rate
        maximum_reach_google = df1["1+ on-target reach"].max()
        df1['Reach Percentage'] = (df1['1+ on-target reach'] / maximum_reach_google) * 100

        df1['Previous Reach %'] = df1['Reach Percentage'].shift(1)
        df1['Previous Budget_LKR'] = df1['Budget_LKR'].shift(1)
        df1['Efficiency'] = ((df1['Reach Percentage'] - df1['Previous Reach %']) /
                            (df1['Budget_LKR'] - df1['Previous Budget_LKR'])) * 100
        df1['Efficiency'] = df1['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Find the row for the custom reach percentage
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

        # Only show Efficiency from 2nd row onwards
        eff_mask = df1.index != df1.index.min()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
                x=df1['Budget_LKR'], y=df1['1+ on-target reach'],
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

        # Add custom reach percentage marker & line if selected
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
