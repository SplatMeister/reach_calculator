import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimum Detection", layout="centered")

st.title("📊 Optimum Budget Detection – Meta & Google Data")

#######################
# --- META SECTION ---#
#######################
st.header("Meta Data")
st.write("""
Upload your **Meta Reach Planner CSV**.  
Analyze *any* "Reach at X+ frequency" column, and visualize optimum budget and custom reach thresholds.
""")

meta_file = st.file_uploader("Upload your Meta CSV file", type=['csv'], key='meta')
if meta_file is not None:
    df = pd.read_csv(meta_file)
    reach_cols = [col for col in df.columns if col.startswith('Reach at ') and col.endswith('frequency')]
    required_cols = ['Budget'] + reach_cols
    if len(reach_cols) == 0 or 'Budget' not in df.columns:
        st.error("Your CSV must contain at least one 'Reach at X+ frequency' column and a 'Budget' column.")
    else:
        with st.sidebar:
            st.header("Meta Analysis Settings")
            freq_options = sorted([(int(col.split(' ')[2][0]), col) for col in reach_cols], key=lambda x: x[0])
            freq_idx = st.slider("Meta: Select Frequency (Reach at X+)", min_value=1, max_value=10, value=1, step=1)
            selected_col = f"Reach at {freq_idx}+ frequency"
            if selected_col not in df.columns:
                selected_col = freq_options[0][1]
            temp_max_reach = df[selected_col].max()
            min_percentage = int((df[selected_col] / temp_max_reach * 100).min())
            max_percentage = int((df[selected_col] / temp_max_reach * 100).max())
            reach_pct_slider = st.slider(
                "Meta: Custom Reach Percentage",
                min_value=min_percentage,
                max_value=max_percentage,
                value=min(40, max_percentage),
                step=1
            )

        # ---- Calculations ----
        maximum_reach = df[selected_col].max()
        df['Reach Percentage'] = (df[selected_col] / maximum_reach) * 100
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
        optimal_reach = df.loc[optimal_budget_index, selected_col]

        st.success(f"**Meta: Optimal Budget (Change in Efficiency minimum): {optimal_budget:,.2f} LKR**")
        st.write(f"Meta: Efficiency at this point: {optimal_efficiency:.2f}")

        slider_row = df[df['Reach Percentage'] >= reach_pct_slider].iloc[0] if not df[df['Reach Percentage'] >= reach_pct_slider].empty else None

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
                x=df['Budget'], y=df[selected_col],
                mode='lines', name=selected_col,
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
                annotation_text=f"{reach_pct_slider}%",
                annotation_position="top",
                annotation_font_size=14,
                annotation_font_color="purple"
            )
            fig.add_trace(
                go.Scatter(
                    x=[slider_row['Budget']],
                    y=[slider_row[selected_col]],
                    mode='markers+text',
                    marker=dict(size=12, color='purple', line=dict(width=2, color='black')),
                    text=[f"{slider_row['Reach Percentage']:.1f}%"],
                    textposition="top center",
                    name='Selected Reach %'
                ),
                secondary_y=False,
            )

        fig.update_layout(
            title="",  # No chart title at all
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
        fig.update_yaxes(title_text=selected_col, color='royalblue', secondary_y=False)
        fig.update_yaxes(title_text='Efficiency', color='seagreen', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

########################
# --- GOOGLE SECTION ---#
########################
st.header("Google Data")
st.write("""
Upload your **Google Reach CSV** (with `"Total Budget"` and `"1+ on-target reach"` columns).  
Enter your USD to LKR conversion rate.
""")

google_file = st.file_uploader("Upload your Google CSV file", type=['csv'], key='google')
if google_file is not None:
    with st.sidebar:
        st.header("Google Settings")
        conversion_rate = st.number_input("Google: USD to LKR Conversion Rate", value=300.0, min_value=0.0, step=1.0)

    df1 = pd.read_csv(google_file)

    # Apply your cleaning/conversion logic
    df1 = df1.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
    df1 = df1.dropna(subset=['Total Budget', '1+ on-target reach'])
    df1['Budget_LKR'] = df1['Total Budget'] * conversion_rate

    # Your steps for calculation
    maximum_reach_google = df1["1+ on-target reach"].max()
    df1['Reach Percentage'] = (df1['1+ on-target reach'] / maximum_reach_google) * 100
    df1['Previous Reach %'] = df1['Reach Percentage'].shift(1)
    df1['Previous Budget_LKR'] = df1['Budget_LKR'].shift(1)

    # Compute Efficiency as you requested (same as Meta)
    df1['Efficiency'] = ((df1['Reach Percentage'] / df1['Previous Reach %']) /
                         (df1['Budget_LKR'] / df1['Previous Budget_LKR'])) * 100
    df1['Efficiency'] = df1['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')

    # Scaling and optimum as in Meta
    scaler_g = MinMaxScaler()
    df1['Scaled Budget_LKR'] = scaler_g.fit_transform(df1[['Budget_LKR']])
    df1['Scaled Efficiency'] = scaler_g.fit_transform(df1[['Efficiency']])
    df1['Efficiency Change'] = np.diff(df1['Scaled Efficiency'], prepend=np.nan)

    optimal_budget_index_g = df1['Efficiency Change'].idxmin()
    optimal_budget_g = df1.loc[optimal_budget_index_g, 'Budget_LKR']
    optimal_efficiency_g = df1.loc[optimal_budget_index_g, 'Efficiency']
    optimal_reach_g = df1.loc[optimal_budget_index_g, '1+ on-target reach']

    min_percentage_g = int(df1['Reach Percentage'].min())
    max_percentage_g = int(df1['Reach Percentage'].max())
    reach_pct_slider_g = st.slider(
        "Google: Custom Reach Percentage",
        min_value=min_percentage_g,
        max_value=max_percentage_g,
        value=min(40, max_percentage_g),
        step=1,
        key="google_slider"
    )

    slider_row_g = df1[df1['Reach Percentage'] >= reach_pct_slider_g].iloc[0] if not df1[df1['Reach Percentage'] >= reach_pct_slider_g].empty else None

    st.success(f"**Google: Optimal Budget (Change in Efficiency minimum): {optimal_budget_g:,.2f} LKR**")
    st.write(f"Google: Efficiency at this point: {optimal_efficiency_g:.2f}")

    fig_g = make_subplots(specs=[[{"secondary_y": True}]])
    fig_g.add_trace(go.Scatter(
            x=df1['Budget_LKR'], y=df1['1+ on-target reach'],
            mode='lines', name='1+ on-target reach',
            line=dict(color='royalblue', width=3)
        ), secondary_y=False)
    fig_g.add_trace(go.Scatter(
            x=df1['Budget_LKR'], y=df1['Efficiency'],
            mode='lines', name='Efficiency',
            line=dict(color='seagreen', width=3, dash='dash')
        ), secondary_y=True)
    fig_g.add_trace(go.Scatter(
            x=[optimal_budget_g], y=[optimal_reach_g],
            mode='markers+text',
            marker=dict(size=14, color='orange', line=dict(width=2, color='black')),
            text=[f"<b>Optimum<br>Budget:<br>{optimal_budget_g:,.0f}</b>"],
            textposition="middle right",
            name='Optimum Point (Reach)'
        ), secondary_y=False)
    fig_g.add_trace(go.Scatter(
            x=[optimal_budget_g], y=[optimal_efficiency_g],
            mode='markers+text',
            marker=dict(size=14, color='red', line=dict(width=2, color='black')),
            text=[f"<b>Efficiency:<br>{optimal_efficiency_g:.2f}</b>"],
            textposition="bottom left",
            name='Optimum Point (Efficiency)'
        ), secondary_y=True)

    if slider_row_g is not None:
        fig_g.add_vline(
            x=slider_row_g['Budget_LKR'],
            line_dash="dot",
            line_color="purple",
            annotation_text=f"{reach_pct_slider_g}%",
            annotation_position="top",
            annotation_font_size=14,
            annotation_font_color="purple"
        )
        fig_g.add_trace(
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

    fig_g.update_layout(
        title="",  # No chart title at all
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
    fig_g.update_yaxes(title_text="1+ on-target reach", color='royalblue', secondary_y=False)
    fig_g.update_yaxes(title_text='Efficiency', color='seagreen', secondary_y=True)
    st.plotly_chart(fig_g, use_container_width=True)
