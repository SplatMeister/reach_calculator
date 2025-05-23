import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.header("Google Data")
st.write("""
Upload your **Google Reach CSV** (with `"Total Budget"` and `"1+ on-target reach"` columns).
""")

google_file = st.file_uploader("Upload Google CSV", type=['csv'], key="google_csv")
conversion_rate = st.number_input("USD to LKR Conversion Rate", value=300.0, min_value=0.0, step=1.0)
google_slider_val = None

if google_file is not None:
    df1 = pd.read_csv(google_file)
    df1 = df1.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
    df1 = df1.dropna(subset=['Total Budget', '1+ on-target reach'])
    df1['Budget_LKR'] = df1['Total Budget'] * conversion_rate

    maximum_reach_google = df1["1+ on-target reach"].max()
    df1['Reach Percentage'] = (df1['1+ on-target reach'] / maximum_reach_google) * 100

    df1['Previous Reach %'] = df1['Reach Percentage'].shift(1)
    df1['Previous Budget'] = df1['Total Budget'].shift(1)
    df1['Efficiency'] = ((df1['Reach Percentage'] - df1['Previous Reach %']) /
                         (df1['Total Budget'] - df1['Previous Budget'])) * 100
    df1['Efficiency'] = df1['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ----- Elbow/Knee detection -----
    try:
        from kneed import KneeLocator

        x = df1['Total Budget'].values
        y = df1['1+ on-target reach'].values

        kl = KneeLocator(x, y, curve='concave', direction='increasing')
        optimal_budget = kl.knee
        optimal_reach = kl.knee_y

        # Find corresponding efficiency (if exact match not found, use closest)
        eff_idx = (np.abs(df1['Total Budget'] - optimal_budget)).argmin()
        optimal_efficiency = df1.iloc[eff_idx]['Efficiency']
        optimal_budget = df1.iloc[eff_idx]['Total Budget']
        optimal_reach = df1.iloc[eff_idx]['1+ on-target reach']

        st.success(f"**Google: Optimum Budget (Kneedle/Elbow): {optimal_budget:,.2f} USD / {optimal_budget * conversion_rate:,.2f} LKR**")
        st.write(f"Google: Efficiency at this point: {optimal_efficiency:.2f}")

    except ImportError:
        st.error("The 'kneed' library is required for knee/elbow detection. Please run `pip install kneed` in your environment and reload this app.")
        optimal_budget = df1['Total Budget'].iloc[0]
        optimal_reach = df1['1+ on-target reach'].iloc[0]
        optimal_efficiency = df1['Efficiency'].iloc[0]

    # Plotly visualization
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
            x=df1['Total Budget'], y=df1['1+ on-target reach'],
            mode='lines+markers', name='1+ on-target reach',
            line=dict(color='royalblue', width=3)
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
            x=df1['Total Budget'], y=df1['Efficiency'],
            mode='lines+markers', name='Efficiency',
            line=dict(color='orange', width=3, dash='dash')
        ), secondary_y=True)
    # Mark optimum point
    fig.add_trace(go.Scatter(
        x=[optimal_budget], y=[optimal_reach],
        mode='markers+text',
        marker=dict(size=14, color='red', line=dict(width=2, color='black')),
        text=[f"<b>Optimum<br>Budget:<br>{optimal_budget:,.0f}</b>"],
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

    fig.update_layout(
        title="Google: Total Budget vs 1+ on-target reach and Efficiency",
        xaxis=dict(title='Total Budget (USD)'),
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
