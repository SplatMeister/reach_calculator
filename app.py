import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimum Detection", layout="centered")

st.title("📊 Optimum Budget Detection – Meta Reach Data")
st.write("""
Upload your **CSV file** (as exported from Meta Reach Planner).  
The app will process the data, calculate Efficiency, and visualize **Reach & Efficiency vs Budget**
with the optimum point (where efficiency change drops the most) highlighted.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ['Budget', 'Reach at 1+ frequency']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Your CSV must contain the columns: {required_cols}")
    else:
        # Step 1: Find maximum reach
        maximum_reach = df["Reach at 1+ frequency"].max()
        # Step 2: Create percentage of maximum
        df['Reach Percentage'] = (df['Reach at 1+ frequency'] / maximum_reach) * 100
        # Step 3: Shift for previous values
        df['Previous Reach %'] = df['Reach Percentage'].shift(1)
        df['Previous Budget'] = df['Budget'].shift(1)
        # Step 4: Calculate Efficiency
        df['Efficiency'] = ((df['Reach Percentage'] / df['Previous Reach %']) /
                            (df['Budget'] / df['Previous Budget'])) * 100
        # Clean up Efficiency NaN/inf
        df['Efficiency'] = df['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')
        
        # MinMax Scaling
        scaler = MinMaxScaler()
        df['Scaled Budget'] = scaler.fit_transform(df[['Budget']])
        df['Scaled Efficiency'] = scaler.fit_transform(df[['Efficiency']])

        # Calculate change between points
        df['Efficiency Change'] = np.diff(df['Scaled Efficiency'], prepend=np.nan)

        # Find optimal budget (where efficiency change is minimum)
        optimal_budget_index = df['Efficiency Change'].idxmin()
        optimal_budget = df.loc[optimal_budget_index, 'Budget']
        optimal_efficiency = df.loc[optimal_budget_index, 'Efficiency']
        optimal_reach = df.loc[optimal_budget_index, 'Reach at 1+ frequency']

        st.success(f"**Optimal Budget (Change in Efficiency minimum): {optimal_budget:,.2f}**")
        st.write(f"Efficiency at this point: {optimal_efficiency:.2f}")

        # --- Plotly Visualization ---
        fig = go.Figure()

        # Reach at 1+ Frequency (Primary Y axis)
        fig.add_trace(go.Scatter(
            x=df['Budget'], y=df['Reach at 1+ frequency'],
            mode='lines', name='Reach at 1+ Frequency', line=dict(color='royalblue', width=3),
            yaxis="y1"
        ))

        # Efficiency (Secondary Y axis)
        fig.add_trace(go.Scatter(
            x=df['Budget'], y=df['Efficiency'],
            mode='lines', name='Efficiency', line=dict(color='seagreen', width=3, dash='dash'),
            yaxis="y2"
        ))

        # Optimum Budget Marker
        fig.add_trace(go.Scatter(
            x=[optimal_budget], y=[optimal_reach],
            mode='markers+text',
            marker=dict(size=14, color='orange', line=dict(width=2, color='black')),
            text=[f"Optimum<br>Budget:<br>{optimal_budget:,.0f}"],
            textposition="top right",
            name='Optimum Point (Reach)',
            yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=[optimal_budget], y=[optimal_efficiency],
            mode='markers+text',
            marker=dict(size=14, color='red', line=dict(width=2, color='black')),
            text=[f"Efficiency:<br>{optimal_efficiency:.2f}"],
            textposition="bottom right",
            name='Optimum Point (Efficiency)',
            yaxis="y2"
        ))

        fig.update_layout(
            title="Reach at 1+ Frequency & Efficiency vs Budget<br><span style='font-size:16px'>Optimum Point Highlighted</span>",
            xaxis=dict(title='Budget'),
            yaxis=dict(title='Reach at 1+ Frequency', titlefont=dict(color='royalblue'), tickfont=dict(color='royalblue')),
            yaxis2=dict(title='Efficiency', titlefont=dict(color='seagreen'), tickfont=dict(color='seagreen'),
                        overlaying='y', side='right', showgrid=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a CSV file with 'Budget' and 'Reach at 1+ frequency' columns to begin.")
