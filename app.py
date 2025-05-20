import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimum Detection", layout="wide")

st.title("📊 Optimum Budget Detection – Meta Reach Data (Interactive)")
st.write("""
Upload your **CSV file** (as exported from Meta Reach Planner).  
Interactively select Reach column and minimum Reach % to analyze the optimum point and visualize it live.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Find all columns matching "Reach at X+ frequency"
    reach_cols = [col for col in df.columns if col.startswith("Reach at ") and col.endswith("+ frequency")]
    if not reach_cols or 'Budget' not in df.columns:
        st.error("CSV must contain 'Budget' and at least one 'Reach at X+ frequency' column!")
    else:
        # Sidebar - column selector and reach % filter
        st.sidebar.header("Options")
        reach_col = st.sidebar.select_slider(
            "Select Reach Level",
            options=reach_cols,
            value="Reach at 1+ frequency"
        )
        st.sidebar.write("You are analyzing:", reach_col)

        # Step 1: Find maximum reach for selected column
        maximum_reach = df[reach_col].max()
        df['Reach Percentage'] = (df[reach_col] / maximum_reach) * 100

        # Reach % threshold slider
        min_reach_pct = st.sidebar.slider(
            "Minimum Reach Percentage to Include",
            min_value=0, max_value=100, value=0, step=1
        )

        # Filter by reach %
        df_filtered = df[df['Reach Percentage'] >= min_reach_pct].copy()
        if df_filtered.empty:
            st.warning("No data points above the selected reach percentage threshold.")
        else:
            # Step 2: Shift for previous values
            df_filtered['Previous Reach %'] = df_filtered['Reach Percentage'].shift(1)
            df_filtered['Previous Budget'] = df_filtered['Budget'].shift(1)
            # Step 3: Calculate Efficiency
            df_filtered['Efficiency'] = (
                (df_filtered['Reach Percentage'] / df_filtered['Previous Reach %']) /
                (df_filtered['Budget'] / df_filtered['Previous Budget'])
            ) * 100
            df_filtered['Efficiency'] = df_filtered['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')

            # MinMax Scaling
            scaler = MinMaxScaler()
            df_filtered['Scaled Budget'] = scaler.fit_transform(df_filtered[['Budget']])
            df_filtered['Scaled Efficiency'] = scaler.fit_transform(df_filtered[['Efficiency']])

            # Calculate change between points
            df_filtered['Efficiency Change'] = np.diff(df_filtered['Scaled Efficiency'], prepend=np.nan)

            # Find optimal budget (where efficiency change is minimum)
            optimal_budget_index = df_filtered['Efficiency Change'].idxmin()
            optimal_budget = df_filtered.loc[optimal_budget_index, 'Budget']
            optimal_efficiency = df_filtered.loc[optimal_budget_index, 'Efficiency']
            optimal_reach = df_filtered.loc[optimal_budget_index, reach_col]

            # --- Output summary table ---
            max_reach_row = df_filtered[df_filtered[reach_col] == maximum_reach].iloc[0]
            summary_table = pd.DataFrame({
                " ": ["Maximum", "Optimum"],
                "Budget": [max_reach_row['Budget'], optimal_budget],
                f"{reach_col}": [maximum_reach, optimal_reach],
                "Reach %": [100, df_filtered.loc[optimal_budget_index, 'Reach Percentage']],
                "Efficiency": [max_reach_row['Efficiency'], optimal_efficiency]
            }).set_index(" ")

            st.subheader("Summary Table")
            st.table(summary_table.style.format({
                "Budget": "{:,.0f}",
                f"{reach_col}": "{:,.0f}",
                "Reach %": "{:.2f}",
                "Efficiency": "{:.2f}"
            }))

            # --- Plotly make_subplots Visualization ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Primary Y axis: Reach
            fig.add_trace(
                go.Scatter(
                    x=df_filtered['Budget'], y=df_filtered[reach_col],
                    mode='lines', name=reach_col,
                    line=dict(color='royalblue', width=3)
                ),
                secondary_y=False,
            )

            # Secondary Y axis: Efficiency
            fig.add_trace(
                go.Scatter(
                    x=df_filtered['Budget'], y=df_filtered['Efficiency'],
                    mode='lines', name='Efficiency',
                    line=dict(color='seagreen', width=3, dash='dash')
                ),
                secondary_y=True,
            )

            # Optimum Budget Marker: Reach (annotation to the right)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_budget], y=[optimal_reach],
                    mode='markers+text',
                    marker=dict(size=14, color='orange', line=dict(width=2, color='black')),
                    text=[f"<b>Optimum<br>Budget:<br>{optimal_budget:,.0f}</b>"],
                    textposition="middle right",
                    name='Optimum Point (Reach)'
                ),
                secondary_y=False,
            )

            # Optimum Budget Marker: Efficiency (annotation to the left)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_budget], y=[optimal_efficiency],
                    mode='markers+text',
                    marker=dict(size=14, color='red', line=dict(width=2, color='black')),
                    text=[f"<b>Efficiency:<br>{optimal_efficiency:.2f}</b>"],
                    textposition="bottom left",
                    name='Optimum Point (Efficiency)'
                ),
                secondary_y=True,
            )

            fig.update_layout(
                title={
                    'text': f"<b>{reach_col} & Efficiency vs Budget</b><br><span style='font-size:15px; font-weight:normal'>Optimum Point Highlighted</span>",
                    'y':0.90,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis=dict(title='Budget'),
                legend=dict(
                    orientation='h',
                    yanchor='top',
                    y=1.20,
                    xanchor='right',
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=14)
                ),
                template="plotly_white",
                margin=dict(l=40, r=40, t=150, b=40)
            )
            fig.update_yaxes(
                title_text=reach_col, color='royalblue', secondary_y=False)
            fig.update_yaxes(
                title_text='Efficiency', color='seagreen', secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a CSV file with 'Budget' and 'Reach at 1+ frequency' columns to begin.")
