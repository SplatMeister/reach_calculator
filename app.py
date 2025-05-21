import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimum Detection", layout="centered")

st.title("📊 Optimum Budget Detection – Meta Reach Data")
st.write("""
Upload your **CSV file** (as exported from Meta Reach Planner).  
Analyze *any* "Reach at X+ frequency" column, and visualize optimum budget and custom reach thresholds.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Detect all Reach at X+ frequency columns (up to 10+)
    reach_cols = [col for col in df.columns if col.startswith('Reach at ') and col.endswith('frequency')]
    required_cols = ['Budget'] + reach_cols
    if len(reach_cols) == 0 or 'Budget' not in df.columns:
        st.error("Your CSV must contain at least one 'Reach at X+ frequency' column and a 'Budget' column.")
    else:
        # --- Sidebar controls ---
        with st.sidebar:
            st.header("Analysis Settings")
            # Select which Reach column to use
            freq_options = sorted(
                [(int(col.split(' ')[2][0]), col) for col in reach_cols], key=lambda x: x[0])
            freq_labels = [f"Reach at {x[0]}+ frequency" for x in freq_options]
            freq_idx = st.slider("Select Frequency (Reach at X+)", min_value=1, max_value=10, value=1, step=1)
            # Find the selected column
            selected_col = f"Reach at {freq_idx}+ frequency"
            # Validate
            if selected_col not in df.columns:
                # Fallback to first available
                selected_col = freq_options[0][1]
            # Slider for minimum reach %
            temp_max_reach = df[selected_col].max()
            min_percentage = int((df[selected_col] / temp_max_reach * 100).min())
            max_percentage = int((df[selected_col] / temp_max_reach * 100).max())
            reach_pct_slider = st.slider(
                "Minimum Reach Percentage to Include",
                min_value=min_percentage,
                max_value=max_percentage,
                value=min(40, max_percentage),
                step=1
            )

        # --- Main calculation using selected frequency ---
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

        st.success(f"**Optimal Budget (Change in Efficiency minimum): {optimal_budget:,.2f}**")
        st.write(f"Efficiency at this point: {optimal_efficiency:.2f}")

        # Find the row closest to the selected reach % threshold
        slider_row = df[df['Reach Percentage'] >= reach_pct_slider].iloc[0] if not df[df['Reach Percentage'] >= reach_pct_slider].empty else None

        # --- Plotly Visualization ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df['Budget'], y=df[selected_col],
                mode='lines', name=selected_col,
                line=dict(color='royalblue', width=3)
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df['Budget'], y=df['Efficiency'],
                mode='lines', name='Efficiency',
                line=dict(color='seagreen', width=3, dash='dash')
            ),
            secondary_y=True,
        )
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

        # SLIDER vertical marker
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

        # ----------- LEGEND & TITLE & MARGIN FIX HERE --------------
        fig.update_layout(
            title={
                'text': f"<b>{selected_col} & Efficiency vs Budget</b><br><span style='font-size:15px; font-weight:normal'>Optimum Point Highlighted</span>",
                'y': 1.13,  # Move title up above plot area
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(title='Budget'),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.30,    # Move legend further above plot area
                xanchor='right',
                x=1,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            ),
            template="plotly_white",
            margin=dict(l=40, r=40, t=220, b=40)  # Increase top margin
        )
        # ------------------------------------------------------------

        fig.update_yaxes(
            title_text=selected_col, color='royalblue', secondary_y=False)
        fig.update_yaxes(
            title_text='Efficiency', color='seagreen', secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        # --- Table with selected row at the end ---
        st.subheader("Data Table (with Selected Reach % row)")
        show_df = df[['Budget', selected_col, 'Reach Percentage', 'Efficiency']].copy()
        show_df = show_df.round({'Reach Percentage': 2, 'Efficiency': 2})
        if slider_row is not None:
            selected_dict = slider_row[['Budget', selected_col, 'Reach Percentage', 'Efficiency']].to_dict()
            selected_dict = {k: [v] for k, v in selected_dict.items()}
            selected_df = pd.DataFrame(selected_dict)
            selected_df.index = ['Selected Reach %']
            st.dataframe(pd.concat([show_df, selected_df], axis=0))
        else:
            st.dataframe(show_df)
else:
    st.info("Please upload a CSV file with 'Budget' and at least one 'Reach at X+ frequency' column to begin.")
