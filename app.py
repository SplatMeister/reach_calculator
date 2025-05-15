import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Streamlit page config
st.set_page_config(page_title="Meta Reach Planner Analyzer", layout="centered")

st.title("📊 Meta Reach Planner Analysis")
st.write("Upload your Meta Reach Planner CSV or Excel file to analyze optimum and maximum reach insights.")

# File uploader
uploaded_file = st.file_uploader("Upload Reach Planner CSV or Excel File", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        # Detect file type and load accordingly
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, skiprows=15)
        elif uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, skiprows=15, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            df_raw = pd.read_excel(uploaded_file, skiprows=15, engine='xlrd')
        else:
            st.error("❌ Unsupported file format. Please upload a .csv, .xlsx, or .xls file.")
            st.stop()

        # Extract relevant columns and clean
        df = df_raw[['Budget', 'Reach at 1+ Frequency']].copy()
        df = df.sort_values('Budget').reset_index(drop=True)

        # Compute incremental metrics
        df['Previous Reach'] = df['Reach at 1+ Frequency'].shift(1)
        df['Incremental Reach'] = df['Reach at 1+ Frequency'] - df['Previous Reach']
        df['Previous Budget'] = df['Budget'].shift(1)
        df['Incremental Spend'] = df['Budget'] - df['Previous Budget']
        df['Cost per 1000 Incremental Reach'] = (df['Incremental Spend'] / df['Incremental Reach']) * 1000

        # Max Reach
        max_reach_row = df.loc[df['Reach at 1+ Frequency'].idxmax()]
        max_reach = max_reach_row['Reach at 1+ Frequency']
        max_budget = max_reach_row['Budget']

        # Optimum Reach (based on curve flattening)
        optimum_row = df.tail(3).iloc[0]
        optimum_reach = optimum_row['Reach at 1+ Frequency']
        optimum_budget = optimum_row['Budget']

        # Custom reach thresholds
        st.sidebar.header("🎯 Custom Reach Targets")
        custom_percent = st.sidebar.slider("Choose custom % of Max Reach", 10, 100, 70)
        custom_target = max_reach * (custom_percent / 100)

        # Locate closest match
        closest_row = df.iloc[(df['Reach at 1+ Frequency'] - custom_target).abs().argsort()[:1]]
        custom_budget = closest_row['Budget'].values[0]
        custom_reach = closest_row['Reach at 1+ Frequency'].values[0]

        # Display summary
        st.subheader("📋 Summary Table")
        summary = pd.DataFrame({
            'Metric': ['Maximum Reach', 'Optimum Reach (Flattening)', f'{custom_percent}% Reach'],
            'Reach': [max_reach, optimum_reach, custom_reach],
            'Budget (LKR)': [max_budget, optimum_budget, custom_budget]
        })
        st.dataframe(summary, use_container_width=True)

        # Plotly chart
        st.subheader("📈 Reach Curve")
        fig = go.Figure()

        # Line plot
        fig.add_trace(go.Scatter(
            x=df['Budget'],
            y=df['Reach at 1+ Frequency'],
            mode='lines+markers',
            name='Reach Curve',
            line=dict(color='royalblue')
        ))

        # Add vertical markers
        fig.add_vline(x=max_budget, line=dict(color='red', dash='dash'),
                      annotation_text=f'Max Reach\nLKR {max_budget:,.0f}', annotation_position="top right")
        fig.add_vline(x=optimum_budget, line=dict(color='green', dash='dash'),
                      annotation_text=f'Optimum Reach\nLKR {optimum_budget:,.0f}', annotation_position="top left")
        fig.add_vline(x=custom_budget, line=dict(color='purple', dash='dash'),
                      annotation_text=f'{custom_percent}% Reach\nLKR {custom_budget:,.0f}', annotation_position="bottom left")

        # Add dots at key points
        fig.add_trace(go.Scatter(
            x=[max_budget, optimum_budget, custom_budget],
            y=[max_reach, optimum_reach, custom_reach],
            mode='markers',
            marker=dict(color=['red', 'green', 'purple'], size=10),
            name='Key Points'
        ))

        # Layout
        fig.update_layout(
            xaxis_title="Budget (LKR)",
            yaxis_title="Reach at 1+ Frequency",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
