import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Meta Reach Planner Analyzer", layout="centered")
st.title("📊 Meta Reach Planner Analysis")
st.write("Upload your Meta Reach Planner CSV or Excel file to analyze optimum and maximum reach insights.")

# File uploader
uploaded_file = st.file_uploader("Upload Reach Planner CSV or Excel File", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        # Load file based on extension — no skiprows needed
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            df_raw = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            st.error("❌ Unsupported file format.")
            st.stop()

        # Use correct column names
        df = df_raw[['Budget', 'Reach at 1+ frequency']].copy()
        df = df.rename(columns={'Reach at 1+ frequency': 'Reach'})
        df = df.sort_values('Budget').reset_index(drop=True)

        # Metrics
        df['Previous Reach'] = df['Reach'].shift(1)
        df['Incremental Reach'] = df['Reach'] - df['Previous Reach']
        df['Previous Budget'] = df['Budget'].shift(1)
        df['Incremental Spend'] = df['Budget'] - df['Previous Budget']
        df['Cost per 1000 Incremental Reach'] = (df['Incremental Spend'] / df['Incremental Reach']) * 1000

        # Max Reach
        max_row = df.loc[df['Reach'].idxmax()]
        max_reach = max_row['Reach']
        max_budget = max_row['Budget']

        # Optimum Reach = tail where curve flattens
        optimum_row = df.tail(3).iloc[0]
        optimum_reach = optimum_row['Reach']
        optimum_budget = optimum_row['Budget']

        # Sidebar slider
        st.sidebar.header("🎯 Custom Reach Targets")
        custom_percent = st.sidebar.slider("Choose custom % of Max Reach", 10, 100, 70)
        target = max_reach * (custom_percent / 100)

        closest_row = df.iloc[(df['Reach'] - target).abs().argsort()[:1]]
        custom_budget = closest_row['Budget'].values[0]
        custom_reach = closest_row['Reach'].values[0]

        # Summary
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

        fig.add_trace(go.Scatter(
            x=df['Budget'],
            y=df['Reach'],
            mode='lines+markers',
            name='Reach Curve',
            line=dict(color='royalblue')
        ))

        # Markers
        fig.add_vline(x=max_budget, line=dict(color='red', dash='dash'),
                      annotation_text=f'Max Reach\nLKR {max_budget:,.0f}', annotation_position="top right")
        fig.add_vline(x=optimum_budget, line=dict(color='green', dash='dash'),
                      annotation_text=f'Optimum Reach\nLKR {optimum_budget:,.0f}', annotation_position="top left")
        fig.add_vline(x=custom_budget, line=dict(color='purple', dash='dash'),
                      annotation_text=f'{custom_percent}% Reach\nLKR {custom_budget:,.0f}', annotation_position="bottom left")

        fig.add_trace(go.Scatter(
            x=[max_budget, optimum_budget, custom_budget],
            y=[max_reach, optimum_reach, custom_reach],
            mode='markers',
            marker=dict(color=['red', 'green', 'purple'], size=10),
            name='Key Points'
        ))

        fig.update_layout(
            xaxis_title="Budget (LKR)",
            yaxis_title="Reach",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
