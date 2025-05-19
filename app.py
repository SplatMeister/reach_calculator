import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Streamlit page config
st.set_page_config(page_title="Meta Reach Planner Analyzer", layout="centered")

# Custom styles for professional dark mode appearance
st.markdown("""
    <style>
    .stApp {
        background-color: #111111;
        color: white;
    }
    .css-1d391kg, .css-1v0mbdj {
        color: white !important;
    }
    .stDataFrame {
        background-color: #222222 !important;
        color: white !important;
    }
    .stTable td {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Meta Reach Planner Analysis")
st.write("Upload your Meta Reach Planner CSV or Excel file to analyze optimum and maximum reach insights.")

# File uploader
uploaded_file = st.file_uploader("Upload Reach Planner CSV or Excel File", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        # Load the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            df_raw = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            st.error("❌ Unsupported file format.")
            st.stop()

        # Use appropriate columns
        df = df_raw[['Budget', 'Reach at 1+ frequency']].copy()
        df = df.rename(columns={'Reach at 1+ frequency': 'Reach'})
        df = df.sort_values('Budget').reset_index(drop=True)

        # Calculate incremental values
        df['Previous Reach'] = df['Reach'].shift(1)
        df['Incremental Reach'] = df['Reach'] - df['Previous Reach']
        df['Previous Budget'] = df['Budget'].shift(1)
        df['Incremental Spend'] = df['Budget'] - df['Previous Budget']
        df['Cost per 1000 Incremental Reach'] = (df['Incremental Spend'] / df['Incremental Reach'])

        # Identify max reach
        max_row = df.loc[df['Reach'].idxmax()]
        max_reach = max_row['Reach']
        max_budget = max_row['Budget']

        # Optimum reach (last few rows)
        optimum_row = df.tail(3).iloc[0]
        optimum_reach = optimum_row['Reach']
        optimum_budget = optimum_row['Budget']

        # Custom reach slider
        st.sidebar.header("🎯 Custom Reach Targets")
        custom_percent = st.sidebar.slider("Choose custom % of Max Reach", 10, 100, 70)
        target = max_reach * (custom_percent / 100)
        closest_row = df.iloc[(df['Reach'] - target).abs().argsort()[:1]]
        custom_budget = closest_row['Budget'].values[0]
        custom_reach = closest_row['Reach'].values[0]

        # Display summary table with comma formatting
        st.subheader("📋 Summary Table")
        summary = pd.DataFrame({
            'Metric': ['Maximum Reach', 'Optimum Reach (Flattening)', f'{custom_percent}% Reach'],
            'Reach': [f"{int(max_reach):,}", f"{int(optimum_reach):,}", f"{int(custom_reach):,}"],
            'Budget (LKR)': [f"{int(max_budget):,}", f"{int(optimum_budget):,}", f"{int(custom_budget):,}"]
        })
        st.dataframe(summary, use_container_width=True)

        # Plotly dark themed chart
        st.subheader("📈 Reach Curve")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Budget'],
            y=df['Reach'],
            mode='lines+markers',
            name='Reach Curve',
            line=dict(color='deepskyblue')
        ))

        # Add markers
        fig.add_vline(x=max_budget, line=dict(color='red', dash='dash'),
                      annotation_text=f'Max Reach\nLKR {int(max_budget):,}', annotation_position="top right")

        fig.add_vline(x=optimum_budget, line=dict(color='green', dash='dash'),
                      annotation_text=f'Optimum Reach\nLKR {int(optimum_budget):,}', annotation_position="top left")

        fig.add_vline(x=custom_budget, line=dict(color='purple', dash='dash'),
                      annotation_text=f'{custom_percent}% Reach\nLKR {int(custom_budget):,}', annotation_position="bottom left")

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
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color='white'),
            height=500,
            legend=dict(font=dict(color='white'))
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file, Remove the first 15 columns and only include the data sheet only.. {e}")
