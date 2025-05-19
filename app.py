import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from kneed import KneeLocator

# Streamlit config
st.set_page_config(page_title="Meta Reach Planner Analyzer", layout="centered")

# Dark styling
st.markdown("""
    <style>
    .stApp { background-color: #111111; color: white; }
    .stDataFrame, .css-1v0mbdj, .css-1d391kg { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Meta Reach Planner Analyzer")
st.write("Upload a Meta Reach Planner file and select a frequency level to visualize and analyze optimum reach performance.")

# File uploader
uploaded_file = st.file_uploader("Upload Reach Planner CSV or Excel File", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df_raw = pd.read_excel(uploaded_file, engine="openpyxl")
        elif uploaded_file.name.endswith(".xls"):
            df_raw = pd.read_excel(uploaded_file, engine="xlrd")
        else:
            st.error("❌ Unsupported file format.")
            st.stop()

        # Extract and map frequencies
        frequency_columns = [col for col in df_raw.columns if "Reach at " in col and "+ frequency" in col]
        frequency_map = {f"{i+1}+": col for i, col in enumerate(frequency_columns)}

        if not frequency_map:
            st.error("❌ No valid frequency columns found in the file.")
            st.stop()

        # Sidebar: frequency selector
        st.sidebar.header("📶 Frequency Level")
        selected_display = st.sidebar.selectbox("Select Reach Frequency Level", list(frequency_map.keys()))
        selected_column = frequency_map[selected_display]

        # Prepare dataframe
        df = df_raw[['Budget', selected_column]].copy()
        df = df.rename(columns={selected_column: 'Reach'})
        df = df.sort_values('Budget').reset_index(drop=True)

        # Find maximum reach
        max_row = df.loc[df['Reach'].idxmax()]
        max_reach = max_row['Reach']
        max_budget = max_row['Budget']

        # Use kneed to find the "optimum" (flattening) point
        try:
            knee = KneeLocator(df['Budget'], df['Reach'], curve='concave', direction='increasing')
            if knee.knee is not None:
                optimum_budget = df.loc[knee.knee, 'Budget']
                optimum_reach = df.loc[knee.knee, 'Reach']
            else:
                optimum_budget = df['Budget'].iloc[-3]
                optimum_reach = df['Reach'].iloc[-3]
        except:
            optimum_budget = df['Budget'].iloc[-3]
            optimum_reach = df['Reach'].iloc[-3]

        # Sidebar slider for custom % of max reach
        st.sidebar.header("🎯 Custom Reach Target")
        custom_percent = st.sidebar.slider("Choose custom % of Max Reach", 10, 100, 70)
        custom_target = max_reach * (custom_percent / 100)

        closest_row = df.iloc[(df['Reach'] - custom_target).abs().argsort()[:1]]
        custom_budget = closest_row['Budget'].values[0]
        custom_reach = closest_row['Reach'].values[0]

        # Summary table
        st.subheader("📋 Summary Table")
        summary = pd.DataFrame({
            "Metric": ["Maximum Reach", "Optimum Reach (via Kneedle)", f"{custom_percent}% Reach"],
            "Reach": [f"{int(max_reach):,}", f"{int(optimum_reach):,}", f"{int(custom_reach):,}"],
            "Budget (LKR)": [f"{int(max_budget):,}", f"{int(optimum_budget):,}", f"{int(custom_budget):,}"]
        })
        st.dataframe(summary, use_container_width=True)

        # Plotly chart
        st.subheader("📈 Reach Curve")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Budget'], y=df['Reach'],
            mode="lines+markers", name="Reach Curve", line=dict(color='deepskyblue')
        ))

        # Add vertical lines
        fig.add_vline(x=max_budget, line=dict(color='red', dash='dash'),
                      annotation_text=f'Max Reach\nLKR {int(max_budget):,}', annotation_position="top right")
        fig.add_vline(x=optimum_budget, line=dict(color='green', dash='dash'),
                      annotation_text=f'Optimum (Kneed)\nLKR {int(optimum_budget):,}', annotation_position="top left")
        fig.add_vline(x=custom_budget, line=dict(color='purple', dash='dash'),
                      annotation_text=f'{custom_percent}% Reach\nLKR {int(custom_budget):,}', annotation_position="bottom left")

        # Key points
        fig.add_trace(go.Scatter(
            x=[max_budget, optimum_budget, custom_budget],
            y=[max_reach, optimum_reach, custom_reach],
            mode='markers',
            marker=dict(color=['red', 'green', 'purple'], size=10),
            name="Key Points"
        ))

        fig.update_layout(
            xaxis_title="Budget (LKR)",
            yaxis_title=f"Reach at {selected_display} Frequency",
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color='white'),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file ONLY INCLUDE THE DATA SHEET AND REMOVE REMAINING INFORMATION: {e}")
