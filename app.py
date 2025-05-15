import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt

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

        # Optimum Reach (based on where curve flattens - you can refine this logic)
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

        # Plot
        st.subheader("📈 Reach Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Budget'], df['Reach at 1+ Frequency'], marker='o', label='Reach Curve')
        ax.axvline(x=max_budget, color='red', linestyle='--', label=f'Max Reach\nLKR {max_budget:,.0f}')
        ax.axvline(x=optimum_budget, color='green', linestyle='--', label=f'Optimum Reach\nLKR {optimum_budget:,.0f}')
        ax.axvline(x=custom_budget, color='purple', linestyle='--', label=f'{custom_percent}% Reach\nLKR {custom_budget:,.0f}')
        ax.scatter([max_budget, optimum_budget, custom_budget],
                   [max_reach, optimum_reach, custom_reach],
                   color=['red', 'green', 'purple'], s=100, zorder=5)
        ax.set_xlabel("Budget (LKR)")
        ax.set_ylabel("Reach at 1+ Frequency")
        ax.set_title("Reach vs Budget")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
