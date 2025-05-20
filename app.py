import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Budget Optimum Detection", layout="centered")

st.title("📊 Optimum Budget Detection – Meta Reach Data")
st.write("""
Upload your **CSV file** (as exported from Meta Reach Planner).  
The app will process the data, calculate Efficiency, and visualize Reach & Efficiency vs Budget,
with the optimum point (where efficiency change drops the most) highlighted.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Sanity check for required columns
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
        # Replace infinities or NaN in first row (optional)
        df['Efficiency'] = df['Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')
        
        # MinMax Scaling for Budget and Efficiency
        scaler = MinMaxScaler()
        df['Scaled Budget'] = scaler.fit_transform(df[['Budget']])
        df['Scaled Efficiency'] = scaler.fit_transform(df[['Efficiency']])

        # Calculate the slope/difference between consecutive points
        df['Efficiency Change'] = np.diff(df['Scaled Efficiency'], prepend=np.nan)

        # Identify the point where the change in efficiency starts to decrease (knee point)
        optimal_budget_index = df['Efficiency Change'].idxmin()  # Find index of minimum change
        optimal_budget = df.loc[optimal_budget_index, 'Budget']
        optimal_efficiency = df.loc[optimal_budget_index, 'Efficiency']
        optimal_reach = df.loc[optimal_budget_index, 'Reach at 1+ frequency']

        st.success(f"**Optimal Budget (Change in Efficiency minimum): {optimal_budget:,.2f}**")
        st.write(f"Efficiency at this point: {optimal_efficiency:.2f}")

        # Visualization
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Primary y-axis: Reach at 1+ frequency
        ax1.set_xlabel('Budget')
        ax1.set_ylabel('Reach at 1+ Frequency', color='tab:blue')
        ax1.plot(df['Budget'], df['Reach at 1+ frequency'], color='tab:blue', label='Reach at 1+ Frequency')
        ax1.scatter(optimal_budget, optimal_reach, color='red', s=120, label=f'Optimum Budget: {optimal_budget:,.0f}', zorder=5)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Secondary y-axis: Efficiency
        ax2 = ax1.twinx()
        ax2.set_ylabel('Efficiency', color='tab:green')
        ax2.plot(df['Budget'], df['Efficiency'], color='tab:green', linestyle='--', label='Efficiency')
        ax2.scatter(optimal_budget, optimal_efficiency, color='red', s=120, label=f'Efficiency at Optimum', zorder=5)
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Title & layout
        plt.title('Reach at 1+ Frequency & Efficiency vs Budget\nwith Optimum Point Highlighted')
        fig.tight_layout()

        # Legend handling
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

        st.pyplot(fig)

        st.write("#### Data Preview")
        st.dataframe(df.head(20))
else:
    st.info("Please upload a CSV file with 'Budget' and 'Reach at 1+ frequency' columns to begin.")
