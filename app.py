import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimum Budget Finder", layout="centered")

st.title("📊 Optimum Budget Detection – Efficiency vs Budget")
st.write("""
Upload your **CSV file** with at least 'Budget' and 'Efficiency' columns.
The app will find the 'pseudo-knee' optimum point and plot the curve.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if ('Budget' not in df.columns) or ('Efficiency' not in df.columns):
        st.error("CSV must have 'Budget' and 'Efficiency' columns!")
    else:
        # Sort and calculate deltas
        df = df.sort_values('Budget')
        df['Delta_Efficiency'] = df['Efficiency'].diff()
        df['Delta_Budget'] = df['Budget'].diff()
        df['Eff_per_LKR'] = df['Delta_Efficiency'] / df['Delta_Budget']

        # Find pseudo-knee (plateau)
        threshold = 0.00001  # Tune as needed
        pseudo_knee_idx = df[df['Eff_per_LKR'] < threshold].index.min()

        if not np.isnan(pseudo_knee_idx):
            pseudo_knee_budget = df.loc[pseudo_knee_idx, 'Budget']
            pseudo_knee_eff = df.loc[pseudo_knee_idx, 'Efficiency']
            st.success(f"**Optimum Budget (Pseudo-Knee/Plateau): {pseudo_knee_budget:,.2f}**\n(Efficiency: {pseudo_knee_eff:.4f})")
        else:
            pseudo_knee_idx = df['Efficiency'].idxmax()
            pseudo_knee_budget = df.loc[pseudo_knee_idx, 'Budget']
            pseudo_knee_eff = df.loc[pseudo_knee_idx, 'Efficiency']
            st.warning(f"No efficiency plateau found. Using Max Efficiency point: **Budget: {pseudo_knee_budget:,.2f} (Efficiency: {pseudo_knee_eff:.4f})**")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Budget'], df['Efficiency'], label='Efficiency vs Budget', color='tab:blue')
        ax.scatter(pseudo_knee_budget, pseudo_knee_eff, color='orange', s=120,
                   label=f'Optimum: {pseudo_knee_budget:,.0f}\n(Efficiency: {pseudo_knee_eff:.2f})', zorder=5)
        ax.annotate(
            f"{pseudo_knee_budget:,.0f}\n{pseudo_knee_eff:.2f}",
            (pseudo_knee_budget, pseudo_knee_eff),
            textcoords="offset points",
            xytext=(10,10),
            ha='left',
            fontsize=11,
            color='black',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="orange", alpha=0.2)
        )
        ax.set_xlabel('Budget')
        ax.set_ylabel('Efficiency')
        ax.set_title('Efficiency vs Budget with Optimum Point')
        ax.grid(True, linestyle='--', linewidth=0.6)
        ax.legend()
        st.pyplot(fig)
        
        st.write("#### Data Preview")
        st.dataframe(df.head(20))
else:
    st.info("Please upload a CSV file with 'Budget' and 'Efficiency' columns to begin.")
