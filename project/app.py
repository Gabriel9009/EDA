import streamlit as st
import base64
import pandas as pd
import sweetviz as sv
import streamlit.components.v1 as components
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Auto EDA Tool with Missing Data Assistant", layout="wide")
st.title("📊 Auto EDA Tool with Missing Data Assistant")
st.markdown("Upload your CSV or Excel file to explore your data and handle missing values.")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.lower().split('.')[-1]

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # Early check for mixed-type columns
    mixed_cols = [col for col in df.columns if df[col].apply(type).nunique() > 1]
    if mixed_cols:
        st.warning(f"⚠️ Warning: Mixed-type columns detected: {', '.join(mixed_cols)}")
        st.info("These columns will be automatically converted to string type for analysis.")

    st.subheader("🔍 Data Preview (first 5 rows)")
    st.dataframe(df.head())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include='all'))

    st.subheader("🔗 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    if not corr.empty:
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric data for correlation heatmap.")

    st.subheader("📊 Histogram")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        selected_col = st.selectbox("Choose a column to plot:", num_cols)
        fig2 = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No numeric columns available for histogram.")

    # Missing Data Summary
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]

    if missing_info.empty:
        st.info("No missing data detected!")
    else:
        st.subheader("⚠️ Missing Data Summary")
        missing_df = pd.DataFrame({
            "Missing Count": missing_info,
            "Missing Percentage": (missing_info / len(df) * 100).round(2)
        })
        st.dataframe(missing_df)

        # Imputation strategies
        st.subheader("🧹 Missing Data Imputation")
        impute_values = {}
        for col in missing_info.index:
            col_type = df[col].dtype
            st.markdown(f"**Column:** `{col}` (Type: {col_type})")
            if pd.api.types.is_numeric_dtype(df[col]):
                strategy = st.selectbox(
                    f"Imputation strategy for numeric column '{col}'",
                    options=["Mean", "Median", "Mode", "Constant (0)", "Leave as is"],
                    key=f"imp_{col}"
                )
                if strategy == "Mean":
                    impute_values[col] = df[col].mean()
                elif strategy == "Median":
                    impute_values[col] = df[col].median()
                elif strategy == "Mode":
                    impute_values[col] = df[col].mode()[0]
                elif strategy == "Constant (0)":
                    impute_values[col] = 0
                else:
                    impute_values[col] = None
            else:
                strategy = st.selectbox(
                    f"Imputation strategy for categorical column '{col}'",
                    options=["Mode", "Constant ('Unknown')", "Leave as is"],
                    key=f"imp_{col}"
                )
                if strategy == "Mode":
                    impute_values[col] = df[col].mode()[0]
                elif strategy == "Constant ('Unknown')":
                    impute_values[col] = "Unknown"
                else:
                    impute_values[col] = None

        if st.button("Apply Imputation"):
            for col, val in impute_values.items():
                if val is not None:
                    df[col].fillna(val, inplace=True)
            st.success("✅ Missing values imputed as per your selections.")
            st.subheader("🔍 Data Preview After Imputation")
            st.dataframe(df.head())

            # Optionally: Download cleaned data
            st.download_button(
                label="📥 Download Cleaned Data",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="cleaned_data.csv",
                mime='text/csv'
            )

    # Regression
    st.subheader("🎯 Target Variable Selection and Linear Regression Analysis")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    possible_targets = [col for col in numeric_cols if df[col].nunique() > 1]

    target = st.selectbox("Select your target variable for regression", options=[None] + possible_targets)

    if target:
        st.write(f"Selected target: **{target}**")
        features = [col for
