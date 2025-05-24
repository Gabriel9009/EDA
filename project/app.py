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
        features = [col for col in numeric_cols if col != target]

        if not features:
            st.warning("No numeric features available for regression besides the target.")
        else:
            X = df[features].copy()
            y = df[target].copy()

            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            coefs = model.coef_
            coef_df = pd.DataFrame({
                "Feature": features,
                "Coefficient": coefs,
                "Abs Coefficient": np.abs(coefs)
            }).sort_values(by="Abs Coefficient", ascending=False)

            st.write("### Regression Coefficients (standardized features):")
            st.dataframe(coef_df[["Feature", "Coefficient"]].style.format({"Coefficient": "{:.4f}"}))

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            st.markdown(f"""
                **Model Performance:**
                - MAE: `{mae:.4f}`
                - MSE: `{mse:.4f}`
                - RMSE: `{rmse:.4f}`
                - R² Score: `{r2:.4f}`
            """)

            top_feature = coef_df.iloc[0]
            st.success(f"Most powerful predictor: **{top_feature['Feature']}** with coefficient {top_feature['Coefficient']:.4f}")

    # Sweetviz Report
    st.subheader("🧠 Sweetviz Full EDA Report")
    if st.button("Generate Sweetviz Report"):
        with st.spinner("Generating Sweetviz report..."):
            report = sv.analyze(df)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                report.show_html(tmp_file.name)
                html_file = tmp_file.name

            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            components.html(html_content, height=1000, scrolling=True)
            # Convert HTML to base64
            b64 = base64.b64encode(html_content.encode()).decode()

            # Create a download link
            href = f'<a href="data:text/html;base64,{b64}" download="sweetviz_report.html">📥 Download Sweetviz Report</a>'
            st.markdown(href, unsafe_allow_html=True)
