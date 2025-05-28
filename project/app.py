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

# Configure page
st.set_page_config(page_title="Auto EDA Tool with Missing Data Assistant", layout="wide")
st.title("📊 Auto EDA Tool with Missing Data Assistant")
st.markdown("Upload your CSV or Excel file to explore your data and handle missing values.")

# File uploader
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

    # Data Preview
    st.subheader("🔍 Data Preview (first 5 rows)")
    st.dataframe(df.head())

    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include='all'))

    # Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    if not corr.empty:
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric data for correlation heatmap.")

    # Histogram
    st.subheader("📊 Histogram")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        selected_col = st.selectbox("Choose a column to plot:", num_cols)
        fig2 = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No numeric columns available for histogram.")

    # Missing Data Analysis
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

    # Regression Analysis
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

            # Feature Importance Visualization
            st.subheader("📊 Feature Importance Visualization")
            fig_coef = px.bar(coef_df, x='Feature', y='Abs Coefficient', 
                            title='Feature Importance (Absolute Coefficient Values)',
                            color='Coefficient', color_continuous_scale='RdBu',
                            labels={'Abs Coefficient': 'Importance'})
            st.plotly_chart(fig_coef, use_container_width=True)

    # Advanced Data Profiling
    st.subheader("🔍 Advanced Data Profiling")
    if st.checkbox("Show detailed column information"):
        profile_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Unique Values': df.nunique(),
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Sample Values': [df[col].dropna().sample(min(3, len(df))).tolist() for col in df.columns]
        })
        st.dataframe(profile_df)

   
    # Data Transformation
    st.subheader("🛠 Data Transformation")
    transform_col = st.selectbox("Select column to transform:", df.columns)
    
    col1, col2 = st.columns(2)
    with col1:
        transform_type = st.selectbox(
            "Transformation type:",
            options=["Log", "Square root", "Normalize", "Standardize", "One-hot encode"]
        )
    with col2:
        if transform_type in ["Log", "Square root"]:
            if pd.api.types.is_numeric_dtype(df[transform_col]):
                new_col_name = st.text_input("New column name:", f"{transform_col}_{transform_type.lower()}")
            else:
                st.warning("Selected transformation only works for numeric columns")
        else:
            new_col_name = st.text_input("New column name:", f"{transform_col}_{transform_type.lower()}")
    
    if st.button("Apply Transformation"):
        try:
            if transform_type == "Log":
                df[new_col_name] = np.log1p(df[transform_col])
            elif transform_type == "Square root":
                df[new_col_name] = np.sqrt(df[transform_col])
            elif transform_type == "Normalize":
                df[new_col_name] = (df[transform_col] - df[transform_col].min()) / (df[transform_col].max() - df[transform_col].min())
            elif transform_type == "Standardize":
                df[new_col_name] = (df[transform_col] - df[transform_col].mean()) / df[transform_col].std()
            elif transform_type == "One-hot encode":
                encoded = pd.get_dummies(df[transform_col], prefix=transform_col)
                df = pd.concat([df, encoded], axis=1)
            
            st.success(f"Transformation applied successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error applying transformation: {str(e)}")

    # Sweetviz Report
    st.subheader("🧠 Sweetviz Full EDA Report")
    if st.button("Generate Sweetviz Report"):
        with st.spinner("Generating Sweetviz report..."):
            # Create a clean copy for Sweetviz
            df_sweetviz = df.copy()
            
            # Convert mixed-type columns to string
            for col in df_sweetviz.columns:
                if df_sweetviz[col].apply(type).nunique() > 1:
                    df_sweetviz[col] = df_sweetviz[col].astype(str)
            
            # Generate report
            try:
                report = sv.analyze(df_sweetviz)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    report.show_html(tmp_file.name)
                    html_file = tmp_file.name

                with open(html_file, "r", encoding="utf-8") as f:
                    html_content = f.read()

                components.html(html_content, height=1000, scrolling=True)
                
                # Download link
                b64 = base64.b64encode(html_content.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="sweetviz_report.html">📥 Download Sweetviz Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to generate Sweetviz report: {str(e)}")
                st.info("Try excluding problematic columns or converting them to consistent types.")

    # Data Export
    st.subheader("💾 Export Processed Data")
    export_format = st.selectbox("Select export format:", ["CSV", "Excel"])
    
    if st.button("Export Data"):
        try:
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
            else:
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name="processed_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

    # Reset button
    if st.button("🔄 Reset All Changes"):
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Data reset to original state!")
