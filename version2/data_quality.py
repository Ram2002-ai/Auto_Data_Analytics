import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

def quality_report(df):
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>📋 Data Quality Report</h2>
        <p style='color: gray;'>Comprehensive data quality assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall quality score
    st.subheader("📊 Overall Data Quality Score")
    
    # Calculate various quality metrics
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    uniqueness = (1 - df.duplicated().sum() / df.shape[0]) * 100
    
    # Data type consistency
    type_consistency = 100
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column has consistent types
            try:
                pd.to_numeric(df[col], errors='raise')
                # If convertible to numeric, it might be inconsistent
                type_consistency -= 5
            except:
                pass
    
    # Outlier impact
    outlier_impact = 100
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_pct = len(outliers) / len(df) * 100
            if outlier_pct > 10:
                outlier_impact -= 10
    
    quality_score = (completeness + uniqueness + type_consistency + outlier_impact) / 4
    
    # Display gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=quality_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Quality Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2E86AB"},
            'steps': [
                {'range': [0, 50], 'color': "#FF6B6B"},
                {'range': [50, 70], 'color': "#FFD93D"},
                {'range': [70, 85], 'color': "#6BCB77"},
                {'range': [85, 100], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Completeness", f"{completeness:.1f}%",
                 delta=None, delta_color="normal")
    
    with col2:
        st.metric("Uniqueness", f"{uniqueness:.1f}%",
                 delta=None, delta_color="normal")
    
    with col3:
        st.metric("Type Consistency", f"{type_consistency:.1f}%",
                 delta=None, delta_color="normal")
    
    with col4:
        st.metric("Outlier Impact", f"{outlier_impact:.1f}%",
                 delta=None, delta_color="inverse")
    
    # Detailed quality report
    st.subheader("🔍 Detailed Quality Report")
    
    quality_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Unique %': [round((df[col].nunique() / len(df) * 100),2) for col in df.columns],
        'Duplicate Values?': [df[col].duplicated().any() for col in df.columns]
    })
    
    # Add outlier info for numeric columns
    outlier_info = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_info.append(len(outliers))
        else:
            outlier_info.append(0)
    
    quality_df['Outliers'] = outlier_info
    
    st.dataframe(quality_df.style.background_gradient(subset=['Missing %', 'Outliers'], cmap='YlOrRd'),
                use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Quality Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing values bar chart
        missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
        if len(missing_cols) > 0:
            fig = px.bar(x=missing_cols.index, y=missing_cols.values,
                        title="Missing Values by Column",
                        labels={'x': 'Column', 'y': 'Missing Count'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")
    
    with col2:
        # Data type distribution
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values, names=dtype_counts.index.astype(str),
                    title="Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Outlier detection with Isolation Forest
    st.subheader("🕵️ Anomaly Detection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        contamination = st.slider("Expected outlier proportion", 0.01, 0.5, 0.1, 0.01)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(df[numeric_cols].fillna(0))
        
        n_outliers = (outliers == -1).sum()
        st.write(f"**Detected Anomalies:** {n_outliers} rows ({n_outliers/len(df)*100:.2f}%)")
        
        # Visualize outliers (if 2 or 3 numeric columns)
        if len(numeric_cols) >= 2:
            df_with_outliers = df[numeric_cols[:3]].copy()
            df_with_outliers['is_outlier'] = outliers
            
            if len(numeric_cols) == 2:
                fig = px.scatter(df_with_outliers, x=numeric_cols[0], y=numeric_cols[1],
                               color='is_outlier', title="Anomaly Detection Results",
                               color_continuous_scale=['blue', 'red'])
                st.plotly_chart(fig, use_container_width=True)
            elif len(numeric_cols) >= 3:
                fig = px.scatter_3d(df_with_outliers, x=numeric_cols[0], 
                                   y=numeric_cols[1], z=numeric_cols[2],
                                   color='is_outlier', title="Anomaly Detection Results (3D)",
                                   color_continuous_scale=['blue', 'red'])
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available for anomaly detection")
    
    # Recommendations
    st.subheader("💡 Quality Improvement Recommendations")
    
    recommendations = []
    
    # Missing value recommendations
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        recommendations.append(f"• Handle missing values in {len(missing_cols)} columns: {', '.join(missing_cols[:5])}")
    
    # Duplicate recommendations
    if df.duplicated().sum() > 0:
        recommendations.append(f"• Remove {df.duplicated().sum()} duplicate rows")
    
    # Outlier recommendations
    outlier_cols = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        if len(outliers) > len(df) * 0.1:  # More than 10% outliers
            outlier_cols.append(col)
    
    if outlier_cols:
        recommendations.append(f"• Investigate outliers in: {', '.join(outlier_cols[:3])}")
    
    # Data type recommendations
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column should be numeric
            try:
                pd.to_numeric(df[col].dropna().iloc[:100])
                recommendations.append(f"• Convert '{col}' to numeric type")
            except:
                pass
    
    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.success("✅ Dataset quality looks good! No major issues detected.")
    
    # Download quality report
    report_text = f"""
    DATA QUALITY REPORT
    ===================
    
    Overall Quality Score: {quality_score:.1f}/100
    
    Metrics:
    • Completeness: {completeness:.1f}%
    • Uniqueness: {uniqueness:.1f}%
    • Type Consistency: {type_consistency:.1f}%
    • Outlier Impact: {outlier_impact:.1f}%
    
    Dataset Statistics:
    • Rows: {df.shape[0]:,}
    • Columns: {df.shape[1]}
    • Missing Values: {df.isnull().sum().sum():,}
    • Duplicate Rows: {df.duplicated().sum():,}
    
    Recommendations:
    {chr(10).join(recommendations)}
    """
    
    st.download_button(
        label="📥 Download Quality Report",
        data=report_text,
        file_name="data_quality_report.txt",
        mime="text/plain",
        use_container_width=True
    )