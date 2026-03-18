import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def generate_business_insights(df):
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>💡 Automated Business Insights</h2>
        <p style='color: gray;'>AI-powered analysis to uncover hidden patterns and opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Create tabs for different insight categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Trends & Patterns", "🎯 Key Drivers", 
        "⚠️ Anomalies", "💡 Recommendations"
    ])
    
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with col4:
            if numeric_cols:
                total_value = df[numeric_cols].sum().sum()
                st.metric("Total Value", f"{total_value:,.0f}" if total_value < 1e6 else f"{total_value/1e6:,.1f}M")
        
        # Column composition
        st.markdown("### 📋 Column Composition")
        
        comp_data = {
            'Type': ['Numeric', 'Categorical', 'Datetime'],
            'Count': [len(numeric_cols), len(categorical_cols), len(datetime_cols)]
        }
        
        fig = px.pie(comp_data, values='Count', names='Type',
                    title="Column Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data quality score
        st.markdown("### 📊 Data Quality Score")
        
        quality_score = 0
        quality_metrics = []
        
        # Completeness score
        completeness_score = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        quality_metrics.append(completeness_score)
        
        # Uniqueness score (avoid duplicates)
        duplicate_pct = (df.duplicated().sum() / df.shape[0]) * 100
        uniqueness_score = 100 - duplicate_pct
        quality_metrics.append(uniqueness_score)
        
        # Consistency score (data type consistency)
        type_consistency = 100  # Default high
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column has mixed types
                try:
                    pd.to_numeric(df[col], errors='raise')
                    # If convertible to numeric, it's consistent
                except:
                    pass  # Object type is fine
            else:
                # Numeric columns are consistent
                pass
        quality_metrics.append(type_consistency)
        
        # Average quality score
        avg_quality = np.mean(quality_metrics)
        
        # Display gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_quality,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Data Quality"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📈 Trends & Patterns")
        
        if len(numeric_cols) > 0:
            # Correlation analysis
            if len(numeric_cols) >= 2:
                st.markdown("### 🔗 Key Relationships")
                
                corr_matrix = df[numeric_cols].corr()
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_pairs.append({
                            'feature1': numeric_cols[i],
                            'feature2': numeric_cols[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                # Display top 5 correlations
                for pair in corr_pairs[:5]:
                    strength = abs(pair['correlation'])
                    if strength > 0.7:
                        emoji = "🟢"
                        desc = "Strong"
                    elif strength > 0.3:
                        emoji = "🟡"
                        desc = "Moderate"
                    else:
                        emoji = "🔴"
                        desc = "Weak"
                    
                    direction = "positive" if pair['correlation'] > 0 else "negative"
                    
                    st.markdown(
                        f"{emoji} **{pair['feature1']}** & **{pair['feature2']}**: "
                        f"{pair['correlation']:.3f} ({desc} {direction} correlation)"
                    )
                
                # Insight
                if corr_pairs:
                    st.info(f"💡 **Insight**: {corr_pairs[0]['feature1']} and {corr_pairs[0]['feature2']} "
                           f"have the strongest {'positive' if corr_pairs[0]['correlation'] > 0 else 'negative'} "
                           f"relationship in the dataset.")
            
            # Distribution insights
            st.markdown("### 📊 Distribution Analysis")
            
            skewness = df[numeric_cols].skew()
            skewed_cols = skewness[abs(skewness) > 1].index.tolist()
            
            if skewed_cols:
                st.warning(f"⚠️ **Skewed Features**: {', '.join(skewed_cols[:3])}" +
                          (" and more" if len(skewed_cols) > 3 else ""))
                st.markdown("💡 These features might benefit from transformation for better model performance.")
            
            # Show distribution of most skewed feature
            if skewed_cols:
                col_to_show = skewed_cols[0]
                fig = px.histogram(df, x=col_to_show, nbins=30,
                                  title=f"Distribution of {col_to_show} (Most Skewed)",
                                  marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for trend analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🎯 Key Business Drivers")
        
        if len(numeric_cols) > 0:
            # Find features with highest variance (potential impact)
            variances = df[numeric_cols].var().sort_values(ascending=False)
            
            st.markdown("### 📊 High Variance Features")
            st.markdown("Features with high variance often indicate key business drivers")
            
            fig = px.bar(x=variances.index[:10], y=variances.values[:10],
                        title="Top 10 Features by Variance",
                        labels={'x': 'Feature', 'y': 'Variance'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance based on mutual information
            if len(numeric_cols) >= 2:
                st.markdown("### 🔍 Predictive Power")
                
                # Use last numeric column as potential target
                target = numeric_cols[-1]
                features = numeric_cols[:-1]
                
                if len(features) > 0:
                    from sklearn.feature_selection import mutual_info_regression
                    
                    mi_scores = mutual_info_regression(df[features].fillna(0), df[target].fillna(0))
                    mi_df = pd.DataFrame({
                        'feature': features,
                        'importance': mi_scores
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(mi_df.head(10), x='importance', y='feature',
                               orientation='h',
                               title=f"Feature Importance for Predicting {target}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"💡 **Key Driver**: {mi_df.iloc[0]['feature']} appears to be the most "
                           f"important factor for predicting {target}")
        else:
            st.info("No numeric columns available for driver analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Anomaly Detection")
        
        if len(numeric_cols) > 0:
            # Outlier detection using IQR
            outlier_report = []
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                outlier_pct = (len(outliers) / len(df)) * 100
                
                if outlier_pct > 5:
                    outlier_report.append({
                        'column': col,
                        'outlier_pct': outlier_pct,
                        'lower_bound': Q1 - 1.5 * IQR,
                        'upper_bound': Q3 + 1.5 * IQR
                    })
            
            if outlier_report:
                st.warning(f"⚠️ Found {len(outlier_report)} columns with significant outliers")
                
                for item in outlier_report[:5]:
                    st.markdown(f"**{item['column']}**: {item['outlier_pct']:.1f}% outliers "
                              f"(outside [{item['lower_bound']:.2f}, {item['upper_bound']:.2f}])")
                
                # Visualize outliers for first column
                col_to_show = outlier_report[0]['column']
                fig = px.box(df, y=col_to_show, title=f"Outliers in {col_to_show}")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("💡 **Recommendation**: Investigate these outliers - they may represent "
                           "unusual but important business events or data quality issues.")
            else:
                st.success("✅ No significant outliers detected in numeric columns")
        else:
            st.info("No numeric columns available for outlier detection")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("💡 Strategic Recommendations")
        
        # Generate business recommendations based on data insights
        recommendations = []
        
        if len(numeric_cols) > 0:
            # Check for growth opportunities
            growth_cols = []
            for col in numeric_cols:
                if df[col].min() >= 0 and df[col].max() > df[col].min() * 10:
                    growth_cols.append(col)
            
            if growth_cols:
                recommendations.append({
                    'area': 'Growth Opportunity',
                    'recommendation': f"Focus on {growth_cols[0]} which shows high variability "
                                    f"(range: {df[growth_cols[0]].min():.2f} to {df[growth_cols[0]].max():.2f})",
                    'priority': 'High'
                })
            
            # Check for efficiency opportunities
            if len(numeric_cols) >= 2:
                # Find features with high correlation - potential redundancy
                corr_matrix = df[numeric_cols].corr()
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if abs(corr_matrix.iloc[i, j]) > 0.9:
                            recommendations.append({
                                'area': 'Efficiency',
                                'recommendation': f"Consider consolidating {numeric_cols[i]} and {numeric_cols[j]} "
                                                f"as they are highly correlated ({corr_matrix.iloc[i, j]:.2f})",
                                'priority': 'Medium'
                            })
                            break
                    if len(recommendations) > 3:
                        break
        
        if categorical_cols:
            # Check for customer/market segments
            for col in categorical_cols[:2]:
                if df[col].nunique() > 1 and df[col].nunique() <= 10:
                    top_segment = df[col].value_counts().index[0]
                    recommendations.append({
                        'area': 'Segmentation',
                        'recommendation': f"Target the dominant segment in {col}: '{top_segment}' "
                                        f"({df[col].value_counts().iloc[0]:,} records)",
                        'priority': 'Medium'
                    })
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                priority_color = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
                st.markdown(f"{priority_color} **{rec['area']}**: {rec['recommendation']}")
        else:
            st.info("No specific recommendations generated. Try uploading a dataset with more variety.")
        
        # Add download insights option
        st.markdown("---")
        st.markdown("### 📥 Export Insights")
        
        insight_text = f"""
        BUSINESS INSIGHTS REPORT
        =======================
        
        Dataset: {df.shape[0]} rows × {df.shape[1]} columns
        
        KEY METRICS:
        • Total Records: {df.shape[0]:,}
        • Total Features: {df.shape[1]}
        • Data Completeness: {completeness:.1f}%
        
        COLUMN COMPOSITION:
        • Numeric: {len(numeric_cols)}
        • Categorical: {len(categorical_cols)}
        • Datetime: {len(datetime_cols)}
        
        RECOMMENDATIONS:
        """
        
        for rec in recommendations:
            insight_text += f"\n• {rec['area']}: {rec['recommendation']} (Priority: {rec['priority']})"
        
        st.download_button(
            label="📥 Download Insights Report",
            data=insight_text,
            file_name="business_insights.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)