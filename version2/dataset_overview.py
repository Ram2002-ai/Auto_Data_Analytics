import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def eda_analysis(df):
    """
    Comprehensive Exploratory Data Analysis (EDA) with visual insights
    """
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>🔍 Exploratory Data Analysis (EDA)</h2>
        <p style='color: gray;'>Discover patterns, relationships, and insights through visual exploration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Error handling
    if df.empty:
        st.error("❌ The dataset is empty. Please upload a valid dataset.")
        return
    
    try:
        # Create tabs for different EDA aspects
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Data Overview", 
            "🔍 Missing Data Analysis", 
            "📊 Univariate Analysis",
            "🔄 Bivariate Analysis", 
            "📈 Multivariate Analysis", 
            "🎯 Pattern Discovery"
        ])
        
        with tab1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📋 Dataset Overview")
            
            try:
                # Key metrics in cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Total Columns", df.shape[1])
                with col3:
                    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("Memory Usage", f"{memory_usage:.2f} MB")
                with col4:
                    missing_total = df.isnull().sum().sum()
                    st.metric("Missing Values", f"{missing_total:,}")
                
                # Data preview with interactive controls
                st.subheader("🔍 Data Preview")
                col1, col2 = st.columns(2)
                with col1:
                    preview_rows = st.slider("Number of rows to display", 5, 50, 10, key="preview_rows")
                with col2:
                    preview_type = st.radio("Preview type", ["Head", "Tail", "Random Sample"], 
                                           horizontal=True, key="preview_type")
                
                if preview_type == "Head":
                    st.dataframe(df.head(preview_rows), use_container_width=True)
                elif preview_type == "Tail":
                    st.dataframe(df.tail(preview_rows), use_container_width=True)
                else:
                    if len(df) > preview_rows:
                        st.dataframe(df.sample(preview_rows), use_container_width=True)
                    else:
                        st.warning("⚠️ Sample size larger than dataset. Showing all rows.")
                        st.dataframe(df, use_container_width=True)
                
                # Column information with visual indicators
                st.subheader("📋 Column Information")
                
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values,
                    'Null %': (df.isnull().sum().values / len(df) * 100).round(2),
                    'Unique Values': [df[col].nunique() for col in df.columns],
                    'Sample Values': [str(df[col].dropna().iloc[:3].tolist()) if len(df[col].dropna()) > 0 else "All null" for col in df.columns]
                })
                
                # Add color coding for data types
                def color_data_type(val):
                    if 'int' in val or 'float' in val:
                        return 'background-color: #e3f2fd'
                    elif 'object' in val:
                        return 'background-color: #f1f8e9'
                    elif 'datetime' in val:
                        return 'background-color: #fff3e0'
                    return ''
                
                st.dataframe(col_info.style.applymap(color_data_type, subset=['Data Type']),
                            use_container_width=True)
                
                # Data type distribution
                st.subheader("📊 Data Type Distribution")
                
                dtype_counts = df.dtypes.value_counts()
                if len(dtype_counts) > 0:
                    fig = make_subplots(rows=1, cols=2,
                                       specs=[[{"type": "pie"}, {"type": "bar"}]],
                                       subplot_titles=("Pie Chart", "Bar Chart"))
                    
                    fig.add_trace(go.Pie(labels=dtype_counts.index.astype(str), 
                                        values=dtype_counts.values,
                                        hole=0.3), row=1, col=1)
                    
                    fig.add_trace(go.Bar(x=dtype_counts.index.astype(str), 
                                        y=dtype_counts.values,
                                        marker_color=['#42a5f5', '#66bb6a', '#ffa726'][:len(dtype_counts)]), 
                                 row=1, col=2)
                    
                    fig.update_layout(height=400, title_text="Column Types Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No data type information available")
                
                # Dataset statistics
                st.subheader("📈 Dataset Statistics")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.info(f"**Numeric:** {len(numeric_cols)} columns")
                with col2:
                    st.info(f"**Categorical:** {len(categorical_cols)} columns")
                with col3:
                    st.info(f"**Datetime:** {len(datetime_cols)} columns")
                with col4:
                    st.info(f"**Boolean:** {len(bool_cols)} columns")
            
            except Exception as e:
                st.error(f"❌ Error in data overview: {str(e)}")
                st.info("💡 Tip: Check if your dataset contains valid data types")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🔍 Missing Data Analysis")
            
            try:
                if df.isnull().sum().sum() > 0:
                    # Missing data overview
                    missing_df = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Count': df.isnull().sum().values,
                        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
                    }).sort_values('Missing %', ascending=False)
                    
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    
                    if len(missing_df) > 0:
                        # Visualize missing data
                        fig = make_subplots(rows=2, cols=2,
                                           subplot_titles=("Missing Values Heatmap", 
                                                          "Missing Values by Column",
                                                          "Missing Data Patterns",
                                                          "Missing Data Matrix"),
                                           specs=[[{"type": "heatmap"}, {"type": "bar"}],
                                                 [{"type": "scatter"}, {"type": "heatmap"}]])
                        
                        # Heatmap of missing values
                        missing_matrix = df.isnull().astype(int).T
                        fig.add_trace(go.Heatmap(z=missing_matrix.values,
                                                y=missing_matrix.index,
                                                colorscale='Reds',
                                                showscale=False), row=1, col=1)
                        
                        # Bar chart of missing values
                        fig.add_trace(go.Bar(x=missing_df['Column'].head(20), 
                                            y=missing_df['Missing Count'].head(20),
                                            marker_color='#ef5350',
                                            name="Missing Count"), row=1, col=2)
                        
                        # Missing data patterns (rows with missing data)
                        missing_rows = df[df.isnull().any(axis=1)]
                        if len(missing_rows) > 0:
                            pattern_df = missing_rows.isnull().sum(axis=1).value_counts().reset_index()
                            pattern_df.columns = ['Missing Count per Row', 'Number of Rows']
                            pattern_df = pattern_df.sort_values('Missing Count per Row')
                            
                            fig.add_trace(go.Scatter(x=pattern_df['Missing Count per Row'],
                                                    y=pattern_df['Number of Rows'],
                                                    mode='lines+markers',
                                                    name="Patterns"), row=2, col=1)
                        
                        # Missing data matrix for first 50 rows
                        sample_missing = df.head(min(50, len(df))).isnull().astype(int).T
                        fig.add_trace(go.Heatmap(z=sample_missing.values,
                                                y=sample_missing.index,
                                                colorscale='Reds',
                                                showscale=False,
                                                name="Matrix"), row=2, col=2)
                        
                        fig.update_layout(height=800, title_text="Missing Data Analysis",
                                        showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed missing data table
                        st.subheader("📋 Missing Data Details")
                        
                        # Add severity classification
                        def classify_severity(pct):
                            if pct == 0:
                                return "✅ None"
                            elif pct < 5:
                                return "🟢 Low"
                            elif pct < 20:
                                return "🟡 Medium"
                            else:
                                return "🔴 High"
                        
                        missing_df['Severity'] = missing_df['Missing %'].apply(classify_severity)
                        missing_df['Recommendation'] = missing_df['Missing %'].apply(
                            lambda x: "No action needed" if x == 0 else
                                     "Consider imputation" if x < 5 else
                                     "Imputation recommended" if x < 20 else
                                     "Consider dropping column"
                        )
                        
                        st.dataframe(missing_df, use_container_width=True)
                        
                        # Missing data patterns
                        if len(missing_df) > 1:
                            st.subheader("🔄 Missing Data Patterns")
                            
                            # Find columns with similar missing patterns
                            missing_corr = df[missing_df['Column'].tolist()].isnull().corr()
                            
                            if len(missing_corr) > 1:
                                fig = px.imshow(missing_corr,
                                               text_auto=True,
                                               aspect="auto",
                                               color_continuous_scale='RdBu_r',
                                               title="Missing Value Correlation Matrix")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Find highly correlated missing patterns
                                high_corr = []
                                for i in range(len(missing_corr.columns)):
                                    for j in range(i+1, len(missing_corr.columns)):
                                        if abs(missing_corr.iloc[i, j]) > 0.7:
                                            high_corr.append({
                                                'Column 1': missing_corr.columns[i],
                                                'Column 2': missing_corr.columns[j],
                                                'Correlation': missing_corr.iloc[i, j]
                                            })
                                
                                if high_corr:
                                    st.info("🔍 **Columns with similar missing patterns:**")
                                    for item in high_corr[:5]:  # Show top 5
                                        st.write(f"• {item['Column 1']} & {item['Column 2']}: {item['Correlation']:.2f}")
                    else:
                        st.success("✅ No missing values found in the dataset!")
                else:
                    st.success("✅ No missing values found in the dataset!")
                    
                    # Show complete data visualization
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="number+gauge",
                        value=100,
                        title={'text': "Data Completeness"},
                        gauge={'axis': {'range': [0, 100]},
                              'bar': {'color': "green"},
                              'steps': [{'range': [0, 100], 'color': "lightgreen"}]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error in missing data analysis: {str(e)}")
                st.info("💡 Tip: Ensure your dataset has valid data for missing value analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📊 Univariate Analysis")
            
            try:
                col_type = st.radio("Select column type", ["Numeric", "Categorical", "Datetime"], 
                                   horizontal=True, key="univariate_type")
                
                if col_type == "Numeric":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Select numeric column", numeric_cols, key="univariate_num")
                        
                        data = df[selected_col].dropna()
                        
                        if len(data) > 0:
                            # Create comprehensive visualization
                            fig = make_subplots(rows=2, cols=3,
                                               subplot_titles=("Histogram", "Box Plot", "Violin Plot",
                                                             "ECDF", "QQ Plot", "Summary Stats"),
                                               specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                                                     [{"type": "xy"}, {"type": "xy"}, {"type": "domain"}]])
                            
                            # Histogram
                            fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Histogram", 
                                                       marker_color='#42a5f5'), row=1, col=1)
                            
                            # Box plot
                            fig.add_trace(go.Box(y=data, name="Box Plot", boxpoints='outliers',
                                                marker_color='#66bb6a'), row=1, col=2)
                            
                            # Violin plot
                            fig.add_trace(go.Violin(y=data, name="Violin Plot", box_visible=True,
                                                   line_color='black', fillcolor='#ffa726',
                                                   opacity=0.6), row=1, col=3)
                            
                            # ECDF
                            sorted_data = np.sort(data)
                            ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
                            fig.add_trace(go.Scatter(x=sorted_data, y=ecdf, mode='lines',
                                                    name="ECDF", line=dict(color='#ab47bc')),
                                         row=2, col=1)
                            
                            # QQ plot
                            theoretical_q = np.random.normal(data.mean(), data.std(), len(data))
                            theoretical_q.sort()
                            fig.add_trace(go.Scatter(x=theoretical_q, y=sorted_data,
                                                    mode='markers', name="QQ Plot",
                                                    marker=dict(color='#7e57c2', size=3)),
                                         row=2, col=2)
                            
                            # Add reference line to QQ plot
                            min_val = min(theoretical_q.min(), sorted_data.min())
                            max_val = max(theoretical_q.max(), sorted_data.max())
                            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                                    mode='lines', line=dict(color='red', dash='dash'),
                                                    showlegend=False), row=2, col=2)
                            
                            # Summary statistics as table
                            stats_text = f"""
                            <b>Summary Statistics</b><br>
                            Count: {len(data):,}<br>
                            Mean: {data.mean():.4f}<br>
                            Std: {data.std():.4f}<br>
                            Min: {data.min():.4f}<br>
                            Q1: {data.quantile(0.25):.4f}<br>
                            Median: {data.median():.4f}<br>
                            Q3: {data.quantile(0.75):.4f}<br>
                            Max: {data.max():.4f}<br>
                            IQR: {data.quantile(0.75) - data.quantile(0.25):.4f}<br>
                            Skewness: {data.skew():.4f}<br>
                            Kurtosis: {data.kurtosis():.4f}
                            """
                            
                            fig.add_annotation(x=0.5, y=0.5, text=stats_text,
                                             showarrow=False, font=dict(size=10),
                                             row=2, col=3, align='left')
                            
                            fig.update_layout(height=800, title_text=f"Univariate Analysis: {selected_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Outlier detection
                            Q1 = data.quantile(0.25)
                            Q3 = data.quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Outliers Count", len(outliers))
                            with col2:
                                st.metric("Outliers %", f"{len(outliers)/len(data)*100:.2f}%")
                            
                            if len(outliers) > 0:
                                with st.expander("View outlier values"):
                                    st.write(outliers.tolist()[:20])  # Show first 20 outliers
                                    if len(outliers) > 20:
                                        st.info(f"... and {len(outliers) - 20} more outliers")
                    else:
                        st.warning("⚠️ No numeric columns available for analysis")
                
                elif col_type == "Categorical":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if categorical_cols:
                        selected_col = st.selectbox("Select categorical column", categorical_cols, 
                                                   key="univariate_cat")
                        
                        # Get value counts
                        value_counts = df[selected_col].value_counts().reset_index()
                        value_counts.columns = [selected_col, 'count']
                        value_counts['percentage'] = (value_counts['count'] / len(df) * 100).round(2)
                        
                        if len(value_counts) > 0:
                            # Create visualizations
                            fig = make_subplots(rows=2, cols=2,
                                               subplot_titles=("Bar Chart (Top 20)", "Pie Chart (Top 10)",
                                                              "Treemap (Top 10)", "Frequency Table"),
                                               specs=[[{"type": "xy"}, {"type": "domain"}],
                                                     [{"type": "domain"}, {"type": "table"}]])
                            
                            # Bar chart (top 20)
                            top20 = value_counts.head(20)
                            fig.add_trace(go.Bar(x=top20[selected_col], 
                                                y=top20['count'],
                                                marker_color='#42a5f5',
                                                name="Count"), row=1, col=1)
                            
                            # Pie chart (top 10)
                            top10 = value_counts.head(10)
                            fig.add_trace(go.Pie(labels=top10[selected_col], 
                                                values=top10['count'],
                                                hole=0.3,
                                                textinfo='percent+label',
                                                name="Proportion"), row=1, col=2)
                            
                            # Treemap (top 10)
                            fig.add_trace(go.Treemap(labels=top10[selected_col],
                                                    parents=['']*len(top10),
                                                    values=top10['count'],
                                                    textinfo='label+value',
                                                    name="Treemap"), row=2, col=1)
                            
                            # Frequency table (top 10)
                            fig.add_trace(go.Table(header=dict(values=[selected_col, 'Count', 'Percentage']),
                                                  cells=dict(values=[top10[selected_col].tolist(),
                                                                    top10['count'].tolist(),
                                                                    top10['percentage'].tolist()]),
                                                  name="Table"), row=2, col=2)
                            
                            fig.update_layout(height=800, title_text=f"Categorical Analysis: {selected_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary statistics for categorical
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Unique Values", f"{value_counts.shape[0]:,}")
                            with col2:
                                st.metric("Most Frequent", f"{value_counts.iloc[0, 0]}")
                            with col3:
                                st.metric("Frequency", f"{value_counts.iloc[0, 1]:,} ({value_counts.iloc[0, 2]}%)")
                            
                            # Cardinality warning
                            if value_counts.shape[0] > 50:
                                st.warning(f"⚠️ High cardinality detected: {value_counts.shape[0]} unique values. Consider grouping rare categories.")
                    else:
                        st.warning("⚠️ No categorical columns available for analysis")
                
                elif col_type == "Datetime":
                    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                    if datetime_cols:
                        selected_col = st.selectbox("Select datetime column", datetime_cols, 
                                                   key="univariate_datetime")
                        
                        # Extract temporal features
                        df_temp = df[selected_col].dropna()
                        
                        if len(df_temp) > 0:
                            # Create temporal distributions
                            fig = make_subplots(rows=2, cols=2,
                                               subplot_titles=("Year Distribution", "Month Distribution",
                                                              "Day of Week Distribution", "Hour Distribution"),
                                               specs=[[{"type": "xy"}, {"type": "xy"}],
                                                     [{"type": "xy"}, {"type": "xy"}]])
                            
                            # Year distribution
                            years = df_temp.dt.year.value_counts().sort_index()
                            if len(years) > 0:
                                fig.add_trace(go.Bar(x=years.index.astype(str), y=years.values,
                                                    marker_color='#42a5f5', name="Year"), row=1, col=1)
                            
                            # Month distribution
                            months = df_temp.dt.month.value_counts().sort_index()
                            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            if len(months) > 0:
                                fig.add_trace(go.Bar(x=[month_names[i-1] for i in months.index],
                                                    y=months.values, marker_color='#66bb6a',
                                                    name="Month"), row=1, col=2)
                            
                            # Day of week distribution
                            days = df_temp.dt.dayofweek.value_counts().sort_index()
                            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                            if len(days) > 0:
                                fig.add_trace(go.Bar(x=[day_names[i] for i in days.index],
                                                    y=days.values, marker_color='#ffa726',
                                                    name="Day of Week"), row=2, col=1)
                            
                            # Hour distribution (if time component exists)
                            if df_temp.dt.hour.nunique() > 1:
                                hours = df_temp.dt.hour.value_counts().sort_index()
                                fig.add_trace(go.Bar(x=hours.index.astype(str), y=hours.values,
                                                    marker_color='#ab47bc', name="Hour"), row=2, col=2)
                            
                            fig.update_layout(height=800, title_text=f"Temporal Analysis: {selected_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Date range information
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Start Date", df_temp.min().strftime('%Y-%m-%d'))
                            with col2:
                                st.metric("End Date", df_temp.max().strftime('%Y-%m-%d'))
                            with col3:
                                date_range = (df_temp.max() - df_temp.min()).days
                                st.metric("Date Range", f"{date_range} days")
                    else:
                        st.warning("⚠️ No datetime columns available for analysis")
            
            except Exception as e:
                st.error(f"❌ Error in univariate analysis: {str(e)}")
                st.info("💡 Tip: Ensure the selected column contains valid data for analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🔄 Bivariate Analysis")
            
            try:
                analysis_type = st.radio("Select analysis type", 
                                        ["Numeric vs Numeric", "Numeric vs Categorical", 
                                         "Categorical vs Categorical"],
                                        horizontal=True, key="bivariate_type")
                
                if analysis_type == "Numeric vs Numeric":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox("Select X axis", numeric_cols, key="bi_x")
                        with col2:
                            y_col = st.selectbox("Select Y axis", [c for c in numeric_cols if c != x_col], 
                                                key="bi_y")
                        
                        # Clean data for analysis
                        plot_df = df[[x_col, y_col]].dropna()
                        
                        if len(plot_df) > 0:
                            # Create comprehensive visualization
                            fig = make_subplots(rows=2, cols=3,
                                               subplot_titles=("Scatter Plot", "Hexbin Plot", "Density Contour",
                                                             "Marginal Distributions", "Residuals", "Statistics"),
                                               specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                                                     [{"type": "xy"}, {"type": "xy"}, {"type": "domain"}]])
                            
                            # Scatter plot with trendline
                            fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df[y_col],
                                                    mode='markers', name="Scatter",
                                                    marker=dict(size=5, opacity=0.6, color='#42a5f5')),
                                         row=1, col=1)
                            
                            # Add trendline
                            try:
                                z = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
                                p = np.poly1d(z)
                                x_range = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
                                fig.add_trace(go.Scatter(x=x_range, y=p(x_range),
                                                        mode='lines', name="Trend",
                                                        line=dict(color='red', width=2)), row=1, col=1)
                            except:
                                pass
                            
                            # Hexbin plot
                            fig.add_trace(go.Histogram2d(x=plot_df[x_col], y=plot_df[y_col],
                                                        colorscale='Viridis',
                                                        name="Hexbin"), row=1, col=2)
                            
                            # Density contour
                            fig.add_trace(go.Histogram2dContour(x=plot_df[x_col], y=plot_df[y_col],
                                                               colorscale='Viridis',
                                                               name="Contour"), row=1, col=3)
                            
                            # Marginal distributions
                            fig.add_trace(go.Histogram(x=plot_df[x_col], name=f"{x_col}",
                                                      marker_color='#66bb6a'), row=2, col=1)
                            fig.add_trace(go.Histogram(y=plot_df[y_col], name=f"{y_col}",
                                                      marker_color='#ffa726', orientation='h'), 
                                         row=2, col=1)
                            
                            # Residuals
                            try:
                                residuals = plot_df[y_col] - p(plot_df[x_col])
                                fig.add_trace(go.Scatter(x=plot_df[x_col], y=residuals,
                                                        mode='markers', name="Residuals",
                                                        marker=dict(size=3, opacity=0.5, color='#ab47bc')),
                                             row=2, col=2)
                                fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
                            except:
                                pass
                            
                            # Statistics
                            corr = plot_df[x_col].corr(plot_df[y_col])
                            stats_text = f"""
                            <b>Statistics</b><br>
                            Correlation: {corr:.4f}<br>
                            R²: {corr**2:.4f}<br>
                            Covariance: {plot_df[x_col].cov(plot_df[y_col]):.4f}<br>
                            Sample Size: {len(plot_df)}<br>
                            """
                            
                            fig.add_annotation(x=0.5, y=0.5, text=stats_text,
                                             showarrow=False, font=dict(size=10),
                                             row=2, col=3, align='left')
                            
                            fig.update_layout(height=800, title_text=f"Bivariate Analysis: {x_col} vs {y_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Correlation interpretation
                            if abs(corr) > 0.7:
                                st.success(f"✅ Strong {'positive' if corr > 0 else 'negative'} correlation detected")
                            elif abs(corr) > 0.3:
                                st.info(f"ℹ️ Moderate {'positive' if corr > 0 else 'negative'} correlation detected")
                            else:
                                st.warning(f"⚠️ Weak or no correlation detected")
                    else:
                        st.warning("⚠️ Need at least 2 numeric columns for this analysis")
                
                elif analysis_type == "Numeric vs Categorical":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if numeric_cols and categorical_cols:
                        col1, col2 = st.columns(2)
                        with col1:
                            num_col = st.selectbox("Select numeric column", numeric_cols, key="bi_num")
                        with col2:
                            cat_col = st.selectbox("Select categorical column", categorical_cols, key="bi_cat")
                        
                        # Clean data
                        plot_df = df[[num_col, cat_col]].dropna()
                        
                        if len(plot_df) > 0 and plot_df[cat_col].nunique() <= 30:
                            # Create visualizations
                            fig = make_subplots(rows=2, cols=2,
                                               subplot_titles=("Box Plot", "Violin Plot",
                                                              "Strip Plot", "Bar Chart (Means ± SD)"),
                                               specs=[[{"type": "xy"}, {"type": "xy"}],
                                                     [{"type": "xy"}, {"type": "xy"}]])
                            
                            # Box plot
                            fig.add_trace(go.Box(x=plot_df[cat_col], y=plot_df[num_col],
                                                name="Box Plot", marker_color='#42a5f5'), row=1, col=1)
                            
                            # Violin plot
                            fig.add_trace(go.Violin(x=plot_df[cat_col], y=plot_df[num_col],
                                                   box_visible=True, line_color='black',
                                                   fillcolor='#66bb6a', opacity=0.6,
                                                   name="Violin Plot"), row=1, col=2)
                            
                            # Strip plot
                            fig.add_trace(go.Scatter(x=plot_df[cat_col], y=plot_df[num_col],
                                                    mode='markers', name="Strip Plot",
                                                    marker=dict(size=3, opacity=0.3, color='#ffa726')),
                                         row=2, col=1)
                            
                            # Bar chart with error bars
                            stats_by_cat = plot_df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count']).reset_index()
                            stats_by_cat = stats_by_cat.sort_values('mean', ascending=False).head(15)
                            
                            fig.add_trace(go.Bar(x=stats_by_cat[cat_col], y=stats_by_cat['mean'],
                                                error_y=dict(type='data', array=stats_by_cat['std']),
                                                name="Mean ± SD", marker_color='#ab47bc'),
                                         row=2, col=2)
                            
                            fig.update_layout(height=800, title_text=f"{num_col} by {cat_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ANOVA test for groups with >2 categories
                            if plot_df[cat_col].nunique() >= 2:
                                groups = [group[num_col].values for name, group in plot_df.groupby(cat_col)]
                                if all(len(g) > 0 for g in groups):
                                    f_stat, p_val = stats.f_oneway(*groups)
                                    st.write(f"**One-way ANOVA Results:** F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")
                                    if p_val < 0.05:
                                        st.success("✅ Significant differences exist between groups")
                                    else:
                                        st.info("ℹ️ No significant differences found between groups")
                        elif plot_df[cat_col].nunique() > 30:
                            st.warning(f"⚠️ Categorical column has {plot_df[cat_col].nunique()} unique values. Consider grouping or selecting another column.")
                    else:
                        st.warning("⚠️ Need both numeric and categorical columns for this analysis")
                
                elif analysis_type == "Categorical vs Categorical":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if len(categorical_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            cat1 = st.selectbox("Select first categorical column", categorical_cols, key="bi_cat1")
                        with col2:
                            cat2 = st.selectbox("Select second categorical column", 
                                               [c for c in categorical_cols if c != cat1], key="bi_cat2")
                        
                        # Create contingency table
                        contingency = pd.crosstab(df[cat1], df[cat2])
                        
                        if contingency.size > 0:
                            fig = make_subplots(rows=1, cols=2,
                                               subplot_titles=("Stacked Bar Chart", "Heatmap"),
                                               specs=[[{"type": "xy"}, {"type": "heatmap"}]])
                            
                            # Stacked bar chart
                            for col in contingency.columns[:10]:  # Limit to 10 categories
                                fig.add_trace(go.Bar(x=contingency.index[:10], y=contingency[col][:10],
                                                    name=str(col)), row=1, col=1)
                            
                            # Heatmap
                            fig.add_trace(go.Heatmap(z=contingency.values[:10, :10],
                                                    x=contingency.columns[:10].astype(str),
                                                    y=contingency.index[:10].astype(str),
                                                    colorscale='Viridis',
                                                    text=contingency.values[:10, :10],
                                                    texttemplate="%{text}"), row=1, col=2)
                            
                            fig.update_layout(height=600, title_text=f"Relationship: {cat1} vs {cat2}",
                                            barmode='stack')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Chi-square test
                            from scipy.stats import chi2_contingency
                            chi2, p_val, dof, expected = chi2_contingency(contingency)
                            
                            st.write(f"**Chi-square Test Results:**")
                            st.write(f"χ² = {chi2:.4f}, df = {dof}, p-value = {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.success("✅ Significant association found between variables")
                                
                                # Cramer's V for effect size
                                n = contingency.sum().sum()
                                cramer_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                                st.write(f"**Cramer's V (effect size):** {cramer_v:.4f}")
                            else:
                                st.info("ℹ️ No significant association found")
                    else:
                        st.warning("⚠️ Need at least 2 categorical columns for this analysis")
            
            except Exception as e:
                st.error(f"❌ Error in bivariate analysis: {str(e)}")
                st.info("💡 Tip: Check if selected columns have sufficient data for analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📈 Multivariate Analysis")
            
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 3:
                    analysis_type = st.radio("Select analysis type",
                                            ["Correlation Matrix", "Parallel Coordinates", 
                                             "3D Scatter", "Radar Chart"],
                                            horizontal=True, key="multivariate_type")
                    
                    if analysis_type == "Correlation Matrix":
                        corr_matrix = df[numeric_cols].corr()
                        
                        fig = px.imshow(corr_matrix,
                                       text_auto=True,
                                       aspect="auto",
                                       color_continuous_scale='RdBu_r',
                                       title="Correlation Matrix Heatmap",
                                       zmin=-1, zmax=1)
                        
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find highly correlated pairs
                        high_corr = []
                        for i in range(len(numeric_cols)):
                            for j in range(i+1, len(numeric_cols)):
                                if abs(corr_matrix.iloc[i, j]) > 0.7:
                                    high_corr.append({
                                        'Feature 1': numeric_cols[i],
                                        'Feature 2': numeric_cols[j],
                                        'Correlation': corr_matrix.iloc[i, j]
                                    })
                        
                        if high_corr:
                            st.subheader("🔍 Highly Correlated Pairs (|r| > 0.7)")
                            for item in high_corr:
                                st.write(f"• **{item['Feature 1']}** & **{item['Feature 2']}**: {item['Correlation']:.4f}")
                    
                    elif analysis_type == "Parallel Coordinates":
                        # Select dimensions
                        selected_dims = st.multiselect("Select dimensions (columns)", 
                                                      numeric_cols, 
                                                      default=numeric_cols[:min(4, len(numeric_cols))])
                        
                        if len(selected_dims) >= 2:
                            # Optional color dimension
                            color_dim = st.selectbox("Color by", ["None"] + numeric_cols + 
                                                    df.select_dtypes(include=['object', 'category']).columns.tolist())
                            
                            plot_df = df[selected_dims].dropna()
                            
                            if len(plot_df) > 0:
                                if color_dim == "None":
                                    fig = px.parallel_coordinates(plot_df, 
                                                                 dimensions=selected_dims,
                                                                 title="Parallel Coordinates Plot")
                                else:
                                    if color_dim in numeric_cols:
                                        fig = px.parallel_coordinates(plot_df, 
                                                                     dimensions=selected_dims,
                                                                     color=color_dim,
                                                                     color_continuous_scale=px.colors.diverging.RdBu,
                                                                     title=f"Parallel Coordinates colored by {color_dim}")
                                    else:
                                        # Categorical color
                                        temp_df = df[selected_dims + [color_dim]].dropna()
                                        fig = px.parallel_coordinates(temp_df, 
                                                                     dimensions=selected_dims,
                                                                     color=color_dim,
                                                                     title=f"Parallel Coordinates colored by {color_dim}")
                                
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    elif analysis_type == "3D Scatter":
                        if len(numeric_cols) >= 3:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                x_3d = st.selectbox("X axis", numeric_cols, key="3d_x")
                            with col2:
                                y_3d = st.selectbox("Y axis", [c for c in numeric_cols if c != x_3d], key="3d_y")
                            with col3:
                                z_3d = st.selectbox("Z axis", [c for c in numeric_cols if c not in [x_3d, y_3d]], 
                                                   key="3d_z")
                            
                            color_3d = st.selectbox("Color by", ["None"] + 
                                                   df.select_dtypes(include=['object', 'category']).columns.tolist())
                            
                            plot_df = df[[x_3d, y_3d, z_3d]].dropna()
                            
                            if len(plot_df) > 0:
                                if color_3d == "None":
                                    fig = px.scatter_3d(plot_df, x=x_3d, y=y_3d, z=z_3d,
                                                      title=f"3D Scatter Plot",
                                                      opacity=0.7)
                                else:
                                    temp_df = df[[x_3d, y_3d, z_3d, color_3d]].dropna()
                                    fig = px.scatter_3d(temp_df, x=x_3d, y=y_3d, z=z_3d,
                                                      color=color_3d,
                                                      title=f"3D Scatter colored by {color_3d}",
                                                      opacity=0.7)
                                
                                fig.update_layout(height=700)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    elif analysis_type == "Radar Chart":
                        # Select features for radar
                        radar_features = st.multiselect("Select features for radar chart",
                                                        numeric_cols,
                                                        default=numeric_cols[:min(5, len(numeric_cols))])
                        
                        if len(radar_features) >= 3:
                            # Select how many samples to show
                            n_samples = st.slider("Number of samples to show", 1, min(10, len(df)), 3)
                            
                            fig = go.Figure()
                            
                            for i in range(n_samples):
                                sample = df.iloc[i][radar_features].values
                                fig.add_trace(go.Scatterpolar(
                                    r=sample,
                                    theta=radar_features,
                                    fill='toself',
                                    name=f'Sample {i}'
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[df[radar_features].min().min(), df[radar_features].max().max()]
                                    )),
                                title=f"Radar Chart - First {n_samples} Samples",
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Need at least 3 numeric columns for multivariate analysis")
            
            except Exception as e:
                st.error(f"❌ Error in multivariate analysis: {str(e)}")
                st.info("💡 Tip: Ensure you have enough numeric columns for multivariate analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab6:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🎯 Pattern Discovery")
            
            try:
                analysis_type = st.radio("Select pattern discovery method",
                                        ["Clustering Visualization", "Outlier Detection", 
                                         "Trend Detection", "Seasonal Patterns"],
                                        horizontal=True, key="pattern_type")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if analysis_type == "Clustering Visualization":
                    if len(numeric_cols) >= 2:
                        from sklearn.cluster import KMeans
                        from sklearn.preprocessing import StandardScaler
                        
                        # Select features for clustering
                        cluster_features = st.multiselect("Select features for clustering",
                                                          numeric_cols,
                                                          default=numeric_cols[:min(3, len(numeric_cols))])
                        
                        if len(cluster_features) >= 2:
                            n_clusters = st.slider("Number of clusters", 2, 8, 3)
                            
                            # Prepare data
                            X = df[cluster_features].dropna()
                            
                            if len(X) > 0:
                                # Scale data
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Perform clustering
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                clusters = kmeans.fit_predict(X_scaled)
                                
                                # Create visualization
                                if len(cluster_features) == 2:
                                    fig = px.scatter(x=X[cluster_features[0]], y=X[cluster_features[1]],
                                                   color=clusters.astype(str),
                                                   title=f"K-Means Clustering (k={n_clusters})",
                                                   labels={'x': cluster_features[0], 'y': cluster_features[1],
                                                          'color': 'Cluster'})
                                elif len(cluster_features) >= 3:
                                    fig = px.scatter_3d(x=X[cluster_features[0]], y=X[cluster_features[1]],
                                                       z=X[cluster_features[2]], color=clusters.astype(str),
                                                       title=f"K-Means Clustering (k={n_clusters})",
                                                       labels={cluster_features[0]: cluster_features[0],
                                                              cluster_features[1]: cluster_features[1],
                                                              cluster_features[2]: cluster_features[2],
                                                              'color': 'Cluster'})
                                
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Cluster statistics
                                st.subheader("📊 Cluster Statistics")
                                X['Cluster'] = clusters
                                cluster_stats = X.groupby('Cluster')[cluster_features].mean()
                                st.dataframe(cluster_stats.style.format("{:.4f}"))
                
                elif analysis_type == "Outlier Detection":
                    if len(numeric_cols) >= 2:
                        from sklearn.ensemble import IsolationForest
                        
                        # Select features for outlier detection
                        outlier_features = st.multiselect("Select features for outlier detection",
                                                          numeric_cols,
                                                          default=numeric_cols[:min(3, len(numeric_cols))])
                        
                        if len(outlier_features) >= 2:
                            contamination = st.slider("Expected outlier proportion", 0.01, 0.5, 0.1, 0.01)
                            
                            # Prepare data
                            X = df[outlier_features].dropna()
                            
                            if len(X) > 0:
                                # Detect outliers
                                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                                outliers = iso_forest.fit_predict(X)
                                
                                # Create visualization
                                if len(outlier_features) == 2:
                                    fig = px.scatter(x=X[outlier_features[0]], y=X[outlier_features[1]],
                                                   color=outliers,
                                                   color_continuous_scale=['blue', 'red'],
                                                   title=f"Outlier Detection (contamination={contamination})",
                                                   labels={'x': outlier_features[0], 'y': outlier_features[1],
                                                          'color': 'Outlier'})
                                elif len(outlier_features) >= 3:
                                    fig = px.scatter_3d(x=X[outlier_features[0]], y=X[outlier_features[1]],
                                                       z=X[outlier_features[2]], color=outliers,
                                                       color_continuous_scale=['blue', 'red'],
                                                       title=f"Outlier Detection (contamination={contamination})",
                                                       labels={outlier_features[0]: outlier_features[0],
                                                              outlier_features[1]: outlier_features[1],
                                                              outlier_features[2]: outlier_features[2],
                                                              'color': 'Outlier'})
                                
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Outlier statistics
                                n_outliers = (outliers == -1).sum()
                                st.write(f"**Outliers detected:** {n_outliers} ({n_outliers/len(X)*100:.2f}%)")
                
                elif analysis_type == "Trend Detection":
                    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                    
                    if datetime_cols and numeric_cols:
                        date_col = st.selectbox("Select date column", datetime_cols)
                        value_col = st.selectbox("Select value column", numeric_cols)
                        
                        # Prepare time series data
                        ts_df = df[[date_col, value_col]].dropna().sort_values(date_col)
                        
                        if len(ts_df) > 10:
                            # Calculate moving averages
                            window = st.slider("Moving average window", 2, 30, 7)
                            ts_df['MA'] = ts_df[value_col].rolling(window=window).mean()
                            
                            # Detect trend using linear regression
                            from sklearn.linear_model import LinearRegression
                            
                            X = np.arange(len(ts_df)).reshape(-1, 1)
                            y = ts_df[value_col].values
                            
                            model = LinearRegression()
                            model.fit(X, y)
                            trend = model.predict(X)
                            
                            # Create visualization
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=ts_df[date_col], y=ts_df[value_col],
                                                    mode='lines', name='Original'))
                            fig.add_trace(go.Scatter(x=ts_df[date_col], y=ts_df['MA'],
                                                    mode='lines', name=f'{window}-period MA',
                                                    line=dict(color='orange')))
                            fig.add_trace(go.Scatter(x=ts_df[date_col], y=trend,
                                                    mode='lines', name='Linear Trend',
                                                    line=dict(color='red', dash='dash')))
                            
                            fig.update_layout(title="Trend Detection",
                                            xaxis_title="Date",
                                            yaxis_title=value_col,
                                            height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Trend statistics
                            slope = model.coef_[0]
                            st.write(f"**Trend slope:** {slope:.4f} units per time step")
                            if slope > 0:
                                st.success("✅ Upward trend detected")
                            elif slope < 0:
                                st.warning("⚠️ Downward trend detected")
                            else:
                                st.info("ℹ️ No clear trend detected")
                
                elif analysis_type == "Seasonal Patterns":
                    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                    
                    if datetime_cols and numeric_cols:
                        date_col = st.selectbox("Select date column", datetime_cols, key="seasonal_date")
                        value_col = st.selectbox("Select value column", numeric_cols, key="seasonal_value")
                        
                        # Extract seasonal components
                        df_temp = df[[date_col, value_col]].dropna()
                        df_temp['year'] = pd.DatetimeIndex(df_temp[date_col]).year
                        df_temp['month'] = pd.DatetimeIndex(df_temp[date_col]).month
                        df_temp['quarter'] = pd.DatetimeIndex(df_temp[date_col]).quarter
                        df_temp['dayofweek'] = pd.DatetimeIndex(df_temp[date_col]).dayofweek
                        
                        # Create seasonal visualizations
                        fig = make_subplots(rows=2, cols=2,
                                           subplot_titles=("Year-over-Year", "Monthly Pattern",
                                                          "Quarterly Pattern", "Day of Week Pattern"),
                                           specs=[[{"type": "xy"}, {"type": "xy"}],
                                                 [{"type": "xy"}, {"type": "xy"}]])
                        
                        # Year-over-Year
                        yearly_avg = df_temp.groupby('year')[value_col].mean().reset_index()
                        fig.add_trace(go.Scatter(x=yearly_avg['year'], y=yearly_avg[value_col],
                                                mode='lines+markers', name="Yearly Avg"), row=1, col=1)
                        
                        # Monthly pattern
                        monthly_avg = df_temp.groupby('month')[value_col].mean().reset_index()
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        fig.add_trace(go.Bar(x=[month_names[m-1] for m in monthly_avg['month']],
                                            y=monthly_avg[value_col], name="Monthly Avg"), row=1, col=2)
                        
                        # Quarterly pattern
                        quarterly_avg = df_temp.groupby('quarter')[value_col].mean().reset_index()
                        quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
                        fig.add_trace(go.Bar(x=[quarter_names[q-1] for q in quarterly_avg['quarter']],
                                            y=quarterly_avg[value_col], name="Quarterly Avg"), row=2, col=1)
                        
                        # Day of week pattern
                        dow_avg = df_temp.groupby('dayofweek')[value_col].mean().reset_index()
                        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        fig.add_trace(go.Bar(x=[day_names[d] for d in dow_avg['dayofweek']],
                                            y=dow_avg[value_col], name="Day of Week Avg"), row=2, col=2)
                        
                        fig.update_layout(height=800, title_text="Seasonal Pattern Analysis")
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error in pattern discovery: {str(e)}")
                st.info("💡 Tip: Ensure you have sufficient data for pattern detection")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Critical error in EDA: {str(e)}")
        st.info("💡 Please check your dataset and try again")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export EDA Report")
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        report_text = f"""
        EXPLORATORY DATA ANALYSIS REPORT
        =================================
        
        Dataset Information:
        • Total Rows: {df.shape[0]:,}
        • Total Columns: {df.shape[1]}
        • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        
        Column Types:
        • Numeric: {len(numeric_cols)}
        • Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}
        • Datetime: {len(df.select_dtypes(include=['datetime64']).columns)}
        
        Data Quality:
        • Missing Values: {df.isnull().sum().sum():,}
        • Complete Cases: {df.dropna().shape[0]:,}
        • Duplicate Rows: {df.duplicated().sum():,}
        
        Analysis Performed:
        • Data Overview
        • Missing Data Analysis
        • Univariate Analysis
        • Bivariate Analysis
        • Multivariate Analysis
        • Pattern Discovery
        """
        
        st.download_button(
            label="📥 Download EDA Report",
            data=report_text,
            file_name="eda_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")