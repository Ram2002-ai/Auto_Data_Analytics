import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def auto_visualizations(df):
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>📊 Interactive Data Visualization</h2>
        <p style='color: gray;'>Create beautiful, interactive visualizations with just a few clicks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Visualization type selector
    viz_type = st.selectbox(
        "🎨 Select Visualization Type",
        ["Distribution Plots", "Categorical Plots", "Relationship Plots", 
         "Time Series Plots", "Statistical Plots", "Advanced Plots"]
    )
    
    if viz_type == "Distribution Plots":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📈 Distribution Plots")
        
        if num_cols:
            # Create tabs for different distribution plots
            dist_tab1, dist_tab2, dist_tab3 = st.tabs(["Histogram", "Box Plot", "Violin Plot"])
            
            with dist_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    hist_col = st.selectbox("Select column", num_cols, key="hist")
                with col2:
                    bins = st.slider("Number of bins", 5, 100, 30)
                
                fig = px.histogram(df, x=hist_col, nbins=bins, 
                                  title=f"Distribution of {hist_col}",
                                  marginal="box", opacity=0.7)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with dist_tab2:
                if cat_cols:
                    box_col = st.selectbox("Numeric column", num_cols, key="box_num")
                    box_cat = st.selectbox("Category column (optional)", ["None"] + cat_cols, key="box_cat")
                    
                    if box_cat == "None":
                        fig = px.box(df, y=box_col, title=f"Box Plot of {box_col}")
                    else:
                        fig = px.box(df, x=box_cat, y=box_col, title=f"{box_col} by {box_cat}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Add categorical columns to create grouped box plots")
            
            with dist_tab3:
                if cat_cols:
                    violin_col = st.selectbox("Numeric column", num_cols, key="violin_num")
                    violin_cat = st.selectbox("Category column", cat_cols, key="violin_cat")
                    
                    fig = px.violin(df, x=violin_cat, y=violin_col, 
                                   box=True, points="all",
                                   title=f"Violin Plot of {violin_col} by {violin_cat}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available for distribution plots")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_type == "Categorical Plots":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Categorical Plots")
        
        if cat_cols:
            # Create tabs for categorical plots
            cat_tab1, cat_tab2, cat_tab3 = st.tabs(["Bar Chart", "Pie Chart", "Sunburst Chart"])
            
            with cat_tab1:
                bar_col = st.selectbox("Select categorical column", cat_cols, key="bar")
                
                # Get value counts
                value_counts = df[bar_col].value_counts().reset_index()
                value_counts.columns = [bar_col, 'count']
                
                # Color option
                if num_cols:
                    color_by = st.selectbox("Color by (optional)", ["None"] + num_cols, key="bar_color")
                else:
                    color_by = "None"
                
                if color_by == "None":
                    fig = px.bar(value_counts, x=bar_col, y='count',
                               title=f"Distribution of {bar_col}",
                               color_discrete_sequence=['#636EFA'])
                else:
                    # Aggregate numeric column by category
                    agg_data = df.groupby(bar_col)[color_by].mean().reset_index()
                    fig = px.bar(agg_data, x=bar_col, y=color_by,
                               title=f"Average {color_by} by {bar_col}",
                               color=bar_col)
                
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with cat_tab2:
                pie_col = st.selectbox("Select column for pie chart", cat_cols, key="pie")
                
                # Limit to top 10 categories for readability
                top_n = st.slider("Show top N categories", 3, 20, 10)
                value_counts = df[pie_col].value_counts().head(top_n)
                
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Proportion of {pie_col} (Top {top_n})",
                           hole=0.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with cat_tab3:
                if len(cat_cols) >= 2:
                    st.markdown("**Hierarchical View**")
                    path = st.multiselect("Select hierarchy (order matters)", 
                                        cat_cols, default=cat_cols[:2])
                    
                    if len(path) >= 2:
                        fig = px.sunburst(df, path=path, 
                                        title="Hierarchical Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 categorical columns for sunburst chart")
        else:
            st.warning("No categorical columns available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_type == "Relationship Plots":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔄 Relationship Plots")
        
        if len(num_cols) >= 2:
            rel_tab1, rel_tab2, rel_tab3 = st.tabs(["Scatter Plot", "Line Plot", "Heatmap"])
            
            with rel_tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X axis", num_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y axis", [c for c in num_cols if c != x_col], key="scatter_y")
                with col3:
                    color_col = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="scatter_color")
                
                size_col = st.selectbox("Size by (optional)", ["None"] + num_cols, key="scatter_size")
                
                # Create scatter plot
                if color_col == "None" and size_col == "None":
                    fig = px.scatter(df, x=x_col, y=y_col, 
                                   title=f"{y_col} vs {x_col}",
                                   trendline="ols")
                elif color_col != "None" and size_col == "None":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                   title=f"{y_col} vs {x_col} colored by {color_col}",
                                   trendline="ols")
                elif color_col == "None" and size_col != "None":
                    fig = px.scatter(df, x=x_col, y=y_col, size=size_col,
                                   title=f"{y_col} vs {x_col} sized by {size_col}",
                                   trendline="ols")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                                   title=f"{y_col} vs {x_col}",
                                   trendline="ols")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with rel_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    line_x = st.selectbox("X axis (usually time)", num_cols + date_cols, key="line_x")
                with col2:
                    line_y = st.selectbox("Y axis", num_cols, key="line_y")
                
                line_color = st.selectbox("Color by", ["None"] + cat_cols, key="line_color")
                
                if line_color == "None":
                    fig = px.line(df, x=line_x, y=line_y, 
                                title=f"{line_y} over {line_x}")
                else:
                    fig = px.line(df, x=line_x, y=line_y, color=line_color,
                                title=f"{line_y} over {line_x} by {line_color}")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with rel_tab3:
                # Correlation heatmap
                corr_matrix = df[num_cols].corr()
                
                # Mask for upper triangle
                mask = np.triu(np.ones_like(corr_matrix), k=1)
                masked_corr = corr_matrix * (1 - mask)
                
                fig = px.imshow(masked_corr,
                              text_auto=True,
                              aspect="auto",
                              color_continuous_scale='RdBu_r',
                              title="Correlation Heatmap",
                              zmin=-1, zmax=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strongest correlations
                st.markdown("**Strongest Correlations:**")
                corr_pairs = []
                for i in range(len(num_cols)):
                    for j in range(i+1, len(num_cols)):
                        corr_pairs.append((num_cols[i], num_cols[j], 
                                         corr_matrix.iloc[i, j]))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for col1, col2, corr in corr_pairs[:5]:
                    strength = "🟢" if abs(corr) > 0.7 else "🟡" if abs(corr) > 0.3 else "🔴"
                    st.write(f"{strength} **{col1}** & **{col2}**: {corr:.3f}")
        else:
            st.warning("Need at least 2 numeric columns for relationship plots")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_type == "Time Series Plots":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📅 Time Series Plots")
        
        if date_cols:
            ts_tab1, ts_tab2 = st.tabs(["Time Series", "Resampling"])
            
            with ts_tab1:
                date_col = st.selectbox("Date column", date_cols, key="ts_date")
                value_col = st.selectbox("Value column", num_cols if num_cols else [], key="ts_value")
                
                if num_cols and date_col:
                    # Sort by date
                    df_sorted = df.sort_values(date_col)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_sorted[date_col], y=df_sorted[value_col],
                                            mode='lines+markers', name=value_col))
                    
                    fig.update_layout(title=f"{value_col} over Time",
                                    xaxis_title="Date",
                                    yaxis_title=value_col)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with ts_tab2:
                if num_cols and date_cols:
                    date_col = st.selectbox("Select date column", date_cols, key="resample_date")
                    resample_col = st.selectbox("Select column to resample", num_cols, key="resample_col")
                    
                    freq = st.selectbox("Resampling frequency",
                                      ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
                    
                    freq_map = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "M",
                        "Quarterly": "Q",
                        "Yearly": "Y"
                    }
                    
                    # Set date as index
                    df_date = df.set_index(date_col)
                    
                    # Resample
                    resampled = df_date[resample_col].resample(freq_map[freq]).mean().reset_index()
                    
                    fig = px.line(resampled, x=date_col, y=resample_col,
                                title=f"{resample_col} ({freq} Aggregated)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No datetime columns found. Convert a column to datetime first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_type == "Statistical Plots":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📐 Statistical Plots")
        
        if num_cols:
            stat_tab1, stat_tab2, stat_tab3 = st.tabs(["QQ Plot", "ECDF", "Density Heatmap"])
            
            with stat_tab1:
                qq_col = st.selectbox("Select column for QQ plot", num_cols, key="qq")
                
                # Calculate quantiles
                data = df[qq_col].dropna()
                theoretical_quantiles = np.percentile(np.random.normal(0, 1, len(data)), 
                                                     np.linspace(0, 100, len(data)))
                sample_quantiles = np.percentile(data, np.linspace(0, 100, len(data)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                                        mode='markers', name='Data'))
                
                # Add diagonal line
                min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Normal',
                                        line=dict(color='red', dash='dash')))
                
                fig.update_layout(title=f"QQ Plot - {qq_col}",
                                xaxis_title="Theoretical Quantiles",
                                yaxis_title="Sample Quantiles")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with stat_tab2:
                ecdf_col = st.selectbox("Select column for ECDF", num_cols, key="ecdf")
                
                fig = px.ecdf(df, x=ecdf_col, 
                            title=f"Empirical Cumulative Distribution - {ecdf_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with stat_tab3:
                if len(num_cols) >= 2:
                    x_col = st.selectbox("X axis", num_cols, key="density_x")
                    y_col = st.selectbox("Y axis", [c for c in num_cols if c != x_col], key="density_y")
                    
                    fig = px.density_heatmap(df, x=x_col, y=y_col,
                                           title=f"Density Heatmap: {y_col} vs {x_col}",
                                           marginal_x="histogram",
                                           marginal_y="histogram")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available for statistical plots")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_type == "Advanced Plots":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🚀 Advanced Visualizations")
        
        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["3D Scatter", "Parallel Coordinates", "Radar Chart"])
        
        with adv_tab1:
            if len(num_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_3d = st.selectbox("X axis", num_cols, key="3d_x")
                with col2:
                    y_3d = st.selectbox("Y axis", [c for c in num_cols if c != x_3d], key="3d_y")
                with col3:
                    z_3d = st.selectbox("Z axis", [c for c in num_cols if c not in [x_3d, y_3d]], key="3d_z")
                
                color_3d = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="3d_color")
                
                if color_3d == "None":
                    fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d,
                                      title=f"3D Scatter: {x_3d}, {y_3d}, {z_3d}")
                else:
                    fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d,
                                      title=f"3D Scatter colored by {color_3d}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 3 numeric columns for 3D scatter plot")
        
        with adv_tab2:
            if num_cols:
                selected_dims = st.multiselect("Select dimensions", num_cols, default=num_cols[:4])
                
                if selected_dims and len(selected_dims) >= 2:
                    color_dim = st.selectbox("Color dimension", ["None"] + cat_cols + num_cols)
                    
                    if color_dim == "None":
                        fig = px.parallel_coordinates(df, dimensions=selected_dims,
                                                    title="Parallel Coordinates Plot")
                    else:
                        fig = px.parallel_coordinates(df, dimensions=selected_dims,
                                                    color=color_dim,
                                                    title=f"Parallel Coordinates colored by {color_dim}")
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with adv_tab3:
            if num_cols:
                st.markdown("**Radar Chart** (requires at least 3 numeric columns)")
                selected_radar = st.multiselect("Select metrics for radar chart", 
                                              num_cols, default=num_cols[:3])
                
                if len(selected_radar) >= 3:
                    # Get first row as sample
                    sample = df[selected_radar].iloc[0]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=sample.values,
                        theta=selected_radar,
                        fill='toself'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[sample.min(), sample.max()]
                            )),
                        showlegend=False,
                        title="Radar Chart (First Row)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download plot data option
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("To save any plot, hover over it and click the camera icon 📷")
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="visualization_data.csv",
            mime="text/csv",
            use_container_width=True
        )