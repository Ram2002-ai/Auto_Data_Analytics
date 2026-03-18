import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def statistical_analysis(df):
    """
    Enhanced statistical analysis with advanced statistical tests and visualizations
    """
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>📐 Advanced Statistical Analysis</h2>
        <p style='color: gray;'>Comprehensive statistical tests, hypothesis testing, and probability analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Error handling for empty dataframe
    if df.empty:
        st.error("❌ The dataset is empty. Please upload a valid dataset.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ No numeric columns found. Statistical analysis requires numeric data.")
        return
    
    # Create tabs for different statistical analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Descriptive Stats", 
        "📈 Correlation Analysis", 
        "🔬 Hypothesis Testing",
        "📊 Distribution Analysis", 
        "📉 Time Series Analysis",
        "🎲 Probability & Sampling"
    ])
    
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Descriptive Statistics")
        
        try:
            # Basic statistics with confidence intervals
            stats_df = pd.DataFrame()
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    # Calculate confidence interval
                    ci = stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))
                    
                    stats_df[col] = {
                        'Count': len(data),
                        'Mean': data.mean(),
                        'Std Dev': data.std(),
                        'Variance': data.var(),
                        'Min': data.min(),
                        'Q1 (25%)': data.quantile(0.25),
                        'Median (50%)': data.median(),
                        'Q3 (75%)': data.quantile(0.75),
                        'Max': data.max(),
                        'Range': data.max() - data.min(),
                        'IQR': data.quantile(0.75) - data.quantile(0.25),
                        'Skewness': data.skew(),
                        'Kurtosis': data.kurtosis(),
                        'Coefficient of Variation (%)': (data.std() / data.mean() * 100) if data.mean() != 0 else np.nan,
                        '95% CI Lower': ci[0],
                        '95% CI Upper': ci[1]
                    }
            
            stats_df = pd.DataFrame(stats_df).T
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
            
            # Summary cards
            st.subheader("📊 Summary Cards")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Numeric Columns", len(numeric_cols))
            with col2:
                st.metric("Total Observations", f"{df.shape[0]:,}")
            with col3:
                st.metric("Complete Cases", f"{df.dropna().shape[0]:,}")
            with col4:
                completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Data Completeness", f"{completeness:.1f}%")
            
            # Distribution visualization
            st.subheader("Distribution Analysis")
            selected_col = st.selectbox("Select column for detailed distribution analysis", numeric_cols)
            
            data = df[selected_col].dropna()
            
            fig = make_subplots(rows=2, cols=2,
                               subplot_titles=("Histogram with KDE", "Box Plot", 
                                             "Violin Plot", "Q-Q Plot"),
                               specs=[[{"type": "xy"}, {"type": "xy"}],
                                     [{"type": "xy"}, {"type": "xy"}]])
            
            # Histogram with KDE
            hist_data = go.Histogram(x=data, nbinsx=30, name="Histogram", opacity=0.7)
            fig.add_trace(hist_data, row=1, col=1)
            
            # Box plot
            box_data = go.Box(y=data, name="Box Plot", boxpoints='outliers')
            fig.add_trace(box_data, row=1, col=2)
            
            # Violin plot
            violin_data = go.Violin(y=data, name="Violin Plot", box_visible=True, meanline_visible=True)
            fig.add_trace(violin_data, row=2, col=1)
            
            # Q-Q plot
            theoretical_q = np.random.normal(data.mean(), data.std(), len(data))
            theoretical_q.sort()
            data_sorted = np.sort(data)
            qq_data = go.Scatter(x=theoretical_q, y=data_sorted, mode='markers', name='Q-Q')
            fig.add_trace(qq_data, row=2, col=2)
            
            # Add reference line to Q-Q plot
            min_val = min(theoretical_q.min(), data_sorted.min())
            max_val = max(theoretical_q.max(), data_sorted.max())
            ref_line = go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', name='Reference', line=dict(color='red', dash='dash'))
            fig.add_trace(ref_line, row=2, col=2)
            
            fig.update_layout(height=800, title_text=f"Distribution Analysis of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Outlier detection
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                st.warning(f"⚠️ **Outliers detected**: {len(outliers)} outliers found ({len(outliers)/len(data)*100:.2f}%)")
                with st.expander("View outlier values"):
                    st.write(outliers.tolist())
            else:
                st.success("✅ No outliers detected in this column")
        
        except Exception as e:
            st.error(f"❌ Error in descriptive statistics: {str(e)}")
            st.info("💡 Tip: Check if your data contains non-numeric values or extreme outliers")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📈 Advanced Correlation Analysis")
        
        try:
            if len(numeric_cols) >= 2:
                # Multiple correlation methods
                corr_method = st.radio(
                    "Select correlation method",
                    ["Pearson (linear)", "Spearman (rank)", "Kendall (ordinal)"],
                    horizontal=True
                )
                
                method_map = {
                    "Pearson (linear)": "pearson",
                    "Spearman (rank)": "spearman",
                    "Kendall (ordinal)": "kendall"
                }
                
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr(method=method_map[corr_method])
                
                # Heatmap with improved visualization
                fig = px.imshow(corr_matrix,
                               text_auto=True,
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title=f"{corr_method} Correlation Matrix",
                               zmin=-1, zmax=1)
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation significance testing
                st.subheader("📊 Correlation Significance Testing")
                
                col1, col2 = st.columns(2)
                with col1:
                    feat1 = st.selectbox("Select first feature", numeric_cols, key="corr_feat1")
                with col2:
                    feat2 = st.selectbox("Select second feature", [c for c in numeric_cols if c != feat1], key="corr_feat2")
                
                data1 = df[feat1].dropna()
                data2 = df[feat2].dropna()
                
                # Align data
                combined = pd.concat([data1, data2], axis=1).dropna()
                if len(combined) > 0:
                    corr_coef, p_value = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
                    
                    st.write(f"**Pearson correlation coefficient:** {corr_coef:.4f}")
                    st.write(f"**P-value:** {p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.success(f"✅ Statistically significant correlation (p < 0.05)")
                    else:
                        st.info(f"ℹ️ No statistically significant correlation (p >= 0.05)")
                    
                    # Confidence interval for correlation
                    n = len(combined)
                    r = corr_coef
                    z = np.arctanh(r)
                    se = 1 / np.sqrt(n - 3)
                    ci_z = stats.norm.interval(0.95, loc=z, scale=se)
                    ci_r = np.tanh(ci_z)
                    
                    st.write(f"**95% Confidence Interval:** [{ci_r[0]:.4f}, {ci_r[1]:.4f}]")
                    
                    # Scatter plot with regression line
                    fig = px.scatter(combined, x=combined.columns[0], y=combined.columns[1],
                                   trendline="ols", title=f"Relationship: {feat1} vs {feat2}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Partial correlation analysis
                st.subheader("🔍 Partial Correlation Analysis")
                if len(numeric_cols) >= 3:
                    from sklearn.linear_model import LinearRegression
                    
                    control_var = st.selectbox("Select control variable", 
                                              [c for c in numeric_cols if c not in [feat1, feat2]])
                    
                    # Calculate partial correlation
                    X_control = df[[control_var]].dropna()
                    y1 = df[feat1].dropna()
                    y2 = df[feat2].dropna()
                    
                    # Align data
                    aligned_data = pd.concat([X_control, y1, y2], axis=1).dropna()
                    
                    if len(aligned_data) > 0:
                        # Residualize
                        model1 = LinearRegression().fit(aligned_data[[control_var]], aligned_data[feat1])
                        res1 = aligned_data[feat1] - model1.predict(aligned_data[[control_var]])
                        
                        model2 = LinearRegression().fit(aligned_data[[control_var]], aligned_data[feat2])
                        res2 = aligned_data[feat2] - model2.predict(aligned_data[[control_var]])
                        
                        partial_corr, partial_p = stats.pearsonr(res1, res2)
                        
                        st.write(f"**Partial correlation (controlling for {control_var}):** {partial_corr:.4f}")
                        st.write(f"**P-value:** {partial_p:.4f}")
                        
                        if abs(partial_corr) < abs(corr_coef):
                            st.info(f"ℹ️ The correlation decreases when controlling for {control_var}, suggesting it may be a confounding variable")
            else:
                st.warning("⚠️ Need at least 2 numeric columns for correlation analysis")
        
        except Exception as e:
            st.error(f"❌ Error in correlation analysis: {str(e)}")
            st.info("💡 Tip: Ensure your data has sufficient non-null values for correlation calculation")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔬 Statistical Hypothesis Testing")
        
        try:
            test_category = st.selectbox(
                "Select test category",
                ["Parametric Tests", "Non-parametric Tests", "ANOVA & Post-hoc", "Goodness of Fit"]
            )
            
            if test_category == "Parametric Tests":
                param_test = st.selectbox(
                    "Select parametric test",
                    ["One-Sample t-test", "Independent t-test", "Paired t-test", "Z-test"]
                )
                
                if param_test == "One-Sample t-test":
                    if numeric_cols:
                        col = st.selectbox("Select variable", numeric_cols)
                        test_value = st.number_input("Test value (population mean)", value=0.0)
                        
                        data = df[col].dropna()
                        if len(data) > 0:
                            t_stat, p_value = stats.ttest_1samp(data, test_value)
                            
                            st.write(f"**t-statistic:** {t_stat:.4f}")
                            st.write(f"**p-value:** {p_value:.4f}")
                            st.write(f"**Degrees of freedom:** {len(data)-1}")
                            
                            # Effect size (Cohen's d)
                            cohens_d = (data.mean() - test_value) / data.std()
                            st.write(f"**Cohen's d (effect size):** {cohens_d:.4f}")
                            
                            if p_value < 0.05:
                                st.success(f"✅ Reject null hypothesis: Mean is significantly different from {test_value}")
                            else:
                                st.info(f"ℹ️ Fail to reject null hypothesis: Mean is not significantly different from {test_value}")
                            
                            # Visualization
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=data, name="Sample", opacity=0.7))
                            fig.add_vline(x=test_value, line_dash="dash", line_color="red",
                                        annotation_text=f"Test value: {test_value}")
                            fig.add_vline(x=data.mean(), line_color="green",
                                        annotation_text=f"Sample mean: {data.mean():.2f}")
                            fig.update_layout(title=f"One-Sample t-test: {col}")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif param_test == "Independent t-test":
                    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                        num_col = st.selectbox("Select numeric variable", numeric_cols, key="ind_num")
                        cat_col = st.selectbox("Select grouping variable", categorical_cols, key="ind_cat")
                        
                        groups = df[cat_col].dropna().unique()
                        if len(groups) == 2:
                            group1 = df[df[cat_col] == groups[0]][num_col].dropna()
                            group2 = df[df[cat_col] == groups[1]][num_col].dropna()
                            
                            # Test for equal variances
                            levene_stat, levene_p = stats.levene(group1, group2)
                            equal_var = levene_p > 0.05
                            
                            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
                            
                            st.write(f"**Groups:** {groups[0]} (n={len(group1)}) vs {groups[1]} (n={len(group2)})")
                            st.write(f"**Levene's test for equal variances:** p={levene_p:.4f}")
                            st.write(f"**Assuming {'equal' if equal_var else 'unequal'} variances")
                            st.write(f"**t-statistic:** {t_stat:.4f}")
                            st.write(f"**p-value:** {p_value:.4f}")
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / 
                                                (len(group1)+len(group2)-2))
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std
                            st.write(f"**Cohen's d (effect size):** {cohens_d:.4f}")
                            
                            if p_value < 0.05:
                                st.success(f"✅ Significant difference found between groups")
                            else:
                                st.info(f"ℹ️ No significant difference found between groups")
                            
                            # Visualization
                            fig = px.box(df, x=cat_col, y=num_col, title=f"Comparison: {num_col} by {cat_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"⚠️ Independent t-test requires exactly 2 groups. Found {len(groups)} groups.")
                
                elif param_test == "Paired t-test":
                    if len(numeric_cols) >= 2:
                        col1 = st.selectbox("Select first measurement", numeric_cols, key="paired1")
                        col2 = st.selectbox("Select second measurement", numeric_cols, key="paired2")
                        
                        paired_data = df[[col1, col2]].dropna()
                        if len(paired_data) > 0:
                            t_stat, p_value = stats.ttest_rel(paired_data[col1], paired_data[col2])
                            
                            st.write(f"**Sample size:** {len(paired_data)}")
                            st.write(f"**Mean difference:** {(paired_data[col1] - paired_data[col2]).mean():.4f}")
                            st.write(f"**t-statistic:** {t_stat:.4f}")
                            st.write(f"**p-value:** {p_value:.4f}")
                            
                            if p_value < 0.05:
                                st.success(f"✅ Significant difference found between measurements")
                            else:
                                st.info(f"ℹ️ No significant difference found between measurements")
                            
                            # Visualization
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=paired_data[col1], y=paired_data[col2],
                                                    mode='markers', text=paired_data.index))
                            fig.add_trace(go.Scatter(x=[paired_data[col1].min(), paired_data[col1].max()],
                                                    y=[paired_data[col1].min(), paired_data[col1].max()],
                                                    mode='lines', name='y=x', line=dict(dash='dash')))
                            fig.update_layout(title=f"Paired Comparison: {col1} vs {col2}")
                            st.plotly_chart(fig, use_container_width=True)
            
            elif test_category == "Non-parametric Tests":
                nonparam_test = st.selectbox(
                    "Select non-parametric test",
                    ["Mann-Whitney U", "Wilcoxon Signed-Rank", "Kruskal-Wallis H", "Friedman Test"]
                )
                
                if nonparam_test == "Mann-Whitney U":
                    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                        num_col = st.selectbox("Select numeric variable", numeric_cols, key="mw_num")
                        cat_col = st.selectbox("Select grouping variable", categorical_cols, key="mw_cat")
                        
                        groups = df[cat_col].dropna().unique()
                        if len(groups) == 2:
                            group1 = df[df[cat_col] == groups[0]][num_col].dropna()
                            group2 = df[df[cat_col] == groups[1]][num_col].dropna()
                            
                            u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                            
                            st.write(f"**U-statistic:** {u_stat:.4f}")
                            st.write(f"**p-value:** {p_value:.4f}")
                            
                            # Effect size (r = Z/√N)
                            from scipy.stats import norm
                            z_score = norm.ppf(p_value/2) if p_value < 1 else 0
                            effect_size = abs(z_score) / np.sqrt(len(group1) + len(group2))
                            st.write(f"**Effect size (r):** {effect_size:.4f}")
                            
                            if p_value < 0.05:
                                st.success(f"✅ Significant difference found between groups")
                            else:
                                st.info(f"ℹ️ No significant difference found between groups")
                            
                            # Visualization
                            fig = px.violin(df, x=cat_col, y=num_col, box=True, points="all",
                                          title=f"Mann-Whitney U Test: {num_col} by {cat_col}")
                            st.plotly_chart(fig, use_container_width=True)
            
            elif test_category == "ANOVA & Post-hoc":
                if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                    num_col = st.selectbox("Select numeric variable", numeric_cols, key="anova_num")
                    cat_col = st.selectbox("Select grouping variable", categorical_cols, key="anova_cat")
                    
                    groups = [df[df[cat_col] == group][num_col].dropna() 
                             for group in df[cat_col].unique() if len(df[df[cat_col] == group]) > 0]
                    
                    if len(groups) >= 2:
                        # One-way ANOVA
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        st.write("**One-way ANOVA Results:**")
                        st.write(f"**F-statistic:** {f_stat:.4f}")
                        st.write(f"**p-value:** {p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.success("✅ Significant differences found between groups")
                            
                            # Post-hoc Tukey HSD
                            if st.button("Run Tukey HSD Post-hoc Test"):
                                tukey = pairwise_tukeyhsd(df[num_col].dropna(), df[cat_col].dropna())
                                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], 
                                                       columns=tukey.summary().data[0])
                                st.dataframe(tukey_df)
                                
                                # Visualize confidence intervals
                                fig = go.Figure()
                                for i, row in enumerate(tukey_df.itertuples()):
                                    if row.padj < 0.05:
                                        color = 'green'
                                    else:
                                        color = 'red'
                                    fig.add_trace(go.Scatter(x=[row[4], row[5]], y=[i, i],
                                                            mode='lines', line=dict(color=color, width=3),
                                                            name=f"{row[1]} vs {row[2]}"))
                                fig.update_layout(title="Tukey HSD Confidence Intervals",
                                                xaxis_title="Mean Difference",
                                                yaxis_title="Comparison")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("ℹ️ No significant differences found between groups")
                        
                        # Visualization
                        fig = px.box(df, x=cat_col, y=num_col, title=f"ANOVA: {num_col} by {cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error in hypothesis testing: {str(e)}")
            st.info("💡 Tip: Ensure you have sufficient data and appropriate variable types for the selected test")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Distribution Analysis & Normality Tests")
        
        try:
            if numeric_cols:
                col = st.selectbox("Select column for distribution analysis", numeric_cols, key="dist_col")
                data = df[col].dropna()
                
                if len(data) > 0:
                    # Multiple normality tests
                    st.markdown("### 🔍 Normality Tests")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Shapiro-Wilk test
                        if len(data) <= 5000:
                            shapiro_stat, shapiro_p = stats.shapiro(data)
                            st.write("**Shapiro-Wilk Test**")
                            st.write(f"Statistic: {shapiro_stat:.4f}")
                            st.write(f"P-value: {shapiro_p:.4f}")
                            if shapiro_p < 0.05:
                                st.error("❌ Not normally distributed")
                            else:
                                st.success("✅ Normally distributed")
                    
                    with col2:
                        # Kolmogorov-Smirnov test
                        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                        st.write("**Kolmogorov-Smirnov Test**")
                        st.write(f"Statistic: {ks_stat:.4f}")
                        st.write(f"P-value: {ks_p:.4f}")
                        if ks_p < 0.05:
                            st.error("❌ Not normally distributed")
                        else:
                            st.success("✅ Normally distributed")
                    
                    # Anderson-Darling test
                    anderson_stat, anderson_crit, anderson_sig = stats.anderson(data, dist='norm')
                    st.write("**Anderson-Darling Test**")
                    st.write(f"Statistic: {anderson_stat:.4f}")
                    for i in range(len(anderson_crit)):
                        st.write(f"Critical value at {anderson_sig[i]}%: {anderson_crit[i]:.4f}")
                    
                    # D'Agostino's K-squared test
                    skew_stat, skew_p = stats.skewtest(data)
                    kurt_stat, kurt_p = stats.kurtosistest(data)
                    
                    st.write("**D'Agostino's Tests**")
                    st.write(f"Skewness test p-value: {skew_p:.4f}")
                    st.write(f"Kurtosis test p-value: {kurt_p:.4f}")
                    
                    # Distribution fitting
                    st.markdown("### 📈 Distribution Fitting")
                    
                    distributions = ['norm', 'expon', 'gamma', 'beta', 'lognorm', 'uniform']
                    selected_dist = st.selectbox("Select distribution to fit", distributions)
                    
                    if selected_dist == 'norm':
                        params = stats.norm.fit(data)
                        pdf = stats.norm.pdf(np.sort(data), *params)
                    elif selected_dist == 'expon':
                        params = stats.expon.fit(data)
                        pdf = stats.expon.pdf(np.sort(data), *params)
                    elif selected_dist == 'gamma':
                        params = stats.gamma.fit(data)
                        pdf = stats.gamma.pdf(np.sort(data), *params)
                    elif selected_dist == 'beta':
                        # Scale data to [0,1] for beta distribution
                        scaled_data = (data - data.min()) / (data.max() - data.min())
                        scaled_data = scaled_data[(scaled_data > 0) & (scaled_data < 1)]
                        if len(scaled_data) > 0:
                            params = stats.beta.fit(scaled_data)
                            pdf = stats.beta.pdf(np.sort(scaled_data), *params)
                    elif selected_dist == 'lognorm':
                        params = stats.lognorm.fit(data)
                        pdf = stats.lognorm.pdf(np.sort(data), *params)
                    elif selected_dist == 'uniform':
                        params = stats.uniform.fit(data)
                        pdf = stats.uniform.pdf(np.sort(data), *params)
                    
                    # Plot histogram with fitted distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Data", opacity=0.7))
                    
                    if selected_dist != 'beta':
                        fig.add_trace(go.Scatter(x=np.sort(data), y=pdf * len(data) * (data.max() - data.min()) / 30,
                                               mode='lines', name=f"Fitted {selected_dist}",
                                               line=dict(color='red', width=2)))
                    
                    fig.update_layout(title=f"Histogram with Fitted {selected_dist} Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Q-Q plot with confidence bands
                    st.markdown("### 📊 Enhanced Q-Q Plot")
                    
                    # Generate theoretical quantiles
                    theoretical_q = np.random.normal(data.mean(), data.std(), len(data))
                    theoretical_q.sort()
                    data_sorted = np.sort(data)
                    
                    # Calculate confidence bands (bootstrap)
                    n_bootstrap = 100
                    bootstrap_lines = []
                    for i in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(data, len(data), replace=True)
                        bootstrap_sample.sort()
                        bootstrap_lines.append(bootstrap_sample)
                    
                    bootstrap_lines = np.array(bootstrap_lines)
                    lower_band = np.percentile(bootstrap_lines, 2.5, axis=0)
                    upper_band = np.percentile(bootstrap_lines, 97.5, axis=0)
                    
                    fig = go.Figure()
                    
                    # Add confidence band
                    fig.add_trace(go.Scatter(x=np.concatenate([theoretical_q, theoretical_q[::-1]]),
                                            y=np.concatenate([lower_band, upper_band[::-1]]),
                                            fill='toself', fillcolor='rgba(0,100,80,0.2)',
                                            line=dict(color='rgba(255,255,255,0)'),
                                            name='95% CI'))
                    
                    # Add data points
                    fig.add_trace(go.Scatter(x=theoretical_q, y=data_sorted,
                                            mode='markers', name='Data'))
                    
                    # Add reference line
                    fig.add_trace(go.Scatter(x=[data_sorted.min(), data_sorted.max()],
                                            y=[data_sorted.min(), data_sorted.max()],
                                            mode='lines', name='Reference',
                                            line=dict(color='red', dash='dash')))
                    
                    fig.update_layout(title=f"Enhanced Q-Q Plot with 95% Confidence Band")
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error in distribution analysis: {str(e)}")
            st.info("💡 Tip: Ensure you have sufficient data points for distribution fitting")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📉 Advanced Time Series Analysis")
        
        try:
            if datetime_cols and numeric_cols:
                date_col = st.selectbox("Select date column", datetime_cols)
                value_col = st.selectbox("Select value column", numeric_cols, key="ts_value_adv")
                
                # Prepare time series data
                ts_df = df[[date_col, value_col]].dropna().sort_values(date_col)
                ts_df.set_index(date_col, inplace=True)
                
                if len(ts_df) >= 10:
                    # Time series decomposition
                    st.markdown("### 🔄 Time Series Decomposition")
                    
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Determine frequency
                    freq_options = {
                        'Auto-detect': None,
                        'Daily (7)': 7,
                        'Weekly (52)': 52,
                        'Monthly (12)': 12,
                        'Quarterly (4)': 4
                    }
                    
                    selected_freq = st.selectbox("Select seasonal period", list(freq_options.keys()))
                    period = freq_options[selected_freq]
                    
                    if period is None:
                        # Auto-detect frequency
                        try:
                            freq = pd.infer_freq(ts_df.index)
                            if freq:
                                period_map = {'D': 7, 'W': 52, 'M': 12, 'Q': 4}
                                period = period_map.get(freq[0], 7)
                        except:
                            period = 7
                    
                    if len(ts_df) >= 2 * period:
                        decomposition = seasonal_decompose(ts_df[value_col], model='additive', period=period)
                        
                        fig = make_subplots(rows=4, cols=1,
                                           subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
                        
                        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[value_col],
                                                mode='lines', name='Original'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.trend,
                                                mode='lines', name='Trend'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.seasonal,
                                                mode='lines', name='Seasonal'), row=3, col=1)
                        fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.resid,
                                                mode='lines', name='Residual'), row=4, col=1)
                        
                        fig.update_layout(height=800, title="Time Series Decomposition")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Stationarity tests
                    st.markdown("### 📊 Stationarity Tests")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # ADF test
                        adf_result = adfuller(ts_df[value_col].dropna())
                        st.write("**Augmented Dickey-Fuller Test**")
                        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                        st.write(f"p-value: {adf_result[1]:.4f}")
                        st.write(f"Critical values:")
                        for key, value in adf_result[4].items():
                            st.write(f"   {key}: {value:.4f}")
                        
                        if adf_result[1] < 0.05:
                            st.success("✅ Series is stationary")
                        else:
                            st.warning("⚠️ Series is non-stationary")
                    
                    with col2:
                        # KPSS test
                        kpss_result = kpss(ts_df[value_col].dropna(), regression='c')
                        st.write("**KPSS Test**")
                        st.write(f"KPSS Statistic: {kpss_result[0]:.4f}")
                        st.write(f"p-value: {kpss_result[1]:.4f}")
                        st.write(f"Critical values:")
                        for key, value in kpss_result[3].items():
                            st.write(f"   {key}: {value:.4f}")
                        
                        if kpss_result[1] < 0.05:
                            st.warning("⚠️ Series is non-stationary")
                        else:
                            st.success("✅ Series is stationary")
                    
                    # ACF and PACF plots
                    st.markdown("### 📈 ACF and PACF Plots")
                    
                    lags = st.slider("Number of lags", 10, 50, 20)
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    plot_acf(ts_df[value_col].dropna(), lags=lags, ax=ax1)
                    plot_pacf(ts_df[value_col].dropna(), lags=lags, ax=ax2)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Forecasting with simple models
                    st.markdown("### 🔮 Simple Forecasting")
                    
                    forecast_periods = st.slider("Forecast periods", 1, 30, 10)
                    
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    
                    model = ExponentialSmoothing(ts_df[value_col], 
                                                seasonal_periods=period,
                                                trend='add', seasonal='add')
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(forecast_periods)
                    
                    # Plot forecast
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[value_col],
                                            mode='lines', name='Historical'))
                    fig.add_trace(go.Scatter(x=forecast.index, y=forecast,
                                            mode='lines+markers', name='Forecast',
                                            line=dict(color='red')))
                    fig.update_layout(title=f"Exponential Smoothing Forecast ({forecast_periods} periods)")
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.info("ℹ️ Need both datetime and numeric columns for time series analysis")
        
        except Exception as e:
            st.error(f"❌ Error in time series analysis: {str(e)}")
            st.info("💡 Tip: Ensure your date column is properly formatted as datetime")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🎲 Probability & Sampling Analysis")
        
        try:
            if numeric_cols:
                col = st.selectbox("Select column for probability analysis", numeric_cols, key="prob_col")
                data = df[col].dropna()
                
                if len(data) > 0:
                    # Probability distribution fitting
                    st.markdown("### 📊 Probability Distribution Fitting")
                    
                    # Calculate empirical CDF
                    sorted_data = np.sort(data)
                    ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sorted_data, y=ecdf,
                                            mode='lines', name='Empirical CDF'))
                    
                    # Fit theoretical distributions
                    dist_options = ['Normal', 'Exponential', 'Gamma', 'Log-normal']
                    selected_dist = st.multiselect("Select distributions to compare", dist_options, default=['Normal'])
                    
                    colors = ['red', 'green', 'blue', 'orange']
                    for i, dist_name in enumerate(selected_dist):
                        if dist_name == 'Normal':
                            params = stats.norm.fit(data)
                            theoretical_cdf = stats.norm.cdf(sorted_data, *params)
                        elif dist_name == 'Exponential':
                            params = stats.expon.fit(data)
                            theoretical_cdf = stats.expon.cdf(sorted_data, *params)
                        elif dist_name == 'Gamma':
                            params = stats.gamma.fit(data)
                            theoretical_cdf = stats.gamma.cdf(sorted_data, *params)
                        elif dist_name == 'Log-normal':
                            params = stats.lognorm.fit(data)
                            theoretical_cdf = stats.lognorm.cdf(sorted_data, *params)
                        
                        fig.add_trace(go.Scatter(x=sorted_data, y=theoretical_cdf,
                                                mode='lines', name=f'{dist_name} CDF',
                                                line=dict(color=colors[i], dash='dash')))
                    
                    fig.update_layout(title="CDF Comparison: Empirical vs Theoretical",
                                    xaxis_title=col, yaxis_title="Cumulative Probability")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Goodness of fit tests
                    st.markdown("### 📈 Goodness of Fit Tests")
                    
                    for dist_name in selected_dist:
                        if dist_name == 'Normal':
                            ks_stat, ks_p = stats.kstest(data, 'norm', args=stats.norm.fit(data))
                        elif dist_name == 'Exponential':
                            ks_stat, ks_p = stats.kstest(data, 'expon', args=stats.expon.fit(data))
                        elif dist_name == 'Gamma':
                            ks_stat, ks_p = stats.kstest(data, 'gamma', args=stats.gamma.fit(data))
                        elif dist_name == 'Log-normal':
                            ks_stat, ks_p = stats.kstest(data, 'lognorm', args=stats.lognorm.fit(data))
                        
                        st.write(f"**{dist_name} Distribution**")
                        st.write(f"KS Statistic: {ks_stat:.4f}")
                        st.write(f"P-value: {ks_p:.4f}")
                        
                        if ks_p < 0.05:
                            st.error(f"❌ Data does NOT follow {dist_name} distribution")
                        else:
                            st.success(f"✅ Data may follow {dist_name} distribution")
                    
                    # Sampling analysis
                    st.markdown("### 🎯 Sampling Analysis")
                    
                    sample_size = st.slider("Sample size", 10, min(500, len(data)), 100)
                    n_samples = st.slider("Number of samples", 10, 1000, 100)
                    
                    # Bootstrap sampling
                    bootstrap_means = []
                    for i in range(n_samples):
                        sample = np.random.choice(data, sample_size, replace=True)
                        bootstrap_means.append(sample.mean())
                    
                    bootstrap_means = np.array(bootstrap_means)
                    
                    # Plot sampling distribution
                    fig = make_subplots(rows=1, cols=2,
                                       subplot_titles=("Sampling Distribution of Mean", 
                                                      "Confidence Intervals"))
                    
                    fig.add_trace(go.Histogram(x=bootstrap_means, nbinsx=30,
                                              name="Sample Means"), row=1, col=1)
                    
                    # Add confidence intervals
                    ci_lower = np.percentile(bootstrap_means, 2.5)
                    ci_upper = np.percentile(bootstrap_means, 97.5)
                    
                    fig.add_trace(go.Scatter(x=[ci_lower, ci_lower], y=[0, 10],
                                            mode='lines', name='95% CI Lower',
                                            line=dict(color='red', dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[ci_upper, ci_upper], y=[0, 10],
                                            mode='lines', name='95% CI Upper',
                                            line=dict(color='red', dash='dash')), row=1, col=1)
                    
                    # Confidence interval plot
                    for i in range(min(20, n_samples)):
                        sample_mean = bootstrap_means[i]
                        fig.add_trace(go.Scatter(x=[i, i], y=[sample_mean - data.std()/np.sqrt(sample_size),
                                                            sample_mean + data.std()/np.sqrt(sample_size)],
                                                mode='lines', line=dict(color='blue', width=1),
                                                showlegend=False), row=1, col=2)
                        fig.add_trace(go.Scatter(x=[i], y=[sample_mean],
                                                mode='markers', marker=dict(color='red', size=5),
                                                showlegend=False), row=1, col=2)
                    
                    fig.update_layout(height=500, title="Bootstrap Sampling Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sampling statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Population Mean", f"{data.mean():.4f}")
                    with col2:
                        st.metric("Mean of Sample Means", f"{bootstrap_means.mean():.4f}")
                    with col3:
                        st.metric("Standard Error", f"{bootstrap_means.std():.4f}")
                    
                    st.write(f"**95% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        except Exception as e:
            st.error(f"❌ Error in probability analysis: {str(e)}")
            st.info("💡 Tip: Ensure you have sufficient data for probability analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export Statistical Report")
    
    try:
        report_text = f"""
        STATISTICAL ANALYSIS REPORT
        ===========================
        
        Dataset Information:
        • Total Rows: {df.shape[0]:,}
        • Total Columns: {df.shape[1]}
        • Numeric Columns: {len(numeric_cols)}
        • Categorical Columns: {len(categorical_cols)}
        • Datetime Columns: {len(datetime_cols)}
        
        Summary Statistics:
        {df[numeric_cols].describe().to_string()}
        
        Analysis Performed:
        • Descriptive Statistics
        • Correlation Analysis
        • Hypothesis Testing
        • Distribution Analysis
        • Time Series Analysis (if applicable)
        • Probability & Sampling Analysis
        """
        
        st.download_button(
            label="📥 Download Complete Statistical Report",
            data=report_text,
            file_name="statistical_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")