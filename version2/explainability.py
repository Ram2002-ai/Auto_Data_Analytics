import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap

def explain_model(model, X, y=None, feature_names=None):
    """
    Explain model predictions using various techniques
    """
    st.subheader("🔍 Model Explainability")
    
    if feature_names is None:
        feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
    
    # Create tabs for different explanation methods
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Values", "Partial Dependence"])
    
    with tab1:
        st.markdown("### 📊 Feature Importance")
        
        # Method selection
        method = st.radio(
            "Importance method",
            ["Built-in", "Permutation"],
            horizontal=True
        )
        
        if method == "Built-in":
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(importance_df.head(20), x='importance', y='feature',
                           orientation='h', title="Feature Importance (Built-in)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Model doesn't have built-in feature importance")
        
        else:  # Permutation importance
            if y is not None:
                with st.spinner("Calculating permutation importance..."):
                    perm_importance = permutation_importance(model, X, y, n_repeats=10)
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': perm_importance.importances_mean,
                        'std': perm_importance.importances_std
                    }).sort_values('importance', ascending=False)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=importance_df['importance'].head(20),
                        y=importance_df['feature'].head(20),
                        orientation='h',
                        error_x=dict(
                            type='data',
                            array=importance_df['std'].head(20),
                            visible=True
                        )
                    ))
                    fig.update_layout(title="Permutation Importance (with error bars)",
                                    xaxis_title="Importance")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need target values for permutation importance")
    
    with tab2:
        st.markdown("### 📈 SHAP Values")
        
        if hasattr(model, 'predict'):
            with st.spinner("Calculating SHAP values (this may take a moment)..."):
                try:
                    # Create explainer based on model type
                    if str(type(model)).find('sklearn') != -1:
                        explainer = shap.Explainer(model, X[:100])  # Use subset for speed
                    else:
                        explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.Explainer(model, X[:100])
                    
                    # Calculate SHAP values
                    shap_values = explainer(X[:100])  # Limit to 100 samples for performance
                    
                    # Summary plot
                    st.markdown("#### SHAP Summary Plot")
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X[:100], feature_names=feature_names, show=False)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Waterfall plot for a single prediction
                    st.markdown("#### Single Prediction Explanation")
                    sample_idx = st.slider("Select sample index", 0, min(99, len(X)-1), 0)
                    
                    fig, ax = plt.subplots()
                    shap.waterfall_plot(shap_values[sample_idx], show=False)
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error calculating SHAP values: {str(e)}")
                    st.info("Try using a smaller sample or a different model type")
        else:
            st.warning("Model doesn't support prediction")
    
    with tab3:
        st.markdown("### 📉 Partial Dependence Plots")
        
        if hasattr(model, 'predict') and len(feature_names) > 0:
            from sklearn.inspection import partial_dependence
            
            selected_feature = st.selectbox("Select feature for PDP", feature_names)
            
            if selected_feature:
                feature_idx = list(feature_names).index(selected_feature)
                
                # Calculate partial dependence
                pdp = partial_dependence(model, X, [feature_idx], grid_resolution=50)
                
                # Create plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pdp['values'][0],
                    y=pdp['average'][0],
                    mode='lines+markers',
                    name='Partial Dependence'
                ))
                
                fig.update_layout(
                    title=f"Partial Dependence Plot for {selected_feature}",
                    xaxis_title=selected_feature,
                    yaxis_title="Prediction"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual conditional expectation (ICE) plots
                if st.checkbox("Show ICE plots"):
                    ice_data = []
                    for i in range(min(10, X.shape[0])):  # Show up to 10 lines
                        ice = partial_dependence(model, X.iloc[i:i+1], [feature_idx], grid_resolution=20)
                        ice_data.append(ice['average'][0])
                    
                    fig = go.Figure()
                    for i, ice in enumerate(ice_data):
                        fig.add_trace(go.Scatter(
                            x=pdp['values'][0],
                            y=ice,
                            mode='lines',
                            name=f'Sample {i}',
                            line=dict(width=1, color='lightgray')
                        ))
                    
                    # Add average line
                    fig.add_trace(go.Scatter(
                        x=pdp['values'][0],
                        y=pdp['average'][0],
                        mode='lines',
                        name='Average',
                        line=dict(width=3, color='red')
                    ))
                    
                    fig.update_layout(
                        title=f"ICE Plots for {selected_feature}",
                        xaxis_title=selected_feature,
                        yaxis_title="Prediction"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need more features for partial dependence plots")