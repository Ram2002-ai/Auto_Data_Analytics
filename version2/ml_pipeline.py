import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            AdaBoostClassifier, AdaBoostRegressor,
                            VotingClassifier, VotingRegressor)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

class MLPipelineError(Exception):
    """Custom exception for ML pipeline errors"""
    pass

def validate_ml_data(df, target, features):
    """Validate data for machine learning"""
    issues = []
    
    if df.empty:
        issues.append("Dataset is empty")
        return issues
    
    if target not in df.columns:
        issues.append(f"Target column '{target}' not found in dataset")
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        issues.append(f"Features not found: {missing_features}")
    
    # Check for sufficient data
    if df.shape[0] < 10:
        issues.append("Dataset too small (minimum 10 rows required)")
    
    # Check for constant columns
    for col in features:
        if df[col].nunique() == 1:
            issues.append(f"Feature '{col}' is constant")
    
    # Check target for classification
    if target in df.columns:
        if df[target].dtype in ['object', 'category'] or df[target].nunique() <= 20:
            if df[target].nunique() == 1:
                issues.append("Target has only one class")
            elif df[target].nunique() > 50:
                issues.append(f"Target has {df[target].nunique()} classes, which may cause issues")
    
    return issues

def safe_ml_operation(func, *args, **kwargs):
    """Safely execute ML operations with error handling"""
    try:
        result = func(*args, **kwargs)
        return result, None
    except ValueError as e:
        error_msg = f"Value Error: {str(e)}. Check your data types and values."
        return None, error_msg
    except MemoryError as e:
        error_msg = "Memory Error: Dataset too large. Try reducing the number of features or using a sample."
        return None, error_msg
    except Exception as e:
        error_msg = f"ML Error: {str(e)}"
        return None, error_msg

def run_ml_pipeline(df):
    """
    Enhanced machine learning pipeline with comprehensive error handling
    """
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>🤖 Advanced Machine Learning Pipeline</h2>
        <p style='color: gray;'>Train, evaluate, and compare multiple ML models with automatic error handling</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Check if dataset is suitable for ML
        if df.shape[0] < 10:
            st.error("❌ Dataset too small for machine learning (need at least 10 rows)")
            return
        
        # Create tabs for different ML stages
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚙️ Configuration", 
            "📊 Model Training", 
            "📈 Model Evaluation", 
            "🔮 Predictions",
            "📋 ML Report"
        ])
        
        with tab1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Model Configuration")
            
            try:
                # Target selection with validation
                st.markdown("### 🎯 Target Variable")
                
                # Auto-detect potential target columns
                potential_targets = []
                target_types = {}
                
                for col in df.columns:
                    try:
                        if df[col].dtype in ['int64', 'float64']:
                            if df[col].nunique() <= 20:
                                potential_targets.append(col)
                                target_types[col] = "Classification (low cardinality)"
                            else:
                                potential_targets.append(col)
                                target_types[col] = "Regression"
                        elif df[col].dtype in ['object', 'category']:
                            if df[col].nunique() <= 50:
                                potential_targets.append(col)
                                target_types[col] = f"Classification ({df[col].nunique()} classes)"
                    except Exception as e:
                        st.warning(f"⚠️ Couldn't analyze column {col}: {str(e)}")
                
                if not potential_targets:
                    st.error("❌ No suitable target columns found. Need numeric or categorical columns with reasonable cardinality.")
                    return
                
                target = st.selectbox(
                    "Select target column",
                    potential_targets,
                    help=f"Column types: {target_types}"
                )
                
                # Task type detection
                if df[target].dtype in ['object', 'category'] or df[target].nunique() <= 20:
                    task_type = "Classification"
                    unique_values = df[target].nunique()
                    
                    if unique_values == 2:
                        st.success("✅ **Binary Classification** problem detected")
                    elif unique_values <= 10:
                        st.info(f"📊 **Multi-class Classification** with {unique_values} classes")
                    else:
                        st.warning(f"⚠️ **Multi-class Classification** with {unique_values} classes - may be challenging")
                        
                    # Check class balance
                    class_dist = df[target].value_counts(normalize=True)
                    if class_dist.min() < 0.1:
                        st.warning("⚠️ Class imbalance detected. Consider using class weights or resampling.")
                else:
                    task_type = "Regression"
                    st.info("📈 **Regression** task detected")
                    
                    # Check target distribution
                    target_skew = df[target].skew()
                    if abs(target_skew) > 1:
                        st.warning(f"⚠️ Target variable is highly skewed (skewness: {target_skew:.2f}). Consider log transformation.")
                
                # Feature selection
                st.markdown("### 🔍 Feature Selection")
                
                # Auto-select features (exclude target)
                all_features = [col for col in df.columns if col != target]
                
                # Remove problematic columns
                problematic_cols = []
                for col in all_features:
                    try:
                        if df[col].nunique() == 1:
                            problematic_cols.append(col)
                        elif df[col].isnull().sum() > len(df) * 0.5:
                            problematic_cols.append(col)
                    except:
                        problematic_cols.append(col)
                
                if problematic_cols:
                    st.warning(f"⚠️ Problematic columns detected (will be excluded): {problematic_cols}")
                    all_features = [f for f in all_features if f not in problematic_cols]
                
                if not all_features:
                    st.error("❌ No valid features remaining after filtering.")
                    return
                
                # Select features
                selected_features = st.multiselect(
                    "Choose features for modeling",
                    all_features,
                    default=all_features[:min(10, len(all_features))],
                    help="Select the columns to use as features. Using too many features may cause overfitting."
                )
                
                if not selected_features:
                    st.warning("⚠️ Please select at least one feature")
                    return
                
                # Validate selected features
                validation_issues = validate_ml_data(df, target, selected_features)
                if validation_issues:
                    for issue in validation_issues:
                        st.warning(f"⚠️ {issue}")
                
                # Data preprocessing options
                st.markdown("### 🛠️ Preprocessing Options")
                
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
                    scaler_option = st.selectbox("Feature scaling", ["None", "StandardScaler", "MinMaxScaler"])
                
                with col2:
                    cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
                    if task_type == "Classification":
                        handle_imbalance = st.checkbox("Handle class imbalance", value=False,
                                                       help="Use class weights or sampling techniques")
                    else:
                        handle_imbalance = False
                
                # Model selection based on task type
                st.markdown("### 🤖 Model Selection")
                
                if task_type == "Classification":
                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "K-Nearest Neighbors": KNeighborsClassifier(),
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                        "LightGBM": LGBMClassifier(verbose=-1, random_state=42),
                        "AdaBoost": AdaBoostClassifier(random_state=42),
                        "SVM": SVC(probability=True, random_state=42)
                    }
                    
                    # Default models for quick selection
                    default_models = ["Logistic Regression", "Random Forest", "XGBoost"]
                else:  # Regression
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(random_state=42),
                        "Lasso Regression": Lasso(random_state=42),
                        "Decision Tree": DecisionTreeRegressor(random_state=42),
                        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                        "XGBoost": XGBRegressor(random_state=42),
                        "LightGBM": LGBMRegressor(verbose=-1, random_state=42),
                        "AdaBoost": AdaBoostRegressor(random_state=42),
                        "SVR": SVR()
                    }
                    
                    default_models = ["Linear Regression", "Random Forest", "XGBoost"]
                
                selected_models = st.multiselect(
                    "Choose models to train",
                    list(models.keys()),
                    default=default_models,
                    help="Select multiple models to compare performance"
                )
                
                if not selected_models:
                    st.warning("⚠️ Please select at least one model")
                    return
                
                # Advanced options
                with st.expander("⚡ Advanced Options"):
                    do_tuning = st.checkbox("Perform hyperparameter tuning", value=False,
                                           help="Grid search for best parameters (may be slow)")
                    
                    if do_tuning:
                        tuning_folds = st.slider("Tuning CV folds", 2, 5, 3)
                        max_tuning_iter = st.slider("Max tuning iterations per model", 5, 50, 20)
                    
                    use_sampling = st.checkbox("Use data sampling (for large datasets)", value=False,
                                              help="Use a sample for faster experimentation")
                    
                    if use_sampling:
                        sample_size = st.slider("Sample size (%)", 10, 100, 100, 10) / 100
                    
                    random_state = st.number_input("Random seed", value=42, min_value=0, max_value=999)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store configuration in session state
                st.session_state['ml_config'] = {
                    'target': target,
                    'features': selected_features,
                    'task_type': task_type,
                    'test_size': test_size,
                    'scaler': scaler_option,
                    'cv_folds': cv_folds,
                    'handle_imbalance': handle_imbalance,
                    'models': {name: models[name] for name in selected_models},
                    'do_tuning': do_tuning,
                    'random_state': random_state
                }
                
            except Exception as e:
                st.error(f"❌ Error in configuration: {str(e)}")
                st.info("💡 Tip: Check your data types and ensure all columns are valid")
                return
        
        with tab2:
            if 'ml_config' not in st.session_state:
                st.info("ℹ️ Please configure your model in the 'Configuration' tab first")
                return
            
            if st.button("🚀 Start Training", use_container_width=True, type="primary"):
                try:
                    config = st.session_state['ml_config']
                    
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    
                    # Prepare data with error handling
                    with st.spinner("📊 Preparing data..."):
                        try:
                            X = df[config['features']].copy()
                            y = df[config['target']].copy()
                            
                            # Handle missing values
                            if X.isnull().sum().sum() > 0:
                                st.info(f"⚠️ Handling {X.isnull().sum().sum()} missing values in features...")
                                X = X.fillna(X.mean(numeric_only=True)).fillna(X.mode().iloc[0])
                            
                            # Handle categorical features
                            cat_features = X.select_dtypes(include=['object', 'category']).columns
                            if len(cat_features) > 0:
                                st.info(f"🔄 Encoding categorical features: {list(cat_features)}")
                                X = pd.get_dummies(X, columns=cat_features)
                            
                            # Handle target encoding for classification
                            le = None
                            if config['task_type'] == "Classification" and y.dtype == 'object':
                                le = LabelEncoder()
                                y = le.fit_transform(y)
                                st.info(f"📊 Target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                            
                            # Handle class imbalance
                            if config['task_type'] == "Classification" and config['handle_imbalance']:
                                from sklearn.utils.class_weight import compute_class_weight
                                class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
                                st.info(f"⚖️ Using class weights: {dict(zip(np.unique(y), class_weights))}")
                            
                            # Scale features
                            scaler = None
                            if config['scaler'] != "None":
                                if config['scaler'] == "StandardScaler":
                                    scaler = StandardScaler()
                                else:
                                    scaler = MinMaxScaler()
                                X_scaled = scaler.fit_transform(X)
                                X = pd.DataFrame(X_scaled, columns=X.columns)
                            
                            # Split data
                            stratify = y if config['task_type'] == "Classification" else None
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=config['test_size'], 
                                random_state=config['random_state'], 
                                stratify=stratify
                            )
                            
                            st.success(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
                            
                        except Exception as e:
                            st.error(f"❌ Error in data preparation: {str(e)}")
                            return
                    
                    # Train models
                    results = []
                    trained_models = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, (model_name, model) in enumerate(config['models'].items()):
                        status_text.text(f"🔄 Training {model_name}...")
                        
                        try:
                            # Apply class weights if needed
                            if config['task_type'] == "Classification" and config['handle_imbalance']:
                                if hasattr(model, 'class_weight'):
                                    model.set_params(class_weight='balanced')
                            
                            # Train
                            start_time = time.time()
                            model.fit(X_train, y_train)
                            training_time = time.time() - start_time
                            
                            # Store trained model
                            trained_models[model_name] = {
                                'model': model,
                                'scaler': scaler,
                                'label_encoder': le,
                                'features': X.columns.tolist()
                            }
                            
                            # Predict
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            if config['task_type'] == "Classification":
                                try:
                                    accuracy = accuracy_score(y_test, y_pred)
                                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                                    
                                    # Cross-validation
                                    cv_scores = cross_val_score(model, X_train, y_train, cv=config['cv_folds'])
                                    
                                    results.append({
                                        "Model": model_name,
                                        "Accuracy": f"{accuracy:.4f}",
                                        "Precision": f"{precision:.4f}",
                                        "Recall": f"{recall:.4f}",
                                        "F1 Score": f"{f1:.4f}",
                                        "CV Score": f"{cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})",
                                        "Time (s)": f"{training_time:.2f}"
                                    })
                                except Exception as e:
                                    st.warning(f"⚠️ Could not calculate all metrics for {model_name}: {str(e)}")
                            
                            else:  # Regression
                                try:
                                    mse = mean_squared_error(y_test, y_pred)
                                    rmse = np.sqrt(mse)
                                    mae = mean_absolute_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    
                                    # Cross-validation
                                    cv_scores = cross_val_score(model, X_train, y_train, cv=config['cv_folds'], scoring='r2')
                                    
                                    results.append({
                                        "Model": model_name,
                                        "R² Score": f"{r2:.4f}",
                                        "RMSE": f"{rmse:.4f}",
                                        "MAE": f"{mae:.4f}",
                                        "CV R²": f"{cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})",
                                        "Time (s)": f"{training_time:.2f}"
                                    })
                                except Exception as e:
                                    st.warning(f"⚠️ Could not calculate all metrics for {model_name}: {str(e)}")
                            
                        except MemoryError:
                            st.error(f"❌ Out of memory training {model_name}. Try using fewer features or a sample.")
                        except Exception as e:
                            st.warning(f"⚠️ Error training {model_name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(config['models']))
                    
                    status_text.text("✅ Training complete!")
                    
                    if not results:
                        st.error("❌ No models were successfully trained")
                        return
                    
                    # Display results
                    st.subheader("📊 Model Performance Comparison")
                    results_df = pd.DataFrame(results)
                    
                    # Highlight best model
                    if config['task_type'] == "Classification":
                        best_idx = results_df['F1 Score'].astype(float).idxmax()
                    else:
                        best_idx = results_df['R² Score'].astype(float).idxmax()
                    
                    # Style dataframe
                    def highlight_best(s):
                        is_best = s.index == best_idx
                        return ['background-color: #90EE90' if v else '' for v in is_best]
                    
                    st.dataframe(results_df.style.apply(highlight_best), use_container_width=True)
                    
                    # Store results
                    st.session_state['trained_models'] = trained_models
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['task_type'] = config['task_type']
                    st.session_state['results_df'] = results_df
                    
                    # Best model info
                    best_model_name = results_df.iloc[best_idx]['Model']
                    st.success(f"🏆 **Best Model:** {best_model_name}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Critical error in training: {str(e)}")
                    st.info("💡 Try reducing the number of features or models")
        
        with tab3:
            if 'trained_models' not in st.session_state:
                st.info("ℹ️ Train some models first in the 'Model Training' tab")
                return
            
            try:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("📈 Detailed Model Evaluation")
                
                # Model selection for detailed evaluation
                selected_eval_model = st.selectbox(
                    "Select model for detailed evaluation",
                    list(st.session_state['trained_models'].keys())
                )
                
                model_info = st.session_state['trained_models'][selected_eval_model]
                model = model_info['model']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                task_type = st.session_state['task_type']
                
                try:
                    y_pred = model.predict(X_test)
                    
                    if task_type == "Classification":
                        # Confusion Matrix
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig = px.imshow(cm,
                                      text_auto=True,
                                      aspect="auto",
                                      color_continuous_scale='Blues',
                                      title=f"Confusion Matrix - {selected_eval_model}")
                        
                        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Classification Report
                        st.markdown("### Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                        
                        # ROC Curve (for binary classification)
                        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                            st.markdown("### ROC Curve")
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                                    mode='lines',
                                                    name=f'ROC (AUC = {roc_auc:.3f})',
                                                    line=dict(color='blue', width=2)))
                            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                                    mode='lines',
                                                    name='Random',
                                                    line=dict(color='gray', dash='dash')))
                            
                            fig.update_layout(xaxis_title="False Positive Rate",
                                            yaxis_title="True Positive Rate",
                                            title=f"ROC Curve - {selected_eval_model}")
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # Regression
                        # Actual vs Predicted plot
                        st.markdown("### Actual vs Predicted")
                        
                        fig = px.scatter(x=y_test, y=y_pred,
                                       labels={'x': 'Actual', 'y': 'Predicted'},
                                       title=f"Actual vs Predicted - {selected_eval_model}",
                                       trendline="ols")
                        
                        # Add perfect prediction line
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                                mode='lines', name='Perfect Prediction',
                                                line=dict(color='red', dash='dash')))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Residuals plot
                        st.markdown("### Residuals Analysis")
                        residuals = y_test - y_pred
                        
                        fig = make_subplots(rows=1, cols=2,
                                           subplot_titles=("Residuals vs Predicted", "Residuals Distribution"))
                        
                        fig.add_trace(go.Scatter(x=y_pred, y=residuals,
                                                mode='markers',
                                                name='Residuals',
                                                marker=dict(color='blue', opacity=0.5)), row=1, col=1)
                        
                        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
                        
                        fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                                  name='Distribution',
                                                  marker_color='green'), row=1, col=2)
                        
                        fig.update_layout(title=f"Residual Analysis - {selected_eval_model}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Residual statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Residual", f"{residuals.mean():.4f}")
                        with col2:
                            st.metric("Std Residual", f"{residuals.std():.4f}")
                        with col3:
                            st.metric("Residual Range", f"{residuals.max() - residuals.min():.4f}")
                    
                    # Feature Importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("### Feature Importance")
                        feature_importance = pd.DataFrame({
                            'feature': X_test.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=True)
                        
                        fig = px.bar(feature_importance.tail(10),
                                   x='importance', y='feature',
                                   orientation='h',
                                   title="Top 10 Feature Importances",
                                   color='importance',
                                   color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error in evaluation: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error loading evaluation: {str(e)}")
        
        with tab4:
            if 'trained_models' not in st.session_state:
                st.info("ℹ️ Train some models first in the 'Model Training' tab")
                return
            
            try:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("🔮 Make Predictions")
                
                # Model selection for predictions
                selected_pred_model = st.selectbox(
                    "Select model for predictions",
                    list(st.session_state['trained_models'].keys()),
                    key="pred_model"
                )
                
                model_info = st.session_state['trained_models'][selected_pred_model]
                model = model_info['model']
                scaler = model_info['scaler']
                le = model_info.get('label_encoder')
                feature_names = model_info['features']
                
                # Input method
                input_method = st.radio(
                    "Input method",
                    ["Manual input", "Upload new data", "Batch prediction"],
                    horizontal=True
                )
                
                if input_method == "Manual input":
                    st.markdown("### Enter feature values")
                    
                    input_data = {}
                    cols = st.columns(3)
                    
                    for i, feature in enumerate(feature_names):
                        with cols[i % 3]:
                            try:
                                # Get feature range from training data
                                if feature in st.session_state['X_train'].columns:
                                    min_val = float(st.session_state['X_train'][feature].min())
                                    max_val = float(st.session_state['X_train'][feature].max())
                                    mean_val = float(st.session_state['X_train'][feature].mean())
                                    
                                    input_data[feature] = st.slider(
                                        f"{feature}",
                                        min_val, max_val, mean_val,
                                        format="%.4f",
                                        key=f"manual_{feature}"
                                    )
                                else:
                                    input_data[feature] = st.number_input(
                                        f"{feature}",
                                        value=0.0,
                                        key=f"manual_{feature}"
                                    )
                            except Exception as e:
                                st.warning(f"⚠️ Error with {feature}: {str(e)}")
                                input_data[feature] = 0.0
                    
                    if st.button("🔮 Predict", use_container_width=True):
                        try:
                            # Convert input to DataFrame
                            input_df = pd.DataFrame([input_data])
                            
                            # Ensure all features are present
                            for col in feature_names:
                                if col not in input_df.columns:
                                    input_df[col] = 0
                            
                            input_df = input_df[feature_names]
                            
                            # Scale if needed
                            if scaler is not None:
                                input_scaled = scaler.transform(input_df)
                                input_df = pd.DataFrame(input_scaled, columns=feature_names)
                            
                            # Make prediction
                            prediction = model.predict(input_df)[0]
                            
                            # Decode if needed
                            if le is not None:
                                prediction = le.inverse_transform([int(prediction)])[0]
                            
                            # Display prediction with styling
                            st.markdown("""
                            <div class="success-container" style="text-align: center; padding: 2rem;">
                                <h3>🎯 Prediction Result</h3>
                                <h1 style="font-size: 3rem;">{}</h1>
                            </div>
                            """.format(prediction), unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Prediction error: {str(e)}")
                
                elif input_method == "Upload new data":
                    pred_file = st.file_uploader("Upload data for predictions", 
                                                type=["csv", "xlsx"],
                                                key="pred_file")
                    
                    if pred_file:
                        try:
                            if pred_file.name.endswith("csv"):
                                pred_df = pd.read_csv(pred_file)
                            else:
                                pred_df = pd.read_excel(pred_file)
                            
                            st.subheader("📋 Uploaded Data Preview")
                            st.dataframe(pred_df.head())
                            
                            if st.button("🔮 Predict for all rows", use_container_width=True):
                                with st.spinner("Making predictions..."):
                                    try:
                                        # Prepare data
                                        pred_processed = pred_df.copy()
                                        
                                        # Handle categorical features if needed
                                        for col in pred_processed.columns:
                                            if pred_processed[col].dtype == 'object':
                                                pred_processed = pd.get_dummies(pred_processed, columns=[col])
                                        
                                        # Align columns with training data
                                        for col in feature_names:
                                            if col not in pred_processed.columns:
                                                pred_processed[col] = 0
                                        
                                        pred_processed = pred_processed[feature_names]
                                        
                                        # Scale if needed
                                        if scaler is not None:
                                            pred_scaled = scaler.transform(pred_processed)
                                            pred_processed = pd.DataFrame(pred_scaled, columns=feature_names)
                                        
                                        # Make predictions
                                        predictions = model.predict(pred_processed)
                                        
                                        # Decode if needed
                                        if le is not None:
                                            predictions = le.inverse_transform(predictions.astype(int))
                                        
                                        # Add predictions to dataframe
                                        pred_df['Prediction'] = predictions
                                        
                                        st.subheader("📊 Predictions Result")
                                        st.dataframe(pred_df)
                                        
                                        # Download predictions
                                        csv = pred_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Predictions",
                                            data=csv,
                                            file_name="predictions.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                                        
                                    except Exception as e:
                                        st.error(f"❌ Prediction error: {str(e)}")
                        
                        except Exception as e:
                            st.error(f"❌ Error reading file: {str(e)}")
                
                elif input_method == "Batch prediction":
                    st.markdown("### Batch Prediction Settings")
                    
                    n_samples = st.number_input("Number of samples to generate", 
                                                min_value=1, max_value=1000, value=10)
                    
                    if st.button("🎲 Generate Random Samples & Predict", use_container_width=True):
                        try:
                            # Generate random samples based on training data distribution
                            random_samples = {}
                            for feature in feature_names:
                                if feature in st.session_state['X_train'].columns:
                                    mean = st.session_state['X_train'][feature].mean()
                                    std = st.session_state['X_train'][feature].std()
                                    random_samples[feature] = np.random.normal(mean, std, n_samples)
                                else:
                                    random_samples[feature] = np.zeros(n_samples)
                            
                            batch_df = pd.DataFrame(random_samples)
                            
                            # Scale if needed
                            if scaler is not None:
                                batch_scaled = scaler.transform(batch_df)
                                batch_df = pd.DataFrame(batch_scaled, columns=feature_names)
                            
                            # Make predictions
                            predictions = model.predict(batch_df)
                            
                            # Decode if needed
                            if le is not None:
                                predictions = le.inverse_transform(predictions.astype(int))
                            
                            # Add predictions to dataframe
                            batch_df['Prediction'] = predictions
                            
                            st.subheader("📊 Batch Predictions")
                            st.dataframe(batch_df)
                            
                            # Statistics
                            if le is None:  # Numerical predictions
                                st.subheader("📈 Prediction Statistics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean", f"{predictions.mean():.4f}")
                                with col2:
                                    st.metric("Std", f"{predictions.std():.4f}")
                                with col3:
                                    st.metric("Range", f"{predictions.max() - predictions.min():.4f}")
                            
                            # Download predictions
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Batch Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Batch prediction error: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
        
        with tab5:
            if 'results_df' not in st.session_state:
                st.info("ℹ️ Train some models first in the 'Model Training' tab")
                return
            
            try:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("📋 Machine Learning Report")
                
                results_df = st.session_state['results_df']
                config = st.session_state.get('ml_config', {})
                
                # Generate report
                report = f"""
                # Machine Learning Pipeline Report
                
                ## Configuration Summary
                - **Task Type:** {config.get('task_type', 'N/A')}
                - **Target Variable:** {config.get('target', 'N/A')}
                - **Number of Features:** {len(config.get('features', []))}
                - **Test Size:** {config.get('test_size', 0.2)*100:.0f}%
                - **Cross-Validation Folds:** {config.get('cv_folds', 5)}
                - **Feature Scaling:** {config.get('scaler', 'None')}
                
                ## Dataset Information
                - **Total Samples:** {st.session_state.get('X_train', pd.DataFrame()).shape[0] + st.session_state.get('X_test', pd.DataFrame()).shape[0]}
                - **Training Samples:** {st.session_state.get('X_train', pd.DataFrame()).shape[0]}
                - **Test Samples:** {st.session_state.get('X_test', pd.DataFrame()).shape[0]}
                
                ## Model Performance Summary
                
                {results_df.to_string()}
                
                ## Best Model
                **{results_df.iloc[0]['Model']}** performed best based on {'F1 Score' if config.get('task_type') == 'Classification' else 'R² Score'}.
                
                ## Recommendations
                """
                
                # Add recommendations based on results
                if config.get('task_type') == 'Classification':
                    if float(results_df['Accuracy'].iloc[0]) > 0.9:
                        report += "\n- ✓ Excellent model performance achieved"
                    elif float(results_df['Accuracy'].iloc[0]) > 0.7:
                        report += "\n- ✓ Good model performance"
                    else:
                        report += "\n- ⚠️ Model performance could be improved. Consider feature engineering or trying different algorithms"
                else:
                    if float(results_df['R² Score'].iloc[0]) > 0.8:
                        report += "\n- ✓ Excellent model performance achieved"
                    elif float(results_df['R² Score'].iloc[0]) > 0.6:
                        report += "\n- ✓ Good model performance"
                    else:
                        report += "\n- ⚠️ Model performance could be improved. Consider feature engineering or trying different algorithms"
                
                st.markdown(report)
                
                # Download report
                st.download_button(
                    label="📥 Download ML Report",
                    data=report,
                    file_name="ml_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Critical error in ML pipeline: {str(e)}")
        st.info("💡 Please check your data and try again. If the problem persists, try with a smaller dataset.")