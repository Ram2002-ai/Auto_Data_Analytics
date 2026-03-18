import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

def preprocess_data(df):
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>⚙️ Data Preprocessing Pipeline</h2>
        <p style='color: gray;'>Clean, transform, and prepare your data for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different preprocessing steps
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🧹 Clean Data", "🔄 Transform", 
        "📏 Scale & Encode", "📈 Feature Engineering"
    ])
    
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Rows", df.shape[0])
        with col2:
            st.metric("Original Columns", df.shape[1])
        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Data quality before preprocessing
        st.subheader("Data Quality Check")
        
        quality_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(quality_df, use_container_width=True)
        
        # Visualize missing values
        if df.isnull().sum().sum() > 0:
            st.subheader("Missing Value Heatmap")
            missing_df = df.isnull().astype(int)
            fig = px.imshow(missing_df.T, 
                          color_continuous_scale='reds',
                          aspect="auto",
                          title="Missing Values Pattern")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🧹 Data Cleaning Options")
        
        # Create a copy for processing
        processed_df = df.copy()
        
        # Remove duplicates
        st.markdown("### Duplicate Removal")
        duplicates = processed_df.duplicated().sum()
        st.write(f"Duplicate rows found: **{duplicates}**")
        
        if duplicates > 0:
            if st.button("Remove Duplicates", use_container_width=True):
                processed_df = processed_df.drop_duplicates()
                st.success(f"✅ Removed {duplicates} duplicate rows")
        
        # Handle missing values
        st.markdown("### Missing Value Handling")
        
        missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
        
        if missing_cols:
            selected_col = st.selectbox("Select column to handle missing values", missing_cols)
            
            col_type = processed_df[selected_col].dtype
            
            if pd.api.types.is_numeric_dtype(processed_df[selected_col]):
                method = st.radio(
                    "Choose imputation method",
                    ["Mean", "Median", "Mode", "KNN Imputer", "Drop rows", "Fill with value"]
                )
                
                if method == "Mean":
                    processed_df[selected_col].fillna(processed_df[selected_col].mean(), inplace=True)
                elif method == "Median":
                    processed_df[selected_col].fillna(processed_df[selected_col].median(), inplace=True)
                elif method == "Mode":
                    processed_df[selected_col].fillna(processed_df[selected_col].mode()[0], inplace=True)
                elif method == "KNN Imputer":
                    st.info("KNN Imputer will be applied to all numeric columns")
                    if st.button("Apply KNN Imputer"):
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        imputer = KNNImputer(n_neighbors=5)
                        processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])
                elif method == "Drop rows":
                    if st.button(f"Drop rows with missing values in {selected_col}"):
                        processed_df = processed_df.dropna(subset=[selected_col])
                else:
                    fill_value = st.text_input("Enter fill value")
                    if fill_value:
                        if pd.api.types.is_numeric_dtype(processed_df[selected_col]):
                            processed_df[selected_col].fillna(float(fill_value), inplace=True)
                        else:
                            processed_df[selected_col].fillna(fill_value, inplace=True)
            
            else:  # Categorical column
                method = st.radio(
                    "Choose imputation method",
                    ["Mode", "Drop rows", "Fill with value"]
                )
                
                if method == "Mode":
                    processed_df[selected_col].fillna(processed_df[selected_col].mode()[0], inplace=True)
                elif method == "Drop rows":
                    if st.button(f"Drop rows with missing values in {selected_col}"):
                        processed_df = processed_df.dropna(subset=[selected_col])
                else:
                    fill_value = st.text_input("Enter fill value")
                    if fill_value:
                        processed_df[selected_col].fillna(fill_value, inplace=True)
        else:
            st.success("✅ No missing values found!")
        
        # Outlier detection
        st.markdown("### Outlier Detection")
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            selected_num = st.selectbox("Select numeric column for outlier detection", numeric_cols)
            
            # Calculate IQR
            Q1 = processed_df[selected_num].quantile(0.25)
            Q3 = processed_df[selected_num].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = processed_df[
                (processed_df[selected_num] < Q1 - 1.5 * IQR) | 
                (processed_df[selected_num] > Q3 + 1.5 * IQR)
            ]
            
            st.write(f"Outliers detected: **{len(outliers)}** rows")
            
            if len(outliers) > 0:
                if st.button(f"Remove outliers from {selected_num}"):
                    processed_df = processed_df[
                        (processed_df[selected_num] >= Q1 - 1.5 * IQR) & 
                        (processed_df[selected_num] <= Q3 + 1.5 * IQR)
                    ]
                    st.success(f"✅ Removed {len(outliers)} outliers")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state
        st.session_state.data = processed_df
        
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔄 Data Transformations")
        
        processed_df = st.session_state.data.copy() if 'processed_df' not in locals() else processed_df
        
        # Column operations
        st.markdown("### Column Operations")
        
        operation = st.selectbox(
            "Choose operation",
            ["Create new column", "Rename column", "Drop column", "Change data type"]
        )
        
        if operation == "Create new column":
            col1, col2, col3 = st.columns(3)
            with col1:
                new_col_name = st.text_input("New column name")
            with col2:
                col_to_use = st.selectbox("Based on column", processed_df.columns)
            with col3:
                operation_type = st.selectbox(
                    "Operation",
                    ["Square", "Square Root", "Log", "Absolute", "Round", "Binary encode"]
                )
            
            if st.button("Create column") and new_col_name:
                if operation_type == "Square":
                    processed_df[new_col_name] = processed_df[col_to_use] ** 2
                elif operation_type == "Square Root":
                    processed_df[new_col_name] = np.sqrt(processed_df[col_to_use])
                elif operation_type == "Log":
                    processed_df[new_col_name] = np.log1p(processed_df[col_to_use])
                elif operation_type == "Absolute":
                    processed_df[new_col_name] = np.abs(processed_df[col_to_use])
                elif operation_type == "Round":
                    processed_df[new_col_name] = np.round(processed_df[col_to_use])
                elif operation_type == "Binary encode":
                    threshold = st.number_input("Threshold for binary encoding")
                    processed_df[new_col_name] = (processed_df[col_to_use] > threshold).astype(int)
                
                st.success(f"✅ Created column: {new_col_name}")
        
        elif operation == "Rename column":
            col_to_rename = st.selectbox("Select column to rename", processed_df.columns)
            new_name = st.text_input("New column name")
            
            if st.button("Rename") and new_name:
                processed_df.rename(columns={col_to_rename: new_name}, inplace=True)
                st.success(f"✅ Renamed {col_to_rename} to {new_name}")
        
        elif operation == "Drop column":
            cols_to_drop = st.multiselect("Select columns to drop", processed_df.columns)
            
            if st.button("Drop columns") and cols_to_drop:
                processed_df = processed_df.drop(columns=cols_to_drop)
                st.success(f"✅ Dropped columns: {', '.join(cols_to_drop)}")
        
        elif operation == "Change data type":
            col_to_change = st.selectbox("Select column", processed_df.columns)
            new_type = st.selectbox(
                "New data type",
                ["int", "float", "str", "datetime", "category"]
            )
            
            if st.button("Change type"):
                try:
                    if new_type == "int":
                        processed_df[col_to_change] = processed_df[col_to_change].astype(int)
                    elif new_type == "float":
                        processed_df[col_to_change] = processed_df[col_to_change].astype(float)
                    elif new_type == "str":
                        processed_df[col_to_change] = processed_df[col_to_change].astype(str)
                    elif new_type == "datetime":
                        processed_df[col_to_change] = pd.to_datetime(processed_df[col_to_change])
                    elif new_type == "category":
                        processed_df[col_to_change] = processed_df[col_to_change].astype('category')
                    
                    st.success(f"✅ Changed {col_to_change} to {new_type}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state
        st.session_state.data = processed_df
    
    with tab4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📏 Feature Scaling & Encoding")
        
        processed_df = st.session_state.data.copy() if 'processed_df' not in locals() else processed_df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Feature Scaling")
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                scale_cols = st.multiselect("Select columns to scale", numeric_cols)
                scale_method = st.radio("Scaling method", ["StandardScaler", "MinMaxScaler"])
                
                if st.button("Apply Scaling") and scale_cols:
                    if scale_method == "StandardScaler":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    
                    processed_df[scale_cols] = scaler.fit_transform(processed_df[scale_cols])
                    st.success(f"✅ Applied {scale_method} to {len(scale_cols)} columns")
        
        with col2:
            st.markdown("### Categorical Encoding")
            cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                encode_cols = st.multiselect("Select columns to encode", cat_cols)
                encode_method = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding"])
                
                if st.button("Apply Encoding") and encode_cols:
                    if encode_method == "Label Encoding":
                        for col in encode_cols:
                            le = LabelEncoder()
                            processed_df[col + '_encoded'] = le.fit_transform(processed_df[col])
                        st.success(f"✅ Applied Label Encoding to {len(encode_cols)} columns")
                    else:
                        processed_df = pd.get_dummies(processed_df, columns=encode_cols)
                        st.success(f"✅ Applied One-Hot Encoding to {len(encode_cols)} columns")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state
        st.session_state.data = processed_df
    
    with tab5:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📈 Feature Engineering")
        
        processed_df = st.session_state.data.copy() if 'processed_df' not in locals() else processed_df
        
        # Feature interactions
        st.markdown("### Feature Interactions")
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                feat1 = st.selectbox("First feature", numeric_cols)
            with col2:
                feat2 = st.selectbox("Second feature", [c for c in numeric_cols if c != feat1])
            
            interaction_type = st.selectbox(
                "Interaction type",
                ["Multiplication", "Addition", "Subtraction", "Division", "Ratio"]
            )
            
            new_col_name = st.text_input("New column name", f"{feat1}_{interaction_type}_{feat2}")
            
            if st.button("Create Interaction Feature"):
                if interaction_type == "Multiplication":
                    processed_df[new_col_name] = processed_df[feat1] * processed_df[feat2]
                elif interaction_type == "Addition":
                    processed_df[new_col_name] = processed_df[feat1] + processed_df[feat2]
                elif interaction_type == "Subtraction":
                    processed_df[new_col_name] = processed_df[feat1] - processed_df[feat2]
                elif interaction_type == "Division":
                    processed_df[new_col_name] = processed_df[feat1] / (processed_df[feat2] + 1e-8)
                elif interaction_type == "Ratio":
                    processed_df[new_col_name] = processed_df[feat1] / (processed_df[feat2].sum() + 1e-8)
                
                st.success(f"✅ Created feature: {new_col_name}")
        
        # Binning
        st.markdown("### Feature Binning")
        if numeric_cols:
            bin_col = st.selectbox("Select column for binning", numeric_cols)
            n_bins = st.slider("Number of bins", 2, 20, 5)
            bin_labels = [f"Bin_{i}" for i in range(n_bins)]
            
            if st.button("Create Binned Feature"):
                processed_df[bin_col + '_binned'] = pd.cut(processed_df[bin_col], 
                                                          bins=n_bins, 
                                                          labels=bin_labels)
                st.success(f"✅ Created binned feature: {bin_col}_binned")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state
        st.session_state.data = processed_df
    
    # Preview processed data
    st.markdown("---")
    st.subheader("📋 Processed Data Preview")
    
    data_to_show = st.session_state.data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Rows", data_to_show.shape[0])
    with col2:
        st.metric("Final Columns", data_to_show.shape[1])
    with col3:
        final_missing = data_to_show.isnull().sum().sum()
        st.metric("Remaining Missing", final_missing)
    
    st.dataframe(data_to_show.head(10), use_container_width=True)
    
    # Download processed data
    csv = data_to_show.to_csv(index=False)
    st.download_button(
        label="📥 Download Processed Data",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    return data_to_show