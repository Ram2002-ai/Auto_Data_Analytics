import streamlit as st
import pandas as pd
import plotly.io as pio
import traceback
import sys
from datetime import datetime

from data_preprocessing import preprocess_data
from insights import generate_business_insights
from dataset_overview import eda_analysis  # Updated import
from visualization import auto_visualizations
from ml_pipeline import run_ml_pipeline
from statistical_analysis import statistical_analysis
from data_quality import quality_report
from chatbot import data_chatbot

# Set plotly template
pio.templates.default = "plotly_white"

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="AI Data Analyst Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# CUSTOM ERROR HANDLER
# ---------------------------------------

class StreamlitExceptionHandler:
    """Custom exception handler for Streamlit"""
    
    @staticmethod
    def handle_exception(e, context="application"):
        """Handle exceptions with user-friendly messages"""
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Create user-friendly error message
        user_message = f"""
        ### ❌ An error occurred in the {context}
        
        **Error Type:** {error_type}
        
        **What happened:** {error_msg if error_msg else "An unexpected error occurred"}
        
        **Possible solutions:**
        """
        
        # Add specific solutions based on error type
        if "MemoryError" in error_type:
            user_message += """
            - Your dataset might be too large. Try uploading a smaller file.
            - Close other applications to free up memory.
            - Consider sampling your data before uploading.
            """
        elif "KeyError" in error_type or "IndexError" in error_type:
            user_message += """
            - The requested column or index doesn't exist in your dataset.
            - Check if you've selected valid columns for the operation.
            - Try refreshing the page and uploading your data again.
            """
        elif "ValueError" in error_type:
            user_message += """
            - The data values don't match the expected format.
            - Check for invalid values in your dataset (e.g., text in numeric columns).
            - Ensure your data types are correct for the selected operation.
            """
        elif "TypeError" in error_type:
            user_message += """
            - There's a mismatch in data types.
            - Check if you're mixing numeric and text data in operations.
            - Use the preprocessing tab to convert data types appropriately.
            """
        elif "FileNotFoundError" in error_type:
            user_message += """
            - The file couldn't be found. Please upload it again.
            - Check if the file path is correct.
            """
        elif "PermissionError" in error_type:
            user_message += """
            - Permission denied when accessing the file.
            - Make sure the file isn't open in another program.
            """
        elif "pd.errors.EmptyDataError" in error_type:
            user_message += """
            - The uploaded file is empty.
            - Please upload a file containing data.
            """
        elif "pd.errors.ParserError" in error_type:
            user_message += """
            - Couldn't parse the file. Check if it's a valid CSV or Excel file.
            - Ensure the file format matches the selected file type.
            """
        else:
            user_message += """
            - Try refreshing the page and uploading your data again.
            - Check if your data format is compatible with the operation.
            - If the problem persists, try with a smaller sample of your data.
            """
        
        # Add technical details in an expander for debugging
        user_message += f"""
        
        **Technical Details:**
        """
        
        return user_message

# Initialize session state for error tracking
if "error_log" not in st.session_state:
    st.session_state.error_log = []

if "last_successful_operation" not in st.session_state:
    st.session_state.last_successful_operation = None

# ---------------------------------------
# ADVANCED CSS WITH RESPONSIVE DESIGN
# ---------------------------------------

st.markdown("""
<style>
    /* Global Styles */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        animation: fadeInUp 1s;
    }
    
    /* Card Styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Error Message Styling */
    .error-container {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff4757 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(255, 71, 87, 0.3);
        animation: slideInRight 0.5s;
    }
    
    .error-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .error-solution {
        background: rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    /* Success Message Styling */
    .success-container {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: fadeInUp 0.5s;
    }
    
    /* Warning Message Styling */
    .warning-container {
        background: linear-gradient(135deg, #ffd43b 0%, #fcc419 100%);
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: fadeInUp 0.5s;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Chatbot Styling */
    .chat-container {
        max-width: 800px;
        margin: 2rem auto;
        background: #f8f9fa;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        animation: slideInRight 0.5s;
    }
    
    .bot-message {
        background: white;
        color: #2c3e50;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: slideInLeft 0.5s;
    }
    
    /* Loading Spinner */
    .custom-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.8rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .user-message, .bot-message {
            max-width: 95%;
        }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 30px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 25px;
        padding: 0.5rem 2rem;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# HEADER WITH ANIMATION
# ---------------------------------------

st.markdown("""
<div class="header-container">
    <div class="header-title">📊 AI Data Analyst Pro</div>
    <div class="header-subtitle">Intelligent Data Analysis & Visualization Platform</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------

if "data" not in st.session_state:
    st.session_state.data = None
    
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
    
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "upload_error" not in st.session_state:
    st.session_state.upload_error = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "operation_status" not in st.session_state:
    st.session_state.operation_status = {}

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

def safe_dataframe_operation(func, df, *args, **kwargs):
    """Safely execute dataframe operations with error handling"""
    try:
        result = func(df, *args, **kwargs)
        st.session_state.last_successful_operation = func.__name__
        return result, None
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, func.__name__)
        return None, error_msg

def validate_dataset(df):
    """Validate dataset for common issues"""
    issues = []
    
    if df.empty:
        issues.append("The dataset is empty")
    
    if df.shape[0] == 0:
        issues.append("No rows in the dataset")
    
    if df.shape[1] == 0:
        issues.append("No columns in the dataset")
    
    # Check for memory issues
    memory_usage = df.memory_usage(deep=True).sum() / 1024**3  # GB
    if memory_usage > 1:
        issues.append(f"Large dataset detected ({memory_usage:.2f} GB). Some operations may be slow.")
    
    # Check for mixed types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column has mixed types
            types = df[col].apply(type).unique()
            if len(types) > 1:
                issues.append(f"Column '{col}' has mixed data types: {types}")
    
    return issues

def show_validation_warnings(issues):
    """Display validation warnings"""
    if issues:
        st.markdown("""
        <div class="warning-container">
            <strong>⚠️ Data Quality Warnings:</strong><br>
        """ + "<br>".join([f"• {issue}" for issue in issues]) + """
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------
# SIDEBAR WITH ENHANCED NAVIGATION
# ---------------------------------------

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    # Custom radio buttons styling
    page = st.radio(
        "Select Module",
        ["📤 Upload Dataset", "🛠️ Preprocessing", "🔍 EDA", 
         "📈 Visualization", "🤖 Machine Learning", "💡 Insights", 
         "💬 Chatbot", "📋 Data Quality", "📐 Statistical Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Dataset info in sidebar
    if st.session_state.data is not None:
        st.markdown("### 📂 Current Dataset")
        df = st.session_state.data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        
        # Show data quality indicator
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct == 0:
            st.success("✅ Data Quality: Excellent")
        elif missing_pct < 5:
            st.info(f"ℹ️ Data Quality: Good ({missing_pct:.1f}% missing)")
        elif missing_pct < 20:
            st.warning(f"⚠️ Data Quality: Fair ({missing_pct:.1f}% missing)")
        else:
            st.error(f"❌ Data Quality: Poor ({missing_pct:.1f}% missing)")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Data", use_container_width=True):
                st.session_state.data = None
                st.session_state.processed_data = None
                st.session_state.data_loaded = False
                st.rerun()
        
        with col2:
            if st.button("📥 Download Sample", use_container_width=True):
                # Create sample data download
                sample_df = df.head(100)
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="Download Sample",
                    data=csv,
                    file_name="sample_data.csv",
                    mime="text/csv"
                )
        
        # Show operation history
        if st.session_state.operation_status:
            with st.expander("📋 Operation History"):
                for op, status in st.session_state.operation_status.items():
                    if status == "success":
                        st.success(f"✅ {op}")
                    elif status == "error":
                        st.error(f"❌ {op}")
                    else:
                        st.info(f"⏳ {op}")
    else:
        st.info("👆 Upload a dataset to get started")

# ---------------------------------------
# MAIN CONTENT AREA
# ---------------------------------------

# Map page names to functions
page_map = {
    "📤 Upload Dataset": "upload",
    "🛠️ Preprocessing": "preprocess",
    "🔍 EDA": "eda",
    "📈 Visualization": "visualization",
    "🤖 Machine Learning": "ml",
    "💡 Insights": "insights",
    "💬 Chatbot": "chatbot",
    "📋 Data Quality": "quality",
    "📐 Statistical Analysis": "statistical"
}

current_page = page_map[page]

# ---------------------------------------
# UPLOAD DATASET PAGE
# ---------------------------------------

if current_page == "upload":
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Upload Your Dataset")
        
        # File uploader with size limit warning
        file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx"],
            help="Maximum recommended file size: 200MB. Larger files may cause performance issues."
        )
        
        if file:
            try:
                # Check file size
                file_size = file.size / 1024**2  # MB
                if file_size > 200:
                    st.warning(f"⚠️ Large file detected ({file_size:.2f} MB). Processing may be slow.")
                
                with st.spinner("📂 Loading file..."):
                    # Read file based on extension
                    if file.name.endswith("csv"):
                        # Try different encodings
                        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                        df = None
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file, encoding=encoding)
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if df is None:
                            st.error("❌ Could not read CSV file with any common encoding.")
                            st.stop()
                    
                    elif file.name.endswith(("xlsx", "xls")):
                        try:
                            df = pd.read_excel(file)
                        except Exception as e:
                            st.error(f"❌ Error reading Excel file: {str(e)}")
                            st.info("💡 Try saving the file as CSV and uploading again.")
                            st.stop()
                    
                    # Validate dataset
                    issues = validate_dataset(df)
                    show_validation_warnings(issues)
                    
                    if not issues or all("Large dataset" not in issue for issue in issues):
                        # Store in session state
                        st.session_state.data = df
                        st.session_state.uploaded_file_name = file.name
                        st.session_state.data_loaded = True
                        st.session_state.upload_error = None
                        
                        # Show success message
                        st.markdown("""
                        <div class="success-container">
                            <strong>✅ Successfully loaded:</strong> {}<br>
                            <strong>📊 Shape:</strong> {} rows × {} columns
                        </div>
                        """.format(file.name, df.shape[0], df.shape[1]), unsafe_allow_html=True)
                        
                        # File statistics
                        st.markdown("### 📊 File Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", f"{df.shape[0]:,}")
                        with col2:
                            st.metric("Total Columns", df.shape[1])
                        with col3:
                            memory = df.memory_usage(deep=True).sum() / 1024**2
                            st.metric("Memory Usage", f"{memory:.2f} MB")
                        
                        # Data preview with scroll
                        st.markdown("### 👁️ Data Preview")
                        st.dataframe(
                            df.head(10),
                            use_container_width=True,
                            height=300
                        )
                        
                        # Column info with sorting
                        st.markdown("### 📋 Column Information")
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Type': df.dtypes.astype(str),
                            'Non-Null Count': df.count().values,
                            'Null Count': df.isnull().sum().values,
                            'Null %': (df.isnull().sum().values / len(df) * 100).round(2),
                            'Unique Values': [df[col].nunique() for col in df.columns]
                        })
                        
                        # Sort by null count
                        col_info = col_info.sort_values('Null %', ascending=False)
                        
                        st.dataframe(
                            col_info.style.background_gradient(subset=['Null %'], cmap='YlOrRd'),
                            use_container_width=True
                        )
                        
                        # Quick stats
                        st.markdown("### 📈 Quick Statistics")
                        
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            st.dataframe(
                                df[numeric_cols].describe(),
                                use_container_width=True
                            )
                        
                        # Navigation buttons
                        st.markdown("### 🚀 Next Steps")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("🛠️ Go to Preprocessing", use_container_width=True):
                                st.session_state.page = "🛠️ Preprocessing"
                                st.rerun()
                        
                        with col2:
                            if st.button("📊 Go to EDA", use_container_width=True):
                                st.session_state.page = "📊 EDA"
                                st.rerun()
                        
                        with col3:
                            if st.button("📈 Go to Visualization", use_container_width=True):
                                st.session_state.page = "📈 Visualization"
                                st.rerun()
            
            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded file is empty. Please upload a file with data.")
            except pd.errors.ParserError as e:
                st.error(f"❌ Error parsing file: {str(e)}")
                st.info("💡 Check if your CSV file has consistent delimiters and quoting.")
            except MemoryError:
                st.error("❌ Out of memory! The file is too large to process.")
                st.info("💡 Try uploading a smaller file or sampling your data first.")
            except Exception as e:
                error_msg = StreamlitExceptionHandler.handle_exception(e, "file upload")
                st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)
                
                # Log error
                st.session_state.error_log.append({
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample data option
        with st.expander("🔄 Or use sample data"):
            st.markdown("Don't have a dataset? Try our sample data:")
            
            if st.button("Load Sample Dataset", use_container_width=True):
                try:
                    from utils import create_sample_dataset
                    sample_df = create_sample_dataset()
                    st.session_state.data = sample_df
                    st.session_state.uploaded_file_name = "sample_dataset.csv"
                    st.session_state.data_loaded = True
                    st.success("✅ Sample dataset loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")

# ---------------------------------------
# PREPROCESSING PAGE
# ---------------------------------------

elif current_page == "preprocess":
    try:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Validate data before preprocessing
            issues = validate_dataset(df)
            if issues:
                show_validation_warnings(issues)
            
            # Run preprocessing with error handling
            with st.spinner("🔄 Preprocessing data..."):
                processed_df, error = safe_dataframe_operation(preprocess_data, df)
                
                if error:
                    st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                    st.session_state.operation_status['Preprocessing'] = 'error'
                else:
                    st.session_state.processed_data = processed_df
                    st.session_state.operation_status['Preprocessing'] = 'success'
                    
                    # Show success message
                    st.markdown("""
                    <div class="success-container">
                        <strong>✅ Preprocessing completed successfully!</strong><br>
                        You can now proceed to analysis or visualization.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "preprocessing")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# EDA PAGE
# ---------------------------------------

elif current_page == "eda":
    try:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Validate data
            issues = validate_dataset(df)
            if issues:
                show_validation_warnings(issues)
            
            # Run EDA with error handling
            with st.spinner("🔍 Performing Exploratory Data Analysis..."):
                result, error = safe_dataframe_operation(eda_analysis, df)
                
                if error:
                    st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                    st.session_state.operation_status['EDA'] = 'error'
                else:
                    st.session_state.operation_status['EDA'] = 'success'
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "EDA")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# VISUALIZATION PAGE
# ---------------------------------------

elif current_page == "visualization":
    try:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Validate data
            issues = validate_dataset(df)
            if issues:
                show_validation_warnings(issues)
            
            # Run visualization with error handling
            with st.spinner("📊 Generating visualizations..."):
                result, error = safe_dataframe_operation(auto_visualizations, df)
                
                if error:
                    st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                    st.session_state.operation_status['Visualization'] = 'error'
                else:
                    st.session_state.operation_status['Visualization'] = 'success'
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "visualization")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# MACHINE LEARNING PAGE
# ---------------------------------------

elif current_page == "ml":
    try:
        if st.session_state.data is not None:
            data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
            
            # Validate data for ML
            if data_to_use.shape[0] < 10:
                st.warning("⚠️ Dataset too small for machine learning (need at least 10 rows)")
            else:
                # Run ML pipeline with error handling
                with st.spinner("🤖 Running machine learning pipeline..."):
                    result, error = safe_dataframe_operation(run_ml_pipeline, data_to_use)
                    
                    if error:
                        st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                        st.session_state.operation_status['ML'] = 'error'
                    else:
                        st.session_state.operation_status['ML'] = 'success'
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "machine learning")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# INSIGHTS PAGE
# ---------------------------------------

elif current_page == "insights":
    try:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Generate insights with error handling
            with st.spinner("💡 Generating business insights..."):
                result, error = safe_dataframe_operation(generate_business_insights, df)
                
                if error:
                    st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                    st.session_state.operation_status['Insights'] = 'error'
                else:
                    st.session_state.operation_status['Insights'] = 'success'
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "insights generation")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# CHATBOT PAGE
# ---------------------------------------

elif current_page == "chatbot":
    try:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Run chatbot with error handling
            with st.spinner("🤖 Initializing chatbot..."):
                result, error = safe_dataframe_operation(data_chatbot, df)
                
                if error:
                    st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                    st.session_state.operation_status['Chatbot'] = 'error'
                else:
                    st.session_state.operation_status['Chatbot'] = 'success'
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "chatbot")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# DATA QUALITY PAGE
# ---------------------------------------

elif current_page == "quality":
    try:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Run quality report with error handling
            with st.spinner("📋 Generating quality report..."):
                from data_quality import quality_report
                result, error = safe_dataframe_operation(quality_report, df)
                
                if error:
                    st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                    st.session_state.operation_status['Data Quality'] = 'error'
                else:
                    st.session_state.operation_status['Data Quality'] = 'success'
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "data quality")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# STATISTICAL ANALYSIS PAGE
# ---------------------------------------

elif current_page == "statistical":
    try:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Validate numeric data
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                st.warning("⚠️ No numeric columns found. Statistical analysis requires numeric data.")
            else:
                # Run statistical analysis with error handling
                with st.spinner("📐 Performing statistical analysis..."):
                    from statistical_analysis import statistical_analysis
                    result, error = safe_dataframe_operation(statistical_analysis, df)
                    
                    if error:
                        st.markdown(f'<div class="error-container">{error}</div>', unsafe_allow_html=True)
                        st.session_state.operation_status['Statistical Analysis'] = 'error'
                    else:
                        st.session_state.operation_status['Statistical Analysis'] = 'success'
        else:
            st.warning("⚠️ Please upload a dataset first in the Upload section")
    except Exception as e:
        error_msg = StreamlitExceptionHandler.handle_exception(e, "statistical analysis")
        st.markdown(f'<div class="error-container">{error_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------
# ERROR LOG DISPLAY (Hidden by default)
# ---------------------------------------

if st.session_state.error_log and st.checkbox("🔧 Show Error Log (Debug Mode)"):
    st.markdown("### 📋 Error Log")
    for i, error_entry in enumerate(st.session_state.error_log[-5:]):  # Show last 5 errors
        with st.expander(f"Error {i+1}: {error_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            st.code(error_entry['error'])
            st.code(error_entry['traceback'])

# ---------------------------------------
# FOOTER
# ---------------------------------------

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit | Version 2.0 | Enhanced Error Handling</p>",
    unsafe_allow_html=True
)