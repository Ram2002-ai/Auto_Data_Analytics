import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta

def data_chatbot(df):
    """
    Advanced chatbot that provides data access and visualizations based on user questions
    """
    
    st.markdown("""
    <style>
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .chat-header h2 {
        font-size: 2.2rem;
        margin-bottom: 10px;
    }
    .chat-header p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        border-left: 4px solid #1976d2;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .bot-message {
        background: white;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        border-left: 4px solid #4caf50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    .viz-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .insight-badge {
        background: #4caf50;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        display: inline-block;
        margin-right: 5px;
    }
    </style>
    
    <div class="chat-header">
        <h2>🤖 Smart Data Assistant</h2>
        <p>Ask questions and get instant visualizations - I'll show you the data!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "last_viz" not in st.session_state:
        st.session_state.last_viz = None
    
    if "last_data" not in st.session_state:
        st.session_state.last_data = None
    
    # Main layout
    main_col, viz_col = st.columns([1, 1])
    
    with main_col:
        # Chat history
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_messages:
                st.info("""
                👋 **Hi! I can show you data and create visualizations. Try asking:**
                
                **📊 Show Data:**
                • "Show me the first 10 rows"
                • "Show me data where age > 30"
                • "Display top 5 by sales"
                
                **📈 Create Visualizations:**
                • "Show me a bar chart of category"
                • "Plot histogram of age"
                • "Create scatter plot of price vs quantity"
                • "Show trend of sales over time"
                
                **🔍 Analyze:**
                • "What's the average of salary?"
                • "Show statistics for all columns"
                • "Find outliers in price"
                """)
            
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-message"><b>👤 You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # Input area
        st.markdown("<br>", unsafe_allow_html=True)
        input_col1, input_col2 = st.columns([5, 1])
        
        with input_col1:
            user_query = st.text_input("", placeholder="💬 Ask a question or request a visualization...", 
                                       key="chat_input", label_visibility="collapsed")
        
        with input_col2:
            send_button = st.button("📤 Ask", use_container_width=True)
        
        if send_button and user_query:
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            
            # Process query and get response with data/viz
            with st.spinner("🔍 Processing your request..."):
                response, viz_data, table_data = process_query_with_viz(user_query, df)
            
            # Add bot response
            st.session_state.chat_messages.append({"role": "bot", "content": response})
            
            # Store visualization and data for display
            if viz_data:
                st.session_state.last_viz = viz_data
            if table_data is not None:
                st.session_state.last_data = table_data
            
            st.rerun()
    
    with viz_col:
        # Display visualizations and data
        if st.session_state.last_viz:
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Generated Visualization")
            display_visualization(st.session_state.last_viz)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.last_data is not None:
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.markdown("### 📋 Data Result")
            st.dataframe(st.session_state.last_data, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("### 🔍 Quick Actions")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    actions = [
        ("📊 First 10 Rows", "Show me first 10 rows", col1),
        ("📈 Bar Chart", "Show bar chart of first categorical column", col2),
        ("📉 Histogram", "Plot histogram of first numeric column", col3),
        ("🔎 Filter", "Show rows where value > average", col4),
        ("📋 Statistics", "Show me statistics", col5)
    ]
    
    for label, query, col in actions:
        if col.button(label, use_container_width=True):
            st.session_state.chat_messages.append({"role": "user", "content": query})
            response, viz_data, table_data = process_query_with_viz(query, df)
            st.session_state.chat_messages.append({"role": "bot", "content": response})
            if viz_data:
                st.session_state.last_viz = viz_data
            if table_data is not None:
                st.session_state.last_data = table_data
            st.rerun()
    
    # Clear button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat & Visualizations", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.last_viz = None
            st.session_state.last_data = None
            st.rerun()


def process_query_with_viz(query, df):
    """Process query and return response with visualization and data"""
    query_lower = query.lower().strip()
    
    # Get column information
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Extract numbers from query
    numbers = re.findall(r'\d+', query_lower)
    n = int(numbers[0]) if numbers else 10
    
    # 1. SHOW DATA - First/Last/Random rows
    if any(word in query_lower for word in ['first', 'head', 'top']):
        return show_first_rows(df, n)
    
    elif any(word in query_lower for word in ['last', 'tail', 'bottom']):
        return show_last_rows(df, n)
    
    elif 'random' in query_lower or 'sample' in query_lower:
        return show_random_rows(df, n)
    
    # 2. FILTER DATA
    elif any(word in query_lower for word in ['find', 'where', 'filter', 'search', 'with']):
        return filter_data(query_lower, df)
    
    # 3. SORT DATA
    elif 'sort' in query_lower or 'order by' in query_lower:
        return sort_data(query_lower, df)
    
    # 4. BAR CHART
    elif any(word in query_lower for word in ['bar chart', 'bar plot', 'bar graph', 'count plot']):
        return create_bar_chart(query_lower, df, categorical_cols)
    
    # 5. HISTOGRAM
    elif any(word in query_lower for word in ['histogram', 'distribution', 'hist', 'frequency']):
        return create_histogram(query_lower, df, numeric_cols)
    
    # 6. SCATTER PLOT
    elif any(word in query_lower for word in ['scatter', 'scatter plot', 'scatterplot', 'relationship']):
        return create_scatter_plot(query_lower, df, numeric_cols)
    
    # 7. LINE CHART / TREND
    elif any(word in query_lower for word in ['line chart', 'line plot', 'trend', 'over time']):
        return create_line_chart(query_lower, df, numeric_cols, datetime_cols)
    
    # 8. BOX PLOT
    elif any(word in query_lower for word in ['box plot', 'boxplot', 'box', 'outliers']):
        return create_box_plot(query_lower, df, numeric_cols, categorical_cols)
    
    # 9. PIE CHART
    elif any(word in query_lower for word in ['pie chart', 'pie', 'proportion', 'percentage']):
        return create_pie_chart(query_lower, df, categorical_cols)
    
    # 10. HEATMAP / CORRELATION
    elif any(word in query_lower for word in ['heatmap', 'correlation', 'corr', 'heat map']):
        return create_heatmap(df, numeric_cols)
    
    # 11. VIOLIN PLOT
    elif 'violin' in query_lower:
        return create_violin_plot(query_lower, df, numeric_cols, categorical_cols)
    
    # 12. STATISTICS
    elif any(word in query_lower for word in ['statistics', 'stats', 'describe', 'summary']):
        return show_statistics(query_lower, df, numeric_cols, all_cols)
    
    # 13. COLUMN INFORMATION
    elif any(word in query_lower for word in ['column info', 'column details', 'info about']):
        return show_column_info(query_lower, df, all_cols)
    
    # 14. MISSING VALUES
    elif any(word in query_lower for word in ['missing', 'null', 'na', 'empty']):
        return show_missing_values(df)
    
    # 15. OUTLIERS
    elif 'outlier' in query_lower:
        return detect_outliers(query_lower, df, numeric_cols)
    
    # 16. UNIQUE VALUES
    elif any(word in query_lower for word in ['unique', 'distinct', 'categories']):
        return show_unique_values(query_lower, df, all_cols, categorical_cols)
    
    # 17. COMPARE COLUMNS
    elif 'compare' in query_lower:
        return compare_columns(query_lower, df, numeric_cols, categorical_cols)
    
    # 18. HELP
    elif any(word in query_lower for word in ['help', 'what can you do', 'capabilities']):
        return show_help(), None, None
    
    # 19. DEFAULT - Try to understand if asking about a specific column
    else:
        return handle_general_query(query_lower, df, numeric_cols, categorical_cols, all_cols)


def show_first_rows(df, n=10):
    """Show first n rows"""
    data = df.head(n)
    response = f"### 👁️ First {n} Rows\n\nHere's the data you requested:"
    return response, None, data


def show_last_rows(df, n=10):
    """Show last n rows"""
    data = df.tail(n)
    response = f"### 👁️ Last {n} Rows\n\nHere's the data you requested:"
    return response, None, data


def show_random_rows(df, n=5):
    """Show random n rows"""
    data = df.sample(min(n, len(df)))
    response = f"### 🎲 Random Sample of {n} Rows\n\nHere's a random sample from your data:"
    return response, None, data


def filter_data(query, df):
    """Filter data based on conditions"""
    # Common patterns
    patterns = [
        (r'(\w+)\s*>\s*(\d+\.?\d*)', '>'),
        (r'(\w+)\s*<\s*(\d+\.?\d*)', '<'),
        (r'(\w+)\s*>=\s*(\d+\.?\d*)', '>='),
        (r'(\w+)\s*<=\s*(\d+\.?\d*)', '<='),
        (r'(\w+)\s*=\s*(\d+\.?\d*)', '=='),
        (r'(\w+)\s*==\s*(\d+\.?\d*)', '=='),
        (r'(\w+)\s*contains\s*["\']?([^"\']+)["\']?', 'contains'),
        (r'(\w+)\s*is\s*["\']?([^"\']+)["\']?', '=='),
    ]
    
    for pattern, op in patterns:
        match = re.search(pattern, query.lower())
        if match:
            col = match.group(1)
            val = match.group(2)
            
            # Find matching column
            for c in df.columns:
                if c.lower() == col:
                    try:
                        if op in ['>', '<', '>=', '<=']:
                            val = float(val)
                            if op == '>':
                                filtered = df[df[c] > val]
                                condition = f"{c} > {val}"
                            elif op == '<':
                                filtered = df[df[c] < val]
                                condition = f"{c} < {val}"
                            elif op == '>=':
                                filtered = df[df[c] >= val]
                                condition = f"{c} >= {val}"
                            elif op == '<=':
                                filtered = df[df[c] <= val]
                                condition = f"{c} <= {val}"
                        elif op == 'contains':
                            filtered = df[df[c].astype(str).str.contains(val, case=False, na=False)]
                            condition = f"{c} contains '{val}'"
                        else:
                            if df[c].dtype in ['int64', 'float64']:
                                filtered = df[df[c] == float(val)]
                            else:
                                filtered = df[df[c].astype(str).str.lower() == val.lower()]
                            condition = f"{c} = {val}"
                        
                        if len(filtered) > 0:
                            response = f"### 🔍 Found {len(filtered)} rows where {condition}\n\nShowing first 20 results:"
                            return response, None, filtered.head(20)
                        else:
                            return f"❌ No rows found where {condition}", None, None
                    except:
                        pass
    
    return "❌ I couldn't understand the filter condition. Try something like: 'show rows where age > 30'", None, None


def sort_data(query, df):
    """Sort data by column"""
    # Extract column name
    for col in df.columns:
        if col.lower() in query:
            sort_col = col
            break
    else:
        sort_col = df.columns[0] if len(df.columns) > 0 else None
    
    if not sort_col:
        return "❌ Please specify a column to sort by", None, None
    
    # Determine order
    if 'desc' in query or 'highest' in query or 'largest' in query:
        ascending = False
        order = "descending"
    else:
        ascending = True
        order = "ascending"
    
    # Get number
    numbers = re.findall(r'\d+', query)
    n = int(numbers[0]) if numbers else 20
    
    sorted_df = df.sort_values(sort_col, ascending=ascending).head(n)
    
    response = f"### 📊 Sorted by {sort_col} ({order})\n\nShowing top {n} results:"
    return response, None, sorted_df


def create_bar_chart(query, df, categorical_cols):
    """Create bar chart for categorical column"""
    # Find requested column
    col = None
    for c in categorical_cols:
        if c.lower() in query:
            col = c
            break
    
    if not col and categorical_cols:
        col = categorical_cols[0]
    
    if col:
        value_counts = df[col].value_counts().head(20)
        
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            title=f"Bar Chart of {col} (Top 20)",
            labels={'x': col, 'y': 'Count'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2c3e50'),
            xaxis_tickangle=-45,
            height=500
        )
        
        response = f"### 📊 Bar Chart of '{col}'\n\nHere's the distribution of values:"
        return response, fig, None
    
    return "❌ No categorical column found for bar chart", None, None


def create_histogram(query, df, numeric_cols):
    """Create histogram for numeric column"""
    # Find requested column
    col = None
    for c in numeric_cols:
        if c.lower() in query:
            col = c
            break
    
    if not col and numeric_cols:
        col = numeric_cols[0]
    
    if col:
        fig = px.histogram(
            df, 
            x=col, 
            nbins=30,
            title=f"Histogram of {col}",
            marginal="box",
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2c3e50'),
            height=500
        )
        
        # Add statistics
        data = df[col].dropna()
        stats = f"Mean: {data.mean():.2f} | Median: {data.median():.2f} | Std: {data.std():.2f}"
        
        response = f"### 📊 Histogram of '{col}'\n\n{stats}"
        return response, fig, None
    
    return "❌ No numeric column found for histogram", None, None


def create_scatter_plot(query, df, numeric_cols):
    """Create scatter plot between two numeric columns"""
    # Find two numeric columns
    cols = []
    for col in numeric_cols:
        if col.lower() in query:
            cols.append(col)
    
    if len(cols) >= 2:
        x_col, y_col = cols[0], cols[1]
    elif len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
    else:
        return "❌ Need at least 2 numeric columns for scatter plot", None, None
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=f"Scatter Plot: {y_col} vs {x_col}",
        trendline="ols",
        opacity=0.6,
        color_discrete_sequence=['#667eea']
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#2c3e50'),
        height=500
    )
    
    # Calculate correlation
    corr = df[x_col].corr(df[y_col])
    
    response = f"### 📊 Scatter Plot: {y_col} vs {x_col}\n\nCorrelation: {corr:.4f}"
    return response, fig, None


def create_line_chart(query, df, numeric_cols, datetime_cols):
    """Create line chart for time series or sequential data"""
    # Find date column
    date_col = None
    for col in datetime_cols:
        if col.lower() in query:
            date_col = col
            break
    
    if not date_col and datetime_cols:
        date_col = datetime_cols[0]
    
    # Find value column
    val_col = None
    for col in numeric_cols:
        if col.lower() in query:
            val_col = col
            break
    
    if not val_col and numeric_cols:
        val_col = numeric_cols[0]
    
    if date_col and val_col:
        # Sort by date
        plot_df = df[[date_col, val_col]].dropna().sort_values(date_col)
        
        fig = px.line(
            plot_df, 
            x=date_col, 
            y=val_col,
            title=f"Trend of {val_col} over Time",
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2c3e50'),
            height=500
        )
        
        response = f"### 📈 Line Chart: {val_col} over Time"
        return response, fig, None
    
    return "❌ Need a datetime column and numeric column for line chart", None, None


def create_box_plot(query, df, numeric_cols, categorical_cols):
    """Create box plot"""
    # Find numeric column
    num_col = None
    for col in numeric_cols:
        if col.lower() in query:
            num_col = col
            break
    
    if not num_col and numeric_cols:
        num_col = numeric_cols[0]
    
    # Find categorical column for grouping
    cat_col = None
    for col in categorical_cols:
        if col.lower() in query:
            cat_col = col
            break
    
    if num_col:
        if cat_col:
            fig = px.box(
                df, 
                x=cat_col, 
                y=num_col,
                title=f"Box Plot of {num_col} by {cat_col}",
                color_discrete_sequence=['#667eea']
            )
            response = f"### 📊 Box Plot: {num_col} grouped by {cat_col}"
        else:
            fig = px.box(
                df, 
                y=num_col,
                title=f"Box Plot of {num_col}",
                color_discrete_sequence=['#667eea']
            )
            response = f"### 📊 Box Plot of {num_col}"
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2c3e50'),
            height=500
        )
        
        return response, fig, None
    
    return "❌ No numeric column found for box plot", None, None


def create_pie_chart(query, df, categorical_cols):
    """Create pie chart for categorical column"""
    # Find categorical column
    col = None
    for c in categorical_cols:
        if c.lower() in query:
            col = c
            break
    
    if not col and categorical_cols:
        col = categorical_cols[0]
    
    if col:
        value_counts = df[col].value_counts().head(10)
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Pie Chart of {col} (Top 10)",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=500,
            showlegend=True
        )
        
        response = f"### 🥧 Pie Chart of '{col}'\n\nProportion of values:"
        return response, fig, None
    
    return "❌ No categorical column found for pie chart", None, None


def create_heatmap(df, numeric_cols):
    """Create correlation heatmap"""
    if len(numeric_cols) < 2:
        return "❌ Need at least 2 numeric columns for correlation heatmap", None, None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap",
        zmin=-1, zmax=1
    )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    response = "### 🔥 Correlation Heatmap\n\nStrong correlations are shown in dark red/blue:"
    return response, fig, None


def create_violin_plot(query, df, numeric_cols, categorical_cols):
    """Create violin plot"""
    # Find numeric column
    num_col = None
    for col in numeric_cols:
        if col.lower() in query:
            num_col = col
            break
    
    if not num_col and numeric_cols:
        num_col = numeric_cols[0]
    
    # Find categorical column for grouping
    cat_col = None
    for col in categorical_cols:
        if col.lower() in query:
            cat_col = col
            break
    
    if num_col:
        if cat_col:
            fig = px.violin(
                df, 
                x=cat_col, 
                y=num_col,
                title=f"Violin Plot of {num_col} by {cat_col}",
                box=True,
                points="all",
                color_discrete_sequence=['#667eea']
            )
            response = f"### 🎻 Violin Plot: {num_col} grouped by {cat_col}"
        else:
            fig = px.violin(
                df, 
                y=num_col,
                title=f"Violin Plot of {num_col}",
                box=True,
                points="all",
                color_discrete_sequence=['#667eea']
            )
            response = f"### 🎻 Violin Plot of {num_col}"
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2c3e50'),
            height=500
        )
        
        return response, fig, None
    
    return "❌ No numeric column found for violin plot", None, None


def show_statistics(query, df, numeric_cols, all_cols):
    """Show statistics for columns"""
    # Check if asking about specific column
    for col in all_cols:
        if col.lower() in query and col in numeric_cols:
            data = df[col].dropna()
            
            stats_data = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    len(data),
                    f"{data.mean():.4f}",
                    f"{data.std():.4f}",
                    f"{data.min():.4f}",
                    f"{data.quantile(0.25):.4f}",
                    f"{data.median():.4f}",
                    f"{data.quantile(0.75):.4f}",
                    f"{data.max():.4f}",
                    f"{data.skew():.4f}",
                    f"{data.kurtosis():.4f}"
                ]
            })
            
            response = f"### 📊 Statistics for '{col}'"
            return response, None, stats_data
    
    # General statistics for all numeric columns
    if numeric_cols:
        stats_df = df[numeric_cols].describe().T
        stats_df['skew'] = df[numeric_cols].skew()
        stats_df['kurtosis'] = df[numeric_cols].kurtosis()
        
        response = "### 📈 Summary Statistics for Numeric Columns"
        return response, None, stats_df
    
    return "❌ No numeric columns found for statistics", None, None


def show_column_info(query, df, all_cols):
    """Show information about specific column or all columns"""
    # Check if asking about specific column
    for col in all_cols:
        if col.lower() in query:
            info_data = pd.DataFrame({
                'Property': ['Data Type', 'Unique Values', 'Missing Values', 'Missing %', 'Sample Values'],
                'Value': [
                    str(df[col].dtype),
                    df[col].nunique(),
                    df[col].isnull().sum(),
                    f"{(df[col].isnull().sum()/len(df)*100):.2f}%",
                    str(df[col].dropna().iloc[:3].tolist())
                ]
            })
            
            response = f"### 📋 Column Information: '{col}'"
            return response, None, info_data
    
    # General column information
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    response = "### 📋 All Columns Information"
    return response, None, col_info


def show_missing_values(df):
    """Show missing values analysis"""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        return "✅ **Good news!** No missing values found in the dataset.", None, None
    
    missing_data = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': (missing.values / len(df) * 100).round(2)
    }).sort_values('Missing %', ascending=False)
    
    total_missing = missing.sum()
    total_cells = df.shape[0] * df.shape[1]
    
    response = f"### 🔍 Missing Values Analysis\n\n**Total Missing:** {total_missing} out of {total_cells} cells ({total_missing/total_cells*100:.2f}%)"
    return response, None, missing_data


def detect_outliers(query, df, numeric_cols):
    """Detect outliers in numeric columns"""
    # Check if asking about specific column
    target_cols = []
    for col in numeric_cols:
        if col.lower() in query:
            target_cols.append(col)
    
    if not target_cols:
        target_cols = numeric_cols[:3]  # Check first 3 numeric columns
    
    outlier_data = []
    
    for col in target_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
        
        outlier_data.append({
            'Column': col,
            'Outliers Count': len(outliers),
            'Outliers %': f"{(len(outliers)/len(data)*100):.2f}%",
            'Normal Range': f"[{Q1 - 1.5 * IQR:.4f}, {Q3 + 1.5 * IQR:.4f}]",
            'Severity': 'High' if len(outliers)/len(data)*100 > 10 else 'Medium' if len(outliers)/len(data)*100 > 5 else 'Low'
        })
    
    outlier_df = pd.DataFrame(outlier_data)
    
    response = "### ⚠️ Outlier Detection Results"
    return response, None, outlier_df


def show_unique_values(query, df, all_cols, categorical_cols):
    """Show unique values in columns"""
    # Check if asking about specific column
    for col in all_cols:
        if col.lower() in query:
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'Count']
            value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
            
            response = f"### 🎯 Unique Values in '{col}'\n\n**Total Unique:** {df[col].nunique()}"
            return response, None, value_counts.head(20)
    
    # Show for categorical columns
    if categorical_cols:
        unique_data = []
        for col in categorical_cols[:10]:
            unique_data.append({
                'Column': col,
                'Unique Values': df[col].nunique(),
                'Most Common': df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A',
                'Most Common Count': df[col].value_counts().values[0] if len(df[col].value_counts()) > 0 else 0
            })
        
        unique_df = pd.DataFrame(unique_data)
        response = "### 🎯 Unique Values in Categorical Columns"
        return response, None, unique_df
    
    return "❌ No categorical columns found", None, None


def compare_columns(query, df, numeric_cols, categorical_cols):
    """Compare two columns"""
    # Find two columns to compare
    cols = []
    for col in df.columns:
        if col.lower() in query:
            cols.append(col)
    
    if len(cols) >= 2:
        col1, col2 = cols[0], cols[1]
        
        if col1 in numeric_cols and col2 in numeric_cols:
            # Numeric comparison
            comparison_data = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                col1: [
                    df[col1].mean(),
                    df[col1].median(),
                    df[col1].std(),
                    df[col1].min(),
                    df[col1].max()
                ],
                col2: [
                    df[col2].mean(),
                    df[col2].median(),
                    df[col2].std(),
                    df[col2].min(),
                    df[col2].max()
                ]
            })
            
            response = f"### 🔄 Comparison: {col1} vs {col2}"
            return response, None, comparison_data
        
        elif col1 in categorical_cols and col2 in categorical_cols:
            # Categorical comparison - crosstab
            cross_tab = pd.crosstab(df[col1], df[col2])
            response = f"### 🔄 Cross-tabulation: {col1} vs {col2}"
            return response, None, cross_tab
    
    return "❌ Please specify two columns to compare", None, None


def show_help():
    """Show help information"""
    help_text = """
    ### 🤖 I Can Help You With:
    
    **📊 Show Data:**
    • "Show me first 10 rows"
    • "Show me last 5 rows"
    • "Show random sample of 10 rows"
    • "Find rows where age > 30"
    • "Sort by price descending"
    • "Top 5 by sales"
    
    **📈 Create Visualizations:**
    • "Show bar chart of category"
    • "Plot histogram of age"
    • "Create scatter plot of price vs quantity"
    • "Show line chart of sales over time"
    • "Create box plot of salary"
    • "Show pie chart of region"
    • "Display correlation heatmap"
    • "Create violin plot of price"
    
    **🔍 Analyze Data:**
    • "Show statistics for all columns"
    • "Tell me about [column name]"
    • "Any missing values?"
    • "Find outliers in price"
    • "Show unique values in category"
    • "Compare age and income"
    
    **Just ask naturally and I'll show you the data and visualizations!**
    """
    return help_text


def handle_general_query(query, df, numeric_cols, categorical_cols, all_cols):
    """Handle general queries that don't match specific patterns"""
    
    # Check if asking about a specific column
    for col in all_cols:
        if col.lower() in query:
            if col in numeric_cols:
                data = df[col].dropna()
                return f"**{col}** - Mean: {data.mean():.2f}, Min: {data.min():.2f}, Max: {data.max():.2f}", None, None
            else:
                return f"**{col}** - Unique values: {df[col].nunique()}, Most common: {df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A'}", None, None
    
    # Check for dataset size
    if 'size' in query or 'large' in query or 'big' in query:
        size_mb = df.memory_usage(deep=True).sum() / 1024**2
        return f"Dataset size: {size_mb:.2f} MB ({df.shape[0]:,} rows × {df.shape[1]} columns)", None, None
    
    # Default response
    return "❌ I didn't understand. Try asking for data, visualizations, or type 'help'", None, None


def display_visualization(fig):
    """Display the visualization"""
    st.plotly_chart(fig, use_container_width=True)


# Simple version for quick integration
def run_simple_chatbot(df):
    """Simplified chatbot version"""
    st.markdown("### 💬 Simple Data Chat")
    
    if "simple_msgs" not in st.session_state:
        st.session_state.simple_msgs = []
    
    # Chat display
    for msg in st.session_state.simple_msgs:
        if msg["role"] == "user":
            st.info(f"👤 {msg['content']}")
        else:
            st.success(f"🤖 {msg['content']}")
    
    # Input
    user_input = st.text_input("Ask:", key="simple_chat_input")
    
    if st.button("Send") and user_input:
        st.session_state.simple_msgs.append({"role": "user", "content": user_input})
        
        # Simple responses
        response = "I don't understand. Try: rows, columns, missing, stats, chart"
        
        if "row" in user_input.lower():
            response = f"Dataset has {df.shape[0]} rows"
        elif "column" in user_input.lower():
            response = f"Dataset has {df.shape[1]} columns: {', '.join(df.columns[:5])}"
        elif "missing" in user_input.lower():
            missing = df.isnull().sum().sum()
            response = f"Found {missing} missing values" if missing > 0 else "No missing values"
        elif "stat" in user_input.lower():
            numeric = df.select_dtypes(include=[np.number]).columns
            if len(numeric) > 0:
                response = f"Mean of {numeric[0]}: {df[numeric[0]].mean():.2f}"
        elif "chart" in user_input.lower() or "plot" in user_input.lower():
            response = "📊 Creating visualization... (check the plot above)"
            # Simple histogram
            numeric = df.select_dtypes(include=[np.number]).columns
            if len(numeric) > 0:
                fig = px.histogram(df, x=numeric[0], title=f"Distribution of {numeric[0]}")
                st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.simple_msgs.append({"role": "bot", "content": response})
        st.rerun()
    
    if st.button("Clear Chat"):
        st.session_state.simple_msgs = []
        st.rerun()