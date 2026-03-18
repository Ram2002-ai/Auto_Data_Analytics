import pandas as pd
import numpy as np
import streamlit as st

def detect_column_types(df):
    """
    Detect and return column types
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
    boolean = df.select_dtypes(include=['bool']).columns.tolist()
    
    return numeric, categorical, datetime, boolean

def get_basic_stats(df):
    """
    Return basic statistics about the dataset
    """
    stats = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return stats

def suggest_visualizations(df):
    """
    Suggest appropriate visualizations based on data types
    """
    numeric, categorical, datetime, boolean = detect_column_types(df)
    
    suggestions = []
    
    if len(numeric) > 0:
        suggestions.append({
            'type': 'histogram',
            'description': f'Distribution of numeric columns',
            'columns': numeric[:3]
        })
    
    if len(categorical) > 0:
        suggestions.append({
            'type': 'bar_chart',
            'description': f'Category distributions',
            'columns': categorical[:3]
        })
    
    if len(numeric) >= 2:
        suggestions.append({
            'type': 'scatter_plot',
            'description': 'Relationship between numeric variables',
            'columns': numeric[:2]
        })
    
    if len(datetime) > 0 and len(numeric) > 0:
        suggestions.append({
            'type': 'line_chart',
            'description': 'Time series trends',
            'columns': [datetime[0], numeric[0]]
        })
    
    if len(numeric) > 1:
        suggestions.append({
            'type': 'correlation_heatmap',
            'description': 'Correlations between numeric variables'
        })
    
    return suggestions

def format_number(num):
    """
    Format large numbers with commas
    """
    if pd.isna(num):
        return "N/A"
    return f"{num:,.0f}"

def format_percentage(num):
    """
    Format as percentage
    """
    if pd.isna(num):
        return "N/A"
    return f"{num:.1f}%"

def get_data_quality_issues(df):
    """
    Identify data quality issues
    """
    issues = []
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append({
            'type': 'missing_values',
            'severity': 'high' if df.isnull().sum().sum() > len(df) * 0.1 else 'medium',
            'description': f'Missing values in {len(missing_cols)} columns',
            'columns': missing_cols
        })
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append({
            'type': 'duplicates',
            'severity': 'medium' if duplicates > len(df) * 0.05 else 'low',
            'description': f'{duplicates} duplicate rows found',
            'count': duplicates
        })
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues.append({
            'type': 'constant_columns',
            'severity': 'low',
            'description': f'{len(constant_cols)} constant columns found',
            'columns': constant_cols
        })
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        if len(outliers) > len(df) * 0.1:
            issues.append({
                'type': 'outliers',
                'severity': 'medium',
                'description': f'Significant outliers in {col}',
                'column': col,
                'outlier_count': len(outliers)
            })
            break  # Just report first outlier issue
    
    return issues

def get_recommendations(df):
    """
    Generate data analysis recommendations
    """
    numeric, categorical, datetime, boolean = detect_column_types(df)
    
    recommendations = []
    
    # Missing data recommendations
    if df.isnull().sum().sum() > 0:
        recommendations.append("Consider handling missing values using imputation or removal")
    
    # Feature engineering suggestions
    if len(numeric) >= 2:
        recommendations.append("Create interaction features between highly correlated variables")
    
    if datetime:
        recommendations.append("Extract time-based features (hour, day, month, year) from datetime columns")
    
    # Modeling suggestions
    if len(numeric) > 5:
        recommendations.append("Consider dimensionality reduction techniques (PCA, t-SNE)")
    
    if df.shape[0] > 10000:
        recommendations.append("Dataset is large - consider sampling for faster exploration")
    
    # Visualization suggestions
    if len(numeric) > 2:
        recommendations.append("Use pair plots to visualize relationships between multiple variables")
    
    if len(categorical) > 1:
        recommendations.append("Create contingency tables to analyze categorical relationships")
    
    return recommendations

def create_sample_dataset():
    """
    Create a sample dataset for testing
    """
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'id': range(n_rows),
        'age': np.random.normal(40, 15, n_rows).clip(18, 90).astype(int),
        'income': np.random.normal(50000, 20000, n_rows).clip(20000, 150000).astype(int),
        'score': np.random.uniform(0, 100, n_rows).round(2),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'purchased': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'signup_date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
        'satisfaction': np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.1, 0.15, 0.3, 0.25, 0.2])
    }
    
    # Add some missing values
    df = pd.DataFrame(data)
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)
    
    # Add some duplicates
    duplicate_rows = np.random.choice(n_rows, 10, replace=False)
    df = pd.concat([df, df.iloc[duplicate_rows]]).reset_index(drop=True)
    
    return df