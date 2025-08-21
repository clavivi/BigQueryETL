"""Data cleaning and processing module for BigQuery compatibility"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st

def clean_column_name(col_name: str) -> str:
    """
    Clean a single column name for BigQuery compatibility
    
    BigQuery column naming rules:
    - Must start with letter or underscore
    - Can contain letters, numbers, and underscores
    - Maximum length 300 characters
    """
    cleaned = col_name.strip()
    
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    cleaned = cleaned.lower()
    
    cleaned = re.sub(r'_+', '_', cleaned)
    
    cleaned = cleaned.strip('_')
    
    if cleaned and not cleaned[0].isalpha() and cleaned[0] != '_':
        cleaned = f'col_{cleaned}'
    
    if not cleaned:
        cleaned = 'column'
    
    if len(cleaned) > 300:
        cleaned = cleaned[:300]
    
    return cleaned

def clean_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Clean all column names in a DataFrame and handle duplicates
    
    Returns:
        Tuple of (cleaned DataFrame, mapping of original to cleaned names)
    """
    column_mapping = {}
    cleaned_names = []
    name_counts = {}
    
    for col in df.columns:
        cleaned = clean_column_name(str(col))
        
        if cleaned in name_counts:
            name_counts[cleaned] += 1
            cleaned = f"{cleaned}_{name_counts[cleaned]}"
        else:
            name_counts[cleaned] = 0
        
        cleaned_names.append(cleaned)
        column_mapping[str(col)] = cleaned
    
    df_cleaned = df.copy()
    df_cleaned.columns = cleaned_names
    
    return df_cleaned, column_mapping

def detect_date_columns(df: pd.DataFrame, sample_size: int = 100) -> Dict[str, str]:
    """
    Automatically detect date columns and their formats
    
    Returns:
        Dictionary mapping column names to detected date formats
    """
    date_formats = [
        '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y',
        '%m/%d/%y', '%d/%m/%y', '%y-%m-%d', '%m-%d-%y',
        '%Y/%m/%d', '%d-%m-%Y', '%Y%m%d', '%m%d%Y',
        '%d.%m.%Y', '%Y.%m.%d', '%m.%d.%Y',
        '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y'
    ]
    
    date_columns = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(sample_size)
            
            if len(sample) == 0:
                continue
            
            for date_format in date_formats:
                try:
                    parsed_count = 0
                    for value in sample:
                        try:
                            pd.to_datetime(str(value), format=date_format)
                            parsed_count += 1
                        except:
                            pass
                    
                    if parsed_count / len(sample) >= 0.8:
                        date_columns[col] = date_format
                        break
                except:
                    continue
            
            if col not in date_columns:
                try:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    date_columns[col] = 'infer'
                except:
                    pass
    
    return date_columns

def process_date_column(
    series: pd.Series, 
    date_format: str = 'infer',
    errors: str = 'coerce'
) -> Tuple[pd.Series, List[str]]:
    """
    Process a date column to BigQuery DATE format
    
    Returns:
        Tuple of (processed series, list of error messages)
    """
    error_messages = []
    
    if date_format == 'infer':
        processed = pd.to_datetime(series, infer_datetime_format=True, errors=errors)
    else:
        processed = pd.to_datetime(series, format=date_format, errors=errors)
    
    invalid_count = processed.isna().sum() - series.isna().sum()
    if invalid_count > 0:
        error_messages.append(f"Found {invalid_count} invalid date values")
    
    processed = processed.dt.date
    
    return processed, error_messages

def detect_amount_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that likely contain monetary amounts
    
    Returns:
        List of column names that appear to contain amounts
    """
    amount_columns = []
    amount_keywords = [
        'amount', 'price', 'cost', 'fee', 'total', 'subtotal',
        'payment', 'donation', 'gift', 'value', 'balance',
        'revenue', 'income', 'expense', 'salary', 'wage'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        
        if any(keyword in col_lower for keyword in amount_keywords):
            amount_columns.append(col)
            continue
        
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                currency_pattern = r'^[\$€£¥₹]\s*[\d,.-]+$|^[\d,.-]+\s*[\$€£¥₹]$'
                matches = sample.astype(str).str.match(currency_pattern)
                if matches.sum() / len(sample) > 0.5:
                    amount_columns.append(col)
    
    return amount_columns

def process_amount_column(
    series: pd.Series,
    decimal_separator: str = '.',
    thousands_separator: str = ','
) -> Tuple[pd.Series, List[str]]:
    """
    Process amount/currency column to FLOAT64
    
    Returns:
        Tuple of (processed series, list of error messages)
    """
    error_messages = []
    processed = series.copy()
    
    if processed.dtype == 'object':
        processed = processed.astype(str)
        
        processed = processed.str.replace(r'[\$€£¥₹]', '', regex=True)
        
        processed = processed.str.strip()
        
        processed = processed.str.replace(r'\s+', '', regex=True)
        
        if decimal_separator == ',' and thousands_separator == '.':
            processed = processed.str.replace('.', '', regex=False)
            processed = processed.str.replace(',', '.', regex=False)
        else:
            processed = processed.str.replace(',', '', regex=False)
        
        processed = processed.str.replace(r'[()]', '', regex=True)
        
        processed = processed.replace('', np.nan)
        
        try:
            processed = pd.to_numeric(processed, errors='coerce')
            invalid_count = processed.isna().sum() - series.isna().sum()
            if invalid_count > 0:
                error_messages.append(f"Found {invalid_count} invalid numeric values")
        except Exception as e:
            error_messages.append(f"Error converting to numeric: {str(e)}")
    
    return processed, error_messages

def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect appropriate BigQuery data types for each column
    
    Returns:
        Dictionary mapping column names to BigQuery data types
    """
    type_mapping = {}
    
    date_columns = detect_date_columns(df)
    amount_columns = detect_amount_columns(df)
    
    for col in df.columns:
        if col in date_columns:
            type_mapping[col] = 'DATE'
        elif col in amount_columns:
            type_mapping[col] = 'FLOAT64'
        elif df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
            type_mapping[col] = 'INT64'
        elif df[col].dtype in ['float64', 'float32']:
            type_mapping[col] = 'FLOAT64'
        elif df[col].dtype == 'bool':
            type_mapping[col] = 'BOOL'
        else:
            type_mapping[col] = 'STRING'
    
    return type_mapping

def clean_dataframe(
    df: pd.DataFrame,
    date_columns: Optional[Dict[str, str]] = None,
    amount_columns: Optional[List[str]] = None,
    decimal_separator: str = '.',
    thousands_separator: str = ','
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive DataFrame cleaning for BigQuery compatibility
    
    Args:
        df: Original DataFrame
        date_columns: Dict mapping column names to date formats
        amount_columns: List of columns to treat as amounts
        decimal_separator: Decimal separator for amounts
        thousands_separator: Thousands separator for amounts
    
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    df_cleaned, column_mapping = clean_column_names(df)
    
    if date_columns is None:
        date_columns = detect_date_columns(df_cleaned)
    
    if amount_columns is None:
        amount_columns = detect_amount_columns(df_cleaned)
    
    cleaning_report = {
        'column_mapping': column_mapping,
        'date_columns': {},
        'amount_columns': {},
        'data_types': {},
        'errors': []
    }
    
    for col, date_format in date_columns.items():
        if col in df_cleaned.columns:
            processed, errors = process_date_column(df_cleaned[col], date_format)
            df_cleaned[col] = processed
            cleaning_report['date_columns'][col] = {
                'format': date_format,
                'errors': errors
            }
    
    for col in amount_columns:
        if col in df_cleaned.columns:
            processed, errors = process_amount_column(
                df_cleaned[col], 
                decimal_separator, 
                thousands_separator
            )
            df_cleaned[col] = processed
            cleaning_report['amount_columns'][col] = {
                'errors': errors
            }
    
    cleaning_report['data_types'] = detect_data_types(df_cleaned)
    
    return df_cleaned, cleaning_report

def get_cleaning_summary(cleaning_report: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary DataFrame of the cleaning process
    
    Returns:
        DataFrame with cleaning summary information
    """
    summary_data = []
    
    for original, cleaned in cleaning_report['column_mapping'].items():
        row = {
            'Original Name': original,
            'Cleaned Name': cleaned,
            'Data Type': cleaning_report['data_types'].get(cleaned, 'STRING'),
            'Special Processing': []
        }
        
        if cleaned in cleaning_report['date_columns']:
            row['Special Processing'].append('Date')
        
        if cleaned in cleaning_report['amount_columns']:
            row['Special Processing'].append('Amount')
        
        row['Special Processing'] = ', '.join(row['Special Processing']) if row['Special Processing'] else 'None'
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def validate_cleaned_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate cleaned data for BigQuery compatibility
    
    Returns:
        Tuple of (is_valid, list of validation messages)
    """
    messages = []
    is_valid = True
    
    if df.empty:
        messages.append("❌ DataFrame is empty")
        is_valid = False
    
    for col in df.columns:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
            messages.append(f"❌ Invalid column name: {col}")
            is_valid = False
    
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        messages.append(f"❌ Duplicate columns found: {', '.join(duplicate_cols)}")
        is_valid = False
    
    for col in df.columns:
        null_pct = (df[col].isna().sum() / len(df)) * 100
        if null_pct > 90:
            messages.append(f"⚠️ Column '{col}' is {null_pct:.1f}% null")
    
    if is_valid and not messages:
        messages.append("✅ Data is valid for BigQuery upload")
    
    return is_valid, messages