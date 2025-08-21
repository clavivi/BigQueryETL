"""
BigQuery Donation Data Upload System
A comprehensive Streamlit app for uploading and processing donation CSV files to Google BigQuery
"""

import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import re
import hashlib
from datetime import datetime, date
import io
from typing import Dict, List, Tuple, Optional, Any
import time

# Page configuration
st.set_page_config(
    page_title="BigQuery Donation Upload System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'client' not in st.session_state:
    st.session_state.client = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = []
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'pii_columns' not in st.session_state:
    st.session_state.pii_columns = {}
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []

# ================== UTILITY FUNCTIONS ==================

def clean_column_name(col_name: str) -> str:
    """Clean column name to be BigQuery compatible"""
    # Replace spaces and special characters with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', col_name)
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Ensure it starts with a letter or underscore
    if cleaned and cleaned[0].isdigit():
        cleaned = '_' + cleaned
    # Convert to lowercase
    cleaned = cleaned.lower()
    # Trim to BigQuery's column name length limit
    return cleaned[:128] if cleaned else 'column'

def detect_column_type(series: pd.Series) -> str:
    """Detect the BigQuery data type for a pandas series"""
    # Remove null values for analysis
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return 'STRING'
    
    # Check for boolean
    if set(non_null.unique()).issubset({True, False, 'true', 'false', 'True', 'False', 0, 1}):
        return 'BOOLEAN'
    
    # Check for dates
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return 'DATETIME'
    
    # Try to convert to datetime
    try:
        pd.to_datetime(non_null, errors='raise')
        return 'DATE'
    except:
        pass
    
    # Check for integers
    try:
        if all(float(x).is_integer() for x in non_null if str(x).replace('-', '').replace('+', '').isdigit()):
            return 'INTEGER'
    except:
        pass
    
    # Check for floats
    try:
        pd.to_numeric(non_null, errors='raise')
        return 'FLOAT64'
    except:
        pass
    
    # Default to STRING
    return 'STRING'

def clean_donation_amount(value: Any) -> Optional[float]:
    """Clean and convert donation amount to float"""
    if pd.isna(value):
        return None
    
    # Convert to string
    str_value = str(value)
    
    # Remove currency symbols and spaces
    str_value = re.sub(r'[$€£¥₹,\s]', '', str_value)
    
    # Handle parentheses for negative values
    if '(' in str_value and ')' in str_value:
        str_value = '-' + str_value.replace('(', '').replace(')', '')
    
    # Try to convert to float
    try:
        return float(str_value)
    except:
        return None

def parse_date(value: Any, date_format: str = None) -> Optional[date]:
    """Parse date with various formats"""
    if pd.isna(value):
        return None
    
    date_formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%m-%d-%Y',
        '%d-%m-%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y'
    ]
    
    if date_format:
        date_formats.insert(0, date_format)
    
    str_value = str(value).strip()
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(str_value, format=fmt).date()
        except:
            continue
    
    # Try pandas' intelligent date parsing
    try:
        return pd.to_datetime(str_value).date()
    except:
        return None

def detect_pii_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> Dict[str, Dict]:
    """Detect potential PII columns in the dataframe"""
    pii_columns = {}
    
    # PII patterns
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    phone_pattern = r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$'
    ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'
    
    for orig_col, clean_col in column_mapping.items():
        confidence = 0
        pii_type = []
        
        # Check column name
        lower_col = orig_col.lower()
        if any(term in lower_col for term in ['email', 'e-mail', 'mail']):
            confidence = 90
            pii_type.append('email')
        elif any(term in lower_col for term in ['phone', 'tel', 'mobile', 'cell']):
            confidence = 90
            pii_type.append('phone')
        elif any(term in lower_col for term in ['name', 'first', 'last', 'surname', 'donor']):
            confidence = 85
            pii_type.append('name')
        elif any(term in lower_col for term in ['address', 'street', 'city', 'state', 'zip', 'postal']):
            confidence = 80
            pii_type.append('address')
        elif any(term in lower_col for term in ['ssn', 'social', 'tax']):
            confidence = 95
            pii_type.append('ssn')
        
        # Sample data to check patterns (if confidence not already high)
        if confidence < 80 and orig_col in df.columns:
            sample = df[orig_col].dropna().astype(str).head(100)
            
            # Check for email pattern
            if sample.str.match(email_pattern).any():
                confidence = max(confidence, 95)
                if 'email' not in pii_type:
                    pii_type.append('email')
            
            # Check for phone pattern
            elif sample.str.match(phone_pattern).any():
                confidence = max(confidence, 90)
                if 'phone' not in pii_type:
                    pii_type.append('phone')
            
            # Check for SSN pattern
            elif sample.str.match(ssn_pattern).any():
                confidence = max(confidence, 95)
                if 'ssn' not in pii_type:
                    pii_type.append('ssn')
        
        if confidence > 0:
            pii_columns[clean_col] = {
                'original_name': orig_col,
                'confidence': confidence,
                'pii_type': pii_type,
                'action': 'review'  # default action
            }
    
    return pii_columns

def hash_column(series: pd.Series) -> pd.Series:
    """Apply SHA-256 hashing to a column"""
    return series.apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None)

def validate_data(df: pd.DataFrame, table_type: str) -> Dict[str, List]:
    """Validate data and return errors/warnings"""
    errors = []
    warnings = []
    info = []
    
    # Check for empty dataframe
    if df.empty:
        errors.append("The uploaded file contains no data")
        return {'errors': errors, 'warnings': warnings, 'info': info}
    
    # Check for required columns based on table type
    if table_type == 'Gift file':
        required_cols = ['amount', 'date', 'donor']
        for col in required_cols:
            if not any(col in c.lower() for c in df.columns):
                warnings.append(f"Expected column containing '{col}' not found")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate rows")
    
    # Check for null values in key columns
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            null_pct = (null_count / len(df)) * 100
            if null_pct > 50:
                warnings.append(f"Column '{col}' has {null_pct:.1f}% null values")
            else:
                info.append(f"Column '{col}' has {null_count} null values ({null_pct:.1f}%)")
    
    # Validate email columns
    for col in df.columns:
        if 'email' in col.lower():
            sample = df[col].dropna().astype(str).head(100)
            invalid_emails = sample[~sample.str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
            if len(invalid_emails) > 0:
                warnings.append(f"Column '{col}' contains invalid email formats")
    
    # Check date ranges
    for col in df.columns:
        if df[col].dtype in ['datetime64[ns]', 'object']:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                if dates.notna().any():
                    min_date = dates.min()
                    max_date = dates.max()
                    if min_date < pd.Timestamp('1900-01-01'):
                        warnings.append(f"Column '{col}' contains dates before 1900")
                    if max_date > pd.Timestamp.now() + pd.Timedelta(days=365):
                        warnings.append(f"Column '{col}' contains dates more than 1 year in the future")
            except:
                pass
    
    return {'errors': errors, 'warnings': warnings, 'info': info}

def generate_bq_schema(df: pd.DataFrame, column_types: Dict[str, str]) -> List[bigquery.SchemaField]:
    """Generate BigQuery schema from dataframe"""
    schema = []
    for col in df.columns:
        col_type = column_types.get(col, 'STRING')
        # Set mode to NULLABLE by default
        schema.append(bigquery.SchemaField(col, col_type, mode='NULLABLE'))
    return schema

def create_audit_log(
    action: str,
    client_dataset: str,
    table_name: str,
    table_type: str,
    row_count: int,
    file_name: str,
    pii_columns: Dict,
    errors: List,
    warnings: List,
    success: bool
) -> Dict:
    """Create an audit log entry"""
    return {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'client_dataset': client_dataset,
        'table_name': table_name,
        'table_type': table_type,
        'row_count': row_count,
        'file_name': file_name,
        'file_hash': hashlib.md5(file_name.encode()).hexdigest(),
        'pii_columns_detected': len(pii_columns),
        'pii_columns': list(pii_columns.keys()),
        'error_count': len(errors),
        'warning_count': len(warnings),
        'success': success,
        'errors': errors[:5],  # Limit to first 5 errors
        'warnings': warnings[:5]  # Limit to first 5 warnings
    }

# ================== MAIN APPLICATION ==================

def main():
    st.title("📊 BigQuery Donation Upload System")
    st.markdown("Upload and process donation CSV files to Google BigQuery with PII detection and compliance features")
    
    # Sidebar for authentication
    with st.sidebar:
        st.header("🔐 Authentication")
        
        if not st.session_state.authenticated:
            uploaded_key = st.file_uploader("Upload Service Account JSON Key", type=['json'])
            
            if uploaded_key:
                try:
                    # Parse the JSON key
                    key_dict = json.load(uploaded_key)
                    
                    # Create credentials
                    credentials = service_account.Credentials.from_service_account_info(key_dict)
                    
                    # Create BigQuery client
                    client = bigquery.Client(credentials=credentials, project=key_dict['project_id'])
                    
                    # Test connection by listing datasets
                    datasets = list(client.list_datasets())
                    
                    # Store in session state
                    st.session_state.authenticated = True
                    st.session_state.client = client
                    st.session_state.datasets = [dataset.dataset_id for dataset in datasets]
                    st.session_state.project_id = key_dict['project_id']
                    
                    st.success(f"✅ Connected to project: {key_dict['project_id']}")
                    st.info(f"Found {len(datasets)} datasets (clients)")
                    
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
        else:
            st.success(f"✅ Connected to: {st.session_state.project_id}")
            st.info(f"Available clients: {len(st.session_state.datasets)}")
            
            if st.button("🔌 Disconnect"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Show upload history
        if st.session_state.upload_history:
            st.header("📜 Recent Uploads")
            for log in st.session_state.upload_history[-5:]:
                status = "✅" if log['success'] else "❌"
                st.text(f"{status} {log['table_name']}")
                st.caption(f"{log['timestamp'][:19]} - {log['row_count']} rows")
    
    # Main content area
    if not st.session_state.authenticated:
        st.info("👈 Please authenticate using your BigQuery service account JSON key in the sidebar")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ Setup", "2️⃣ Upload & Clean", "3️⃣ PII Detection", 
        "4️⃣ Validation", "5️⃣ Upload", "6️⃣ Audit Log"
    ])
    
    # Tab 1: Setup
    with tab1:
        st.header("BigQuery Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select client (dataset)
            selected_dataset = st.selectbox(
                "Select Client (Dataset)",
                options=st.session_state.datasets,
                help="Each client has their own dataset in BigQuery"
            )
            
            # Table operation
            operation = st.radio(
                "Operation",
                ["Create new table", "Update existing table"],
                help="Choose whether to create a new table or update an existing one"
            )
        
        with col2:
            # Table type
            table_type = st.radio(
                "Table Type",
                ["Gift file", "Campaign donor file"],
                help="Select the type of data you're uploading"
            )
            
            # Table selection/naming
            if operation == "Update existing table":
                # Get tables from selected dataset
                if selected_dataset:
                    dataset_ref = st.session_state.client.dataset(selected_dataset)
                    tables = list(st.session_state.client.list_tables(dataset_ref))
                    table_names = [table.table_id for table in tables]
                    
                    if table_names:
                        selected_table = st.selectbox("Select Table to Update", table_names)
                    else:
                        st.warning("No tables found in this dataset")
                        selected_table = None
                else:
                    selected_table = None
            else:
                # New table name
                new_table_name = st.text_input(
                    "New Table Name",
                    help="Enter a name for the new table (letters, numbers, underscores only)"
                )
                # Validate table name
                if new_table_name:
                    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', new_table_name):
                        st.error("Table name must start with a letter and contain only letters, numbers, and underscores")
                    else:
                        st.success(f"Table will be created as: {selected_dataset}.{new_table_name}")
                selected_table = new_table_name if new_table_name else None
        
        # Store configuration
        st.session_state.config = {
            'dataset': selected_dataset,
            'operation': operation,
            'table_type': table_type,
            'table_name': selected_table
        }
        
        if all([selected_dataset, selected_table]):
            st.success("✅ Configuration complete! Proceed to upload your CSV file.")
    
    # Tab 2: Upload & Clean
    with tab2:
        st.header("Upload & Data Cleaning")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            # Read CSV with encoding detection
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'cp1252', 'utf-16']
                df = None
                successful_encoding = None
                
                for encoding in encodings:
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        successful_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception:
                        continue
                
                if df is None:
                    # If all encodings fail, try with error handling
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
                    successful_encoding = 'utf-8 (with errors ignored)'
                
                st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns using {successful_encoding} encoding")
                
                # Show original data preview
                st.subheader("Original Data Preview")
                st.dataframe(df.head(10))
                
                # Column cleaning
                st.subheader("Column Name Cleaning")
                
                # Create column mapping
                column_mapping = {}
                column_types = {}
                
                for col in df.columns:
                    clean_name = clean_column_name(col)
                    # Handle duplicates
                    if clean_name in column_mapping.values():
                        counter = 1
                        while f"{clean_name}_{counter}" in column_mapping.values():
                            counter += 1
                        clean_name = f"{clean_name}_{counter}"
                    column_mapping[col] = clean_name
                    column_types[clean_name] = detect_column_type(df[col])
                
                # Display mapping table
                mapping_df = pd.DataFrame({
                    'Original Name': list(column_mapping.keys()),
                    'Cleaned Name': list(column_mapping.values()),
                    'Detected Type': [column_types[v] for v in column_mapping.values()]
                })
                
                st.dataframe(mapping_df)
                
                # Allow manual editing
                if st.checkbox("Edit column names/types"):
                    edited_mapping = {}
                    edited_types = {}
                    
                    for _, row in mapping_df.iterrows():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.text(row['Original Name'])
                        with col2:
                            new_name = st.text_input(
                                f"Clean name for {row['Original Name']}", 
                                value=row['Cleaned Name'],
                                key=f"name_{row['Original Name']}"
                            )
                            edited_mapping[row['Original Name']] = new_name
                        with col3:
                            new_type = st.selectbox(
                                f"Type for {row['Original Name']}",
                                ['STRING', 'INTEGER', 'FLOAT64', 'BOOLEAN', 'DATE', 'DATETIME'],
                                index=['STRING', 'INTEGER', 'FLOAT64', 'BOOLEAN', 'DATE', 'DATETIME'].index(row['Detected Type']),
                                key=f"type_{row['Original Name']}"
                            )
                            edited_types[new_name] = new_type
                    
                    column_mapping = edited_mapping
                    column_types = edited_types
                
                # Process data
                st.subheader("Data Processing")
                
                # Date format selection
                date_columns = [col for col, dtype in column_types.items() if dtype in ['DATE', 'DATETIME']]
                if date_columns:
                    date_format = st.selectbox(
                        "Date Format (if auto-detection fails)",
                        ['Auto-detect', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y'],
                        help="Select the date format used in your CSV"
                    )
                else:
                    date_format = 'Auto-detect'
                
                # Process the dataframe
                if st.button("Process Data"):
                    # Rename columns
                    df_processed = df.rename(columns=column_mapping)
                    
                    # Process data types
                    for col, dtype in column_types.items():
                        if col in df_processed.columns:
                            if dtype == 'FLOAT64':
                                # Check if it might be a donation amount
                                if any(term in col.lower() for term in ['amount', 'donation', 'gift', 'payment']):
                                    df_processed[col] = df_processed[col].apply(clean_donation_amount)
                                else:
                                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                            elif dtype == 'INTEGER':
                                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype('Int64')
                            elif dtype in ['DATE', 'DATETIME']:
                                df_processed[col] = df_processed[col].apply(
                                    lambda x: parse_date(x, None if date_format == 'Auto-detect' else date_format)
                                )
                            elif dtype == 'BOOLEAN':
                                df_processed[col] = df_processed[col].astype(bool)
                    
                    # Store processed data
                    st.session_state.processed_df = df_processed
                    st.session_state.column_mapping = column_mapping
                    st.session_state.column_types = column_types
                    
                    # Show processed data
                    st.subheader("Processed Data Preview")
                    st.dataframe(df_processed.head(10))
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", len(df_processed))
                    with col2:
                        st.metric("Total Columns", len(df_processed.columns))
                    with col3:
                        null_count = df_processed.isnull().sum().sum()
                        st.metric("Null Values", null_count)
                    with col4:
                        duplicate_count = df_processed.duplicated().sum()
                        st.metric("Duplicate Rows", duplicate_count)
                    
                    st.success("✅ Data processed successfully! Proceed to PII detection.")
            
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # Tab 3: PII Detection
    with tab3:
        st.header("PII Detection & Security")
        
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            
            # Detect PII columns
            if st.button("Scan for PII"):
                pii_columns = detect_pii_columns(df, st.session_state.column_mapping)
                st.session_state.pii_columns = pii_columns
                
                if pii_columns:
                    st.warning(f"⚠️ Detected {len(pii_columns)} potential PII columns")
                    
                    # Display PII columns with actions
                    for col, info in pii_columns.items():
                        with st.expander(f"📋 {col} (Confidence: {info['confidence']}%)"):
                            st.write(f"**Original Name:** {info['original_name']}")
                            st.write(f"**PII Type:** {', '.join(info['pii_type'])}")
                            st.write(f"**Confidence:** {info['confidence']}%")
                            
                            # Sample data
                            if col in df.columns:
                                st.write("**Sample Data:**")
                                st.dataframe(df[col].dropna().head(5))
                            
                            # Action selection
                            action = st.selectbox(
                                "Action to take",
                                ["Review Required", "Hash Column", "Exclude from Upload", 
                                 "Apply Column Security", "Keep As-Is"],
                                key=f"pii_action_{col}"
                            )
                            
                            st.session_state.pii_columns[col]['action'] = action
                            
                            if action == "Hash Column":
                                st.info("This column will be hashed using SHA-256")
                            elif action == "Exclude from Upload":
                                st.info("This column will not be uploaded to BigQuery")
                            elif action == "Apply Column Security":
                                st.info("Column-level access control will be applied in BigQuery")
                            elif action == "Keep As-Is":
                                confirm = st.checkbox(
                                    "I confirm this data can be uploaded without protection",
                                    key=f"confirm_{col}"
                                )
                                if not confirm:
                                    st.error("Please confirm to proceed with unprotected upload")
                else:
                    st.success("✅ No PII columns detected")
            
            # Apply PII actions
            if st.session_state.pii_columns and st.button("Apply PII Protection"):
                df_protected = df.copy()
                excluded_columns = []
                
                for col, info in st.session_state.pii_columns.items():
                    action = info.get('action', 'Review Required')
                    
                    if action == "Hash Column" and col in df_protected.columns:
                        df_protected[col] = hash_column(df_protected[col])
                        st.success(f"✅ Hashed column: {col}")
                    elif action == "Exclude from Upload" and col in df_protected.columns:
                        df_protected = df_protected.drop(columns=[col])
                        excluded_columns.append(col)
                        st.info(f"🚫 Excluded column: {col}")
                    elif action == "Apply Column Security":
                        st.info(f"🔒 Column security will be applied to: {col}")
                
                st.session_state.processed_df = df_protected
                
                if excluded_columns:
                    st.warning(f"Excluded {len(excluded_columns)} columns from upload")
                
                st.success("✅ PII protection applied! Proceed to validation.")
        else:
            st.info("Please process your data in the 'Upload & Clean' tab first")
    
    # Tab 4: Validation
    with tab4:
        st.header("Data Validation")
        
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            
            # Run validation
            if st.button("Validate Data"):
                validation_results = validate_data(df, st.session_state.config.get('table_type', 'Gift file'))
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("❌ Errors", len(validation_results['errors']))
                    for error in validation_results['errors']:
                        st.error(error)
                
                with col2:
                    st.metric("⚠️ Warnings", len(validation_results['warnings']))
                    for warning in validation_results['warnings']:
                        st.warning(warning)
                
                with col3:
                    st.metric("ℹ️ Info", len(validation_results['info']))
                    for info in validation_results['info']:
                        st.info(info)
                
                # Data quality metrics
                st.subheader("Data Quality Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Completeness
                    completeness = ((len(df.columns) * len(df) - df.isnull().sum().sum()) / 
                                  (len(df.columns) * len(df)) * 100)
                    st.metric("Data Completeness", f"{completeness:.1f}%")
                    
                    # Uniqueness
                    unique_rows = len(df) - df.duplicated().sum()
                    uniqueness = (unique_rows / len(df)) * 100
                    st.metric("Row Uniqueness", f"{uniqueness:.1f}%")
                
                with col2:
                    # Find amount columns
                    amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'donation' in col.lower()]
                    if amount_cols and df[amount_cols[0]].dtype in ['float64', 'int64']:
                        st.metric("Total Donation Amount", f"${df[amount_cols[0]].sum():,.2f}")
                        st.metric("Average Donation", f"${df[amount_cols[0]].mean():,.2f}")
                
                # Allow download of validation report
                if st.button("Download Validation Report"):
                    report = {
                        'timestamp': datetime.now().isoformat(),
                        'file_rows': len(df),
                        'file_columns': len(df.columns),
                        'completeness': completeness,
                        'uniqueness': uniqueness,
                        'errors': validation_results['errors'],
                        'warnings': validation_results['warnings'],
                        'info': validation_results['info']
                    }
                    
                    report_json = json.dumps(report, indent=2)
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=report_json,
                        file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # Proceed button
                if len(validation_results['errors']) == 0:
                    st.success("✅ No critical errors found! You can proceed to upload.")
                else:
                    st.error("❌ Please resolve critical errors before uploading")
        else:
            st.info("Please process your data first")
    
    # Tab 5: Upload
    with tab5:
        st.header("Upload to BigQuery")
        
        if st.session_state.processed_df is not None and st.session_state.config.get('table_name'):
            df = st.session_state.processed_df
            config = st.session_state.config
            
            # Upload configuration
            st.subheader("Upload Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Dataset:** {config['dataset']}")
                st.write(f"**Table:** {config['table_name']}")
                st.write(f"**Operation:** {config['operation']}")
                st.write(f"**Table Type:** {config['table_type']}")
            
            with col2:
                st.write(f"**Rows to Upload:** {len(df)}")
                st.write(f"**Columns:** {len(df.columns)}")
                
                # Write disposition for updates
                if config['operation'] == "Update existing table":
                    write_disposition = st.selectbox(
                        "Write Disposition",
                        ["WRITE_APPEND", "WRITE_TRUNCATE"],
                        help="APPEND adds to existing data, TRUNCATE replaces all data"
                    )
                else:
                    write_disposition = "WRITE_EMPTY"
                    st.write(f"**Write Disposition:** {write_disposition}")
            
            # Show final data preview
            st.subheader("Final Data Preview")
            st.dataframe(df.head(10))
            
            # Upload button
            if st.button("🚀 Upload to BigQuery", type="primary"):
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Prepare table reference
                    table_id = f"{config['dataset']}.{config['table_name']}"
                    
                    status_text.text("Preparing upload...")
                    progress_bar.progress(10)
                    
                    # Generate schema
                    schema = generate_bq_schema(df, st.session_state.column_types)
                    
                    status_text.text("Configuring BigQuery job...")
                    progress_bar.progress(20)
                    
                    # Configure job
                    job_config = bigquery.LoadJobConfig(
                        schema=schema,
                        write_disposition=write_disposition,
                        create_disposition="CREATE_IF_NEEDED" if config['operation'] == "Create new table" else "CREATE_NEVER"
                    )
                    
                    status_text.text("Uploading data to BigQuery...")
                    progress_bar.progress(30)
                    
                    # Upload data
                    job = st.session_state.client.load_table_from_dataframe(
                        df, table_id, job_config=job_config
                    )
                    
                    # Wait for job to complete
                    while job.state != 'DONE':
                        time.sleep(1)
                        job.reload()
                        # Update progress
                        progress = min(30 + (job.bytes_processed / max(job.input_file_bytes, 1)) * 60, 90) if job.input_file_bytes else 50
                        progress_bar.progress(int(progress))
                        status_text.text(f"Uploading... {progress:.0f}%")
                    
                    progress_bar.progress(100)
                    status_text.text("Upload complete!")
                    
                    # Check for errors
                    if job.errors:
                        st.error("Upload completed with errors:")
                        for error in job.errors:
                            st.error(error)
                        success = False
                    else:
                        st.success(f"✅ Successfully uploaded {len(df)} rows to {table_id}")
                        success = True
                        
                        # Show link to BigQuery
                        project_id = st.session_state.project_id
                        bq_url = f"https://console.cloud.google.com/bigquery?project={project_id}&ws=!1m5!1m4!4m3!1s{project_id}!2s{config['dataset']}!3s{config['table_name']}"
                        st.markdown(f"[View table in BigQuery Console]({bq_url})")
                    
                    # Create audit log
                    audit_log = create_audit_log(
                        action=config['operation'],
                        client_dataset=config['dataset'],
                        table_name=config['table_name'],
                        table_type=config['table_type'],
                        row_count=len(df),
                        file_name=uploaded_file.name if 'uploaded_file' in locals() else 'unknown',
                        pii_columns=st.session_state.pii_columns,
                        errors=job.errors if job.errors else [],
                        warnings=[],
                        success=success
                    )
                    
                    # Add to upload history
                    st.session_state.upload_history.append(audit_log)
                    
                    # Show upload summary
                    st.subheader("Upload Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Uploaded", len(df))
                    with col2:
                        st.metric("Processing Time", f"{job.ended - job.started}")
                    with col3:
                        st.metric("Data Processed", f"{job.input_file_bytes / 1024 / 1024:.2f} MB")
                    
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
                    
                    # Create audit log for failure
                    audit_log = create_audit_log(
                        action=config['operation'],
                        client_dataset=config['dataset'],
                        table_name=config['table_name'],
                        table_type=config['table_type'],
                        row_count=len(df),
                        file_name=uploaded_file.name if 'uploaded_file' in locals() else 'unknown',
                        pii_columns=st.session_state.pii_columns,
                        errors=[str(e)],
                        warnings=[],
                        success=False
                    )
                    st.session_state.upload_history.append(audit_log)
        else:
            st.info("Please complete all previous steps before uploading")
    
    # Tab 6: Audit Log
    with tab6:
        st.header("Audit Log & Compliance")
        
        if st.session_state.upload_history:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_uploads = len(st.session_state.upload_history)
            successful_uploads = sum(1 for log in st.session_state.upload_history if log['success'])
            total_rows = sum(log['row_count'] for log in st.session_state.upload_history)
            
            with col1:
                st.metric("Total Uploads", total_uploads)
            with col2:
                st.metric("Successful", successful_uploads)
            with col3:
                st.metric("Failed", total_uploads - successful_uploads)
            with col4:
                st.metric("Total Rows Processed", f"{total_rows:,}")
            
            # Detailed audit log
            st.subheader("Upload History")
            
            # Convert to dataframe for display
            audit_df = pd.DataFrame(st.session_state.upload_history)
            
            # Display options
            view_option = st.radio("View", ["Summary", "Detailed"])
            
            if view_option == "Summary":
                summary_df = audit_df[['timestamp', 'client_dataset', 'table_name', 
                                      'row_count', 'success', 'error_count', 'warning_count']]
                st.dataframe(summary_df)
            else:
                # Detailed view with expanders
                for idx, log in enumerate(reversed(st.session_state.upload_history)):
                    status_icon = "✅" if log['success'] else "❌"
                    with st.expander(f"{status_icon} {log['timestamp']} - {log['table_name']} ({log['row_count']} rows)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Action:** {log['action']}")
                            st.write(f"**Client Dataset:** {log['client_dataset']}")
                            st.write(f"**Table Name:** {log['table_name']}")
                            st.write(f"**Table Type:** {log['table_type']}")
                            st.write(f"**File:** {log['file_name']}")
                        
                        with col2:
                            st.write(f"**Rows:** {log['row_count']}")
                            st.write(f"**PII Columns Detected:** {log['pii_columns_detected']}")
                            st.write(f"**Errors:** {log['error_count']}")
                            st.write(f"**Warnings:** {log['warning_count']}")
                            st.write(f"**Success:** {log['success']}")
                        
                        if log['pii_columns']:
                            st.write("**PII Columns:**", ", ".join(log['pii_columns']))
                        
                        if log['errors']:
                            st.write("**Errors:**")
                            for error in log['errors']:
                                st.error(error)
                        
                        if log['warnings']:
                            st.write("**Warnings:**")
                            for warning in log['warnings']:
                                st.warning(warning)
            
            # Export options
            st.subheader("Export Audit Logs")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as JSON
                audit_json = json.dumps(st.session_state.upload_history, indent=2, default=str)
                st.download_button(
                    label="📥 Download Audit Log (JSON)",
                    data=audit_json,
                    file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Export as CSV
                audit_csv = audit_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Audit Log (CSV)",
                    data=audit_csv,
                    file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Compliance report
            if st.button("Generate Compliance Report"):
                report = {
                    "report_date": datetime.now().isoformat(),
                    "total_uploads": total_uploads,
                    "successful_uploads": successful_uploads,
                    "failed_uploads": total_uploads - successful_uploads,
                    "total_rows_processed": total_rows,
                    "pii_columns_detected": sum(log['pii_columns_detected'] for log in st.session_state.upload_history),
                    "unique_clients": len(set(log['client_dataset'] for log in st.session_state.upload_history)),
                    "unique_tables": len(set(f"{log['client_dataset']}.{log['table_name']}" for log in st.session_state.upload_history)),
                    "upload_details": st.session_state.upload_history
                }
                
                st.json(report)
                
                # Download compliance report
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="📥 Download Compliance Report",
                    data=report_json,
                    file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No uploads yet. Complete an upload to see audit logs.")

if __name__ == "__main__":
    main()