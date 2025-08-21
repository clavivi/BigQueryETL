"""Main Streamlit application for BigQuery Donation Data Upload System"""

import streamlit as st
import pandas as pd
from config.settings import (
    APP_TITLE, APP_ICON, MAX_FILE_SIZE_MB, PREVIEW_ROWS,
    SESSION_STATE_KEYS, ERROR_MESSAGES
)
from utils.auth import (
    initialize_session_state, handle_authentication, 
    get_client, is_authenticated, get_project_id, reset_authentication
)
from utils.bigquery_ops import (
    list_datasets, list_tables, identify_table_type,
    get_table_info, create_table, upload_csv_to_table,
    validate_csv_structure, get_table_preview, check_table_exists
)
from utils.data_cleaning import (
    clean_dataframe, get_cleaning_summary, validate_cleaned_data,
    detect_date_columns, detect_amount_columns, detect_data_types
)

def display_editable_summary(summary_df):
    """Display and allow editing of column summary using standard Streamlit components"""
    st.write("**Column Mapping & Data Types** (Edit below)")
    
    edited_summary = []
    
    for idx, row in summary_df.iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            st.text(row['Original Name'])
        
        with col2:
            new_name = st.text_input(
                f"Cleaned Name {idx}",
                value=row['Cleaned Name'],
                key=f"clean_name_{idx}",
                label_visibility="collapsed"
            )
        
        with col3:
            new_type = st.selectbox(
                f"Data Type {idx}",
                options=['STRING', 'INT64', 'FLOAT64', 'DATE', 'DATETIME', 'BOOL'],
                index=['STRING', 'INT64', 'FLOAT64', 'DATE', 'DATETIME', 'BOOL'].index(row['Data Type']),
                key=f"data_type_{idx}",
                label_visibility="collapsed"
            )
        
        with col4:
            st.text(row['Special Processing'])
        
        edited_summary.append({
            'Original Name': row['Original Name'],
            'Cleaned Name': new_name,
            'Data Type': new_type,
            'Special Processing': row['Special Processing']
        })
    
    return pd.DataFrame(edited_summary)

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide"
    )
    
    st.title(f"{APP_ICON} {APP_TITLE}")
    
    initialize_session_state()
    
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'cleaning_report' not in st.session_state:
        st.session_state.cleaning_report = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    
    with st.sidebar:
        st.header("System Status")
        if is_authenticated():
            st.success("✅ Connected to BigQuery")
            st.info(f"Project: {get_project_id()}")
            if st.button("Disconnect", type="secondary"):
                reset_authentication()
                st.rerun()
        else:
            st.warning("⚠️ Not connected")
    
    st.markdown("---")
    
    st.header("Section 1: Authentication")
    
    if not is_authenticated():
        st.info("Upload your Google Cloud service account JSON key file to connect to BigQuery.")
        
        uploaded_key = st.file_uploader(
            "Choose service account key file",
            type=['json'],
            help="Upload the JSON key file for your Google Cloud service account"
        )
        
        if uploaded_key is not None:
            if handle_authentication(uploaded_key):
                st.rerun()
    else:
        st.success("✅ Authenticated successfully")
        client = get_client()
    
    if not is_authenticated():
        st.warning("Please authenticate to continue")
        st.stop()
    
    st.markdown("---")
    st.header("Section 2: Select Client (Dataset)")
    
    datasets = list_datasets(client)
    
    if datasets:
        selected_dataset = st.selectbox(
            "Select a client (dataset)",
            options=[""] + datasets,
            index=0,
            help="Each dataset represents a client in your system"
        )
        
        if selected_dataset:
            st.session_state[SESSION_STATE_KEYS["selected_dataset"]] = selected_dataset
    else:
        st.error("No datasets found. Please check your permissions.")
        st.stop()
    
    if not selected_dataset:
        st.warning("Please select a dataset to continue")
        st.stop()
    
    st.markdown("---")
    st.header("Section 3: Table Operation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        operation_mode = st.radio(
            "Choose operation",
            options=["Update existing table", "Create new table"],
            help="Select whether to update an existing table or create a new one"
        )
    
    with col2:
        file_type = st.radio(
            "Select file type",
            options=["Gift file", "Campaign donor file"],
            help="Choose the type of data you're uploading"
        )
    
    table_type = "gift_file" if file_type == "Gift file" else "campaign_donor"
    
    st.markdown("---")
    st.header("Section 4: Table Selection")
    
    if operation_mode == "Update existing table":
        tables = list_tables(client, selected_dataset)
        
        if tables:
            table_suggestions = []
            for table in tables:
                detected_type = identify_table_type(table)
                if detected_type == table_type:
                    table_suggestions.insert(0, table)
                else:
                    table_suggestions.append(table)
            
            selected_table = st.selectbox(
                "Select table to update",
                options=[""] + table_suggestions,
                index=0,
                help="Tables matching your file type are shown first"
            )
            
            if selected_table:
                st.session_state[SESSION_STATE_KEYS["selected_table"]] = selected_table
                
                table_info = get_table_info(client, selected_dataset, selected_table)
                if table_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{table_info.get('num_rows', 0):,}")
                    with col2:
                        st.metric("Columns", table_info.get('num_columns', 0))
                    with col3:
                        st.metric("Size", f"{table_info.get('size_mb', 0):.2f} MB")
                    
                    if st.checkbox("Show table preview"):
                        preview_df = get_table_preview(client, selected_dataset, selected_table)
                        if not preview_df.empty:
                            st.dataframe(preview_df, use_container_width=True)
        else:
            st.warning("No tables found in this dataset")
            selected_table = None
    
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            new_table_name = st.text_input(
                "Enter new table name",
                placeholder=f"e.g., {table_type}_2024_01",
                help="Table name can only contain letters, numbers, and underscores"
            )
        with col2:
            st.write("")
            st.write("")
            if new_table_name:
                if check_table_exists(client, selected_dataset, new_table_name):
                    st.error("Table already exists!")
                else:
                    st.success("Table name available")
        
        selected_table = new_table_name
    
    if not selected_table:
        st.warning("Please select or create a table to continue")
        st.stop()
    
    st.markdown("---")
    st.header("Section 5: Upload & Clean CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df.copy()
            
            st.success(f"File loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
            
            is_valid, warnings = validate_csv_structure(df, table_type)
            
            if not is_valid:
                st.error("CSV validation failed:")
                for warning in warnings:
                    st.error(f"• {warning}")
                st.stop()
            elif warnings:
                st.warning("CSV validation warnings:")
                for warning in warnings:
                    st.warning(f"• {warning}")
            
            st.markdown("---")
            st.subheader("📊 Data Cleaning & Processing")
            
            with st.expander("⚙️ Cleaning Options", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Date Format Settings**")
                    detected_dates = detect_date_columns(df)
                    if detected_dates:
                        st.info(f"Detected date columns: {', '.join(detected_dates.keys())}")
                        date_format = st.selectbox(
                            "Date format (if auto-detect fails)",
                            options=["Auto-detect", "MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD", "MM-DD-YYYY"],
                            help="Select the date format used in your CSV"
                        )
                    else:
                        st.info("No date columns detected")
                        date_format = "Auto-detect"
                
                with col2:
                    st.write("**Amount Format Settings**")
                    detected_amounts = detect_amount_columns(df)
                    if detected_amounts:
                        st.info(f"Detected amount columns: {', '.join(detected_amounts)}")
                    decimal_sep = st.selectbox(
                        "Decimal separator",
                        options=[".", ","],
                        help="Character used for decimal points"
                    )
                    thousands_sep = st.selectbox(
                        "Thousands separator",
                        options=[",", ".", "None"],
                        help="Character used for thousands"
                    )
                    thousands_sep = "" if thousands_sep == "None" else thousands_sep
                
                with col3:
                    st.write("**Cleaning Actions**")
                    st.write("")
                    if st.button("🔄 Clean Data", type="primary", use_container_width=True):
                        with st.spinner("Cleaning data..."):
                            date_formats = {}
                            if date_format != "Auto-detect":
                                format_map = {
                                    "MM/DD/YYYY": "%m/%d/%Y",
                                    "DD/MM/YYYY": "%d/%m/%Y",
                                    "YYYY-MM-DD": "%Y-%m-%d",
                                    "MM-DD-YYYY": "%m-%d-%Y"
                                }
                                for col in detected_dates:
                                    date_formats[col] = format_map[date_format]
                            else:
                                date_formats = detected_dates
                            
                            cleaned_df, cleaning_report = clean_dataframe(
                                df,
                                date_columns=date_formats,
                                amount_columns=detected_amounts,
                                decimal_separator=decimal_sep,
                                thousands_separator=thousands_sep
                            )
                            
                            st.session_state.cleaned_df = cleaned_df
                            st.session_state.cleaning_report = cleaning_report
                            st.success("✅ Data cleaned successfully!")
            
            if st.session_state.cleaned_df is not None:
                cleaned_df = st.session_state.cleaned_df
                cleaning_report = st.session_state.cleaning_report
                
                st.subheader("🔄 Column Mapping & Data Types")
                
                summary_df = get_cleaning_summary(cleaning_report)
                
                st.write("Column Mapping Summary:")
                st.dataframe(summary_df, use_container_width=True)
                
                is_valid, validation_messages = validate_cleaned_data(cleaned_df)
                
                st.subheader("✅ Data Validation")
                for message in validation_messages:
                    if "✅" in message:
                        st.success(message)
                    elif "❌" in message:
                        st.error(message)
                    else:
                        st.warning(message)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Original Data Preview (first {PREVIEW_ROWS} rows)")
                    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
                
                with col2:
                    st.subheader(f"Cleaned Data Preview (first {PREVIEW_ROWS} rows)")
                    st.dataframe(cleaned_df.head(PREVIEW_ROWS), use_container_width=True)
                
                if cleaning_report.get('date_columns'):
                    st.info(f"📅 Processed {len(cleaning_report['date_columns'])} date column(s)")
                
                if cleaning_report.get('amount_columns'):
                    st.info(f"💵 Processed {len(cleaning_report['amount_columns'])} amount column(s)")
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if operation_mode == "Update existing table":
                        write_mode = st.selectbox(
                            "Write mode",
                            options=["Append", "Replace"],
                            help="Append adds data to existing table, Replace overwrites all data"
                        )
                    else:
                        write_mode = "Replace"
                        st.info("New table will be created")
                
                with col2:
                    st.write("")
                    st.write("")
                    upload_button = st.button(
                        "📤 Upload to BigQuery",
                        type="primary",
                        use_container_width=True,
                        disabled=not is_valid
                    )
                
                if upload_button and is_valid:
                    with st.spinner("Uploading cleaned data to BigQuery..."):
                        
                        if operation_mode == "Create new table":
                            if not create_table(client, selected_dataset, selected_table, table_type, use_flexible_schema=True):
                                st.stop()
                        
                        success = upload_csv_to_table(
                            client=client,
                            dataset_id=selected_dataset,
                            table_id=selected_table,
                            dataframe=cleaned_df,
                            write_mode=write_mode.lower()
                        )
                        
                        if success:
                            st.balloons()
                            st.success(f"✅ Successfully uploaded {len(cleaned_df):,} rows to {selected_dataset}.{selected_table}")
                            
                            table_info = get_table_info(client, selected_dataset, selected_table)
                            if table_info:
                                st.info(f"Table now contains {table_info.get('num_rows', 0):,} total rows")
            else:
                st.info("👆 Click 'Clean Data' to process your CSV file")
                
                st.subheader(f"Raw Data Preview (first {PREVIEW_ROWS} rows)")
                st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")

if __name__ == "__main__":
    main()