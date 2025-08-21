"""BigQuery operations module"""

import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from config.settings import TABLE_NAME_PATTERNS, ERROR_MESSAGES, SUCCESS_MESSAGES
from config.schemas import get_gift_file_schema, get_campaign_donor_schema, get_flexible_schema

def list_datasets(client: bigquery.Client) -> List[str]:
    """
    List all available datasets (clients) in the project
    
    Args:
        client: BigQuery client object
        
    Returns:
        List of dataset IDs
    """
    try:
        datasets = []
        for dataset in client.list_datasets():
            datasets.append(dataset.dataset_id)
        
        if not datasets:
            st.warning(ERROR_MESSAGES["no_datasets"])
            
        return datasets
        
    except Exception as e:
        st.error(f"Error listing datasets: {str(e)}")
        return []

def list_tables(client: bigquery.Client, dataset_id: str) -> List[str]:
    """
    List all tables in a specific dataset
    
    Args:
        client: BigQuery client object
        dataset_id: Dataset ID to list tables from
        
    Returns:
        List of table IDs
    """
    try:
        tables = []
        dataset_ref = client.dataset(dataset_id)
        
        for table in client.list_tables(dataset_ref):
            tables.append(table.table_id)
            
        return tables
        
    except Exception as e:
        st.error(f"Error listing tables: {str(e)}")
        return []

def identify_table_type(table_name: str) -> str:
    """
    Identify table type based on naming patterns
    
    Args:
        table_name: Name of the table
        
    Returns:
        Table type ('gift_file', 'campaign_donor', or 'unknown')
    """
    table_name_lower = table_name.lower()
    
    for table_type, patterns in TABLE_NAME_PATTERNS.items():
        for pattern in patterns:
            if pattern in table_name_lower:
                return table_type
                
    return "unknown"

def get_table_info(client: bigquery.Client, dataset_id: str, table_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a table
    
    Args:
        client: BigQuery client object
        dataset_id: Dataset ID
        table_id: Table ID
        
    Returns:
        Dictionary containing table information
    """
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)
        
        return {
            "num_rows": table.num_rows,
            "num_columns": len(table.schema),
            "created": table.created,
            "modified": table.modified,
            "size_mb": table.num_bytes / (1024 * 1024) if table.num_bytes else 0,
            "table_type": identify_table_type(table_id)
        }
        
    except Exception as e:
        st.error(f"Error getting table info: {str(e)}")
        return {}

def create_table(
    client: bigquery.Client, 
    dataset_id: str, 
    table_id: str, 
    table_type: str,
    use_flexible_schema: bool = True
) -> bool:
    """
    Create a new table with the appropriate schema
    
    Args:
        client: BigQuery client object
        dataset_id: Dataset ID
        table_id: Table ID
        table_type: Type of table ('gift_file' or 'campaign_donor')
        use_flexible_schema: Whether to use flexible schema (auto-detect)
        
    Returns:
        Boolean indicating success
    """
    try:
        dataset_ref = client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        
        if use_flexible_schema:
            schema = get_flexible_schema()
        else:
            if table_type == "gift_file":
                schema = get_gift_file_schema()
            elif table_type == "campaign_donor":
                schema = get_campaign_donor_schema()
            else:
                schema = get_flexible_schema()
        
        table = bigquery.Table(table_ref, schema=schema)
        
        if not use_flexible_schema:
            table = client.create_table(table)
        else:
            job_config = bigquery.LoadJobConfig(
                autodetect=True,
                write_disposition=bigquery.WriteDisposition.WRITE_EMPTY,
            )
            
        st.success(f"{SUCCESS_MESSAGES['table_created']}: {dataset_id}.{table_id}")
        return True
        
    except Exception as e:
        st.error(f"Error creating table: {str(e)}")
        return False

def upload_csv_to_table(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    dataframe: pd.DataFrame,
    write_mode: str = "append"
) -> bool:
    """
    Upload CSV data to BigQuery table
    
    Args:
        client: BigQuery client object
        dataset_id: Dataset ID
        table_id: Table ID
        dataframe: Pandas DataFrame containing the data
        write_mode: 'append' or 'replace'
        
    Returns:
        Boolean indicating success
    """
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=0,
            autodetect=True,
            write_disposition=(
                bigquery.WriteDisposition.WRITE_TRUNCATE 
                if write_mode == "replace" 
                else bigquery.WriteDisposition.WRITE_APPEND
            ),
        )
        
        dataframe['upload_timestamp'] = datetime.now()
        
        job = client.load_table_from_dataframe(
            dataframe, 
            table_ref, 
            job_config=job_config
        )
        
        job.result()
        
        st.success(f"{SUCCESS_MESSAGES['upload_success']}: {len(dataframe)} rows uploaded to {dataset_id}.{table_id}")
        return True
        
    except Exception as e:
        st.error(f"{ERROR_MESSAGES['upload_failed']}: {str(e)}")
        return False

def validate_csv_structure(df: pd.DataFrame, table_type: str) -> Tuple[bool, List[str]]:
    """
    Validate CSV structure against expected schema
    
    Args:
        df: Pandas DataFrame to validate
        table_type: Type of table to validate against
        
    Returns:
        Tuple of (is_valid, list of warnings/errors)
    """
    warnings = []
    
    if df.empty:
        return False, ["CSV file is empty"]
    
    if len(df.columns) == 0:
        return False, ["No columns found in CSV"]
    
    expected_columns = {
        "gift_file": [
            "donor_id", "first_name", "last_name", "email", 
            "gift_amount", "gift_date"
        ],
        "campaign_donor": [
            "donor_id", "campaign_id", "first_name", "last_name", 
            "email", "total_donated"
        ]
    }
    
    if table_type in expected_columns:
        missing_cols = []
        for col in expected_columns[table_type]:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            warnings.append(f"Missing recommended columns: {', '.join(missing_cols)}")
    
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        return False, [f"Duplicate columns found: {', '.join(duplicate_cols)}"]
    
    for col in df.columns:
        if df[col].isna().all():
            warnings.append(f"Column '{col}' contains only null values")
    
    return True, warnings

def get_table_preview(client: bigquery.Client, dataset_id: str, table_id: str, limit: int = 10) -> pd.DataFrame:
    """
    Get a preview of table data
    
    Args:
        client: BigQuery client object
        dataset_id: Dataset ID
        table_id: Table ID
        limit: Number of rows to preview
        
    Returns:
        DataFrame with preview data
    """
    try:
        query = f"""
        SELECT *
        FROM `{client.project}.{dataset_id}.{table_id}`
        LIMIT {limit}
        """
        
        return client.query(query).to_dataframe()
        
    except Exception as e:
        st.error(f"Error getting table preview: {str(e)}")
        return pd.DataFrame()

def check_table_exists(client: bigquery.Client, dataset_id: str, table_id: str) -> bool:
    """
    Check if a table exists in the dataset
    
    Args:
        client: BigQuery client object
        dataset_id: Dataset ID
        table_id: Table ID
        
    Returns:
        Boolean indicating if table exists
    """
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        client.get_table(table_ref)
        return True
    except:
        return False