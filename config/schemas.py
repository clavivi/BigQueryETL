"""Schema definitions for BigQuery tables"""

from google.cloud import bigquery
from typing import List

def get_gift_file_schema() -> List[bigquery.SchemaField]:
    """Schema for gift/donation files"""
    return [
        bigquery.SchemaField("donor_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("first_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("last_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("email", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("phone", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("address", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("city", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("state", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("zip_code", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("gift_amount", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("gift_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("gift_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("campaign_code", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("payment_method", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("transaction_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("notes", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
    ]

def get_campaign_donor_schema() -> List[bigquery.SchemaField]:
    """Schema for campaign donor files"""
    return [
        bigquery.SchemaField("donor_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("campaign_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("first_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("last_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("email", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("phone", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("address", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("city", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("state", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("zip_code", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("total_donated", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("donation_count", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("first_donation_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("last_donation_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("donor_status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("donor_category", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("opt_in_email", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("opt_in_sms", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
    ]

def get_flexible_schema() -> List[bigquery.SchemaField]:
    """Flexible schema that accepts STRING for all fields (for initial upload)"""
    return []