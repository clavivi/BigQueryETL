"""Configuration settings for BigQuery Donation Upload System"""

from typing import Dict, List

APP_TITLE = "BigQuery Donation Data Upload System"
APP_ICON = "📊"

MAX_FILE_SIZE_MB = 200

TABLE_NAME_PATTERNS = {
    "gift_file": ["gift", "donation", "contribution"],
    "campaign_donor": ["campaign", "donor", "supporter"]
}

PREVIEW_ROWS = 10

SESSION_STATE_KEYS = {
    "auth_client": "bigquery_client",
    "authenticated": "is_authenticated", 
    "selected_dataset": "current_dataset",
    "selected_table": "current_table",
    "credentials": "gcp_credentials"
}

ERROR_MESSAGES = {
    "auth_failed": "Authentication failed. Please check your service account key.",
    "no_datasets": "No datasets found. Please check your permissions.",
    "upload_failed": "Upload failed. Please check your data and try again.",
    "invalid_file": "Invalid file format. Please upload a CSV file."
}

SUCCESS_MESSAGES = {
    "auth_success": "✅ Successfully authenticated with BigQuery",
    "upload_success": "✅ Data uploaded successfully",
    "table_created": "✅ Table created successfully"
}