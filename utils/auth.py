"""Authentication module for Google BigQuery"""

import json
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Optional, Dict, Any
from config.settings import SESSION_STATE_KEYS, ERROR_MESSAGES, SUCCESS_MESSAGES

def parse_service_account_key(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Parse and validate the uploaded service account key file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dictionary containing service account credentials or None if invalid
    """
    try:
        content = uploaded_file.read()
        credentials_dict = json.loads(content)
        
        required_fields = [
            "type", "project_id", "private_key_id", 
            "private_key", "client_email"
        ]
        
        for field in required_fields:
            if field not in credentials_dict:
                st.error(f"Missing required field: {field}")
                return None
                
        if credentials_dict["type"] != "service_account":
            st.error("Invalid credential type. Expected 'service_account'")
            return None
            
        return credentials_dict
        
    except json.JSONDecodeError:
        st.error("Invalid JSON format in service account key file")
        return None
    except Exception as e:
        st.error(f"Error parsing service account key: {str(e)}")
        return None

def authenticate_bigquery(credentials_dict: Dict[str, Any]) -> Optional[bigquery.Client]:
    """
    Authenticate with BigQuery using service account credentials
    
    Args:
        credentials_dict: Dictionary containing service account credentials
        
    Returns:
        BigQuery client object or None if authentication fails
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        
        client = bigquery.Client(
            credentials=credentials,
            project=credentials_dict["project_id"]
        )
        
        list(client.list_datasets(max_results=1))
        
        return client
        
    except Exception as e:
        st.error(f"{ERROR_MESSAGES['auth_failed']}: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables for authentication"""
    if SESSION_STATE_KEYS["authenticated"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["authenticated"]] = False
    if SESSION_STATE_KEYS["auth_client"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["auth_client"]] = None
    if SESSION_STATE_KEYS["credentials"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["credentials"]] = None
    if SESSION_STATE_KEYS["selected_dataset"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["selected_dataset"]] = None
    if SESSION_STATE_KEYS["selected_table"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["selected_table"]] = None

def handle_authentication(uploaded_file) -> bool:
    """
    Handle the complete authentication flow
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Boolean indicating success of authentication
    """
    if uploaded_file is None:
        return False
        
    credentials_dict = parse_service_account_key(uploaded_file)
    if credentials_dict is None:
        return False
        
    client = authenticate_bigquery(credentials_dict)
    if client is None:
        return False
        
    st.session_state[SESSION_STATE_KEYS["auth_client"]] = client
    st.session_state[SESSION_STATE_KEYS["credentials"]] = credentials_dict
    st.session_state[SESSION_STATE_KEYS["authenticated"]] = True
    
    st.success(SUCCESS_MESSAGES["auth_success"])
    st.info(f"Project ID: {credentials_dict['project_id']}")
    
    return True

def get_client() -> Optional[bigquery.Client]:
    """Get the authenticated BigQuery client from session state"""
    return st.session_state.get(SESSION_STATE_KEYS["auth_client"])

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get(SESSION_STATE_KEYS["authenticated"], False)

def get_project_id() -> Optional[str]:
    """Get the current project ID"""
    credentials = st.session_state.get(SESSION_STATE_KEYS["credentials"])
    if credentials:
        return credentials.get("project_id")
    return None

def reset_authentication():
    """Reset authentication state"""
    st.session_state[SESSION_STATE_KEYS["authenticated"]] = False
    st.session_state[SESSION_STATE_KEYS["auth_client"]] = None
    st.session_state[SESSION_STATE_KEYS["credentials"]] = None
    st.session_state[SESSION_STATE_KEYS["selected_dataset"]] = None
    st.session_state[SESSION_STATE_KEYS["selected_table"]] = None