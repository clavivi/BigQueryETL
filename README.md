# BigQuery Donation Data Upload System

A production-ready Streamlit application for uploading and processing donation CSV files to Google BigQuery with proper authentication, data validation, and compliance features.

## Features

- **Secure Authentication**: Upload Google Cloud service account JSON key for BigQuery access
- **Client Management**: Browse and select from available BigQuery datasets (clients)
- **Flexible Table Operations**: 
  - Update existing tables (append or replace data)
  - Create new tables with auto-detected schema
- **Smart File Type Detection**: Automatically identifies gift files vs campaign donor files
- **Data Validation**: Pre-upload validation with warnings for missing columns or data issues
- **Real-time Preview**: View CSV data before uploading
- **Table Information**: Display table statistics and preview existing data

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd BQ_Upload
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Upload your Google Cloud service account JSON key file

3. Select a client (BigQuery dataset)

4. Choose operation mode:
   - Update existing table
   - Create new table

5. Select file type:
   - Gift file (donation records)
   - Campaign donor file (donor information)

6. Upload your CSV file and review the preview

7. Click "Upload to BigQuery" to process the data

## Project Structure

```
BQ_Upload/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── config/
│   ├── __init__.py
│   ├── settings.py       # Application settings and constants
│   └── schemas.py        # BigQuery table schemas
└── utils/
    ├── __init__.py
    ├── auth.py           # Authentication module
    └── bigquery_ops.py   # BigQuery operations
```

## Security Notes

- Service account keys are handled securely in session state
- No credentials are stored locally
- All BigQuery operations use authenticated client connections
- PII data is handled according to BigQuery security best practices

## Requirements

- Python 3.8+
- Google Cloud Project with BigQuery API enabled
- Service account with BigQuery Data Editor permissions
- CSV files with donation or donor data

## Data Formats

### Gift File Schema
- donor_id, first_name, last_name, email
- gift_amount, gift_date, gift_type
- campaign_code, payment_method, transaction_id

### Campaign Donor File Schema  
- donor_id, campaign_id, first_name, last_name
- email, total_donated, donation_count
- first_donation_date, last_donation_date
- donor_status, donor_category

## Error Handling

The application includes comprehensive error handling for:
- Invalid authentication credentials
- Missing BigQuery permissions
- Malformed CSV files
- Network connectivity issues
- BigQuery quota limits

## Support

For issues or questions, please contact your system administrator or create an issue in the repository.