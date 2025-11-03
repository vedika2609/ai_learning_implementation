import os
import vertexai
from google.auth import default
from google.oauth2 import service_account
from google.cloud import bigquery

PROJECT_ID = "light-river-469808-p4"
REGION = "us-central1"

# Only use service_account.json if explicitly set
SERVICE_ACCOUNT_FILE = "gcp_creds.json"

def authenticate():
    """
    Authenticate using either:
      1. GOOGLE_APPLICATION_CREDENTIALS (local dev), or
      2. Application Default Credentials (Cloud Run).
    """
    credentials = None

    if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"🔑 Using local service account file: {SERVICE_ACCOUNT_FILE}")
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE
        )
    else:
        print("🌐 Using Application Default Credentials (Cloud Run).")
        credentials, project_id = default()
        global PROJECT_ID
        if project_id:
            PROJECT_ID = project_id

    print(f"✅ Authenticated to project '{PROJECT_ID}' in region '{REGION}'")
    return credentials, PROJECT_ID, REGION


def init_vertex_ai():
    credentials, project_id, region = authenticate()
    vertexai.init(project=project_id, location=region, credentials=credentials)
    return credentials, project_id, region


def get_bq_client():
    credentials, project_id, region = init_vertex_ai()
    return bigquery.Client(project=project_id, credentials=credentials)



if __name__ == '__main__':
    get_bq_client()