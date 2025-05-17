import supabase
from dotenv import load_dotenv
import os

load_dotenv()

def get_supabase_client():
    url = os.getenv("DATABASE_URL")
    key = url.split("@")[0].split("//")[1]
    client = supabase.create_client(url, key)
    return client

def create_tables(client):
    # Participants table
    client.table("participants").create({
        "id": {"type": "integer"},
        "name": {"type": "text"}
    }).execute()

    # Datasets table
    client.table("datasets").create({
        "id": {"type": "integer"},
        "participant_id": {"type": "integer"},
        "type": {"type": "text"},
        "intensity": {"type": "float"}
    }).execute()

    # Metrics table
    client.table("metrics").create({
        "id": {"type": "integer"},
        "dataset_id": {"type": "integer"},
        "kappa": {"type": "float"},
        "roc_auc": {"type": "float"},
        "t_test": {"type": "float"},
        "p_value": {"type": "float"},
        "f1_score": {"type": "float"},
        "tp": {"type": "integer"},
        "tn": {"type": "integer"},
        "fp": {"type": "integer"},
        "fn": {"type": "integer"}
    }).execute()

    # Predictions table
    client.table("predictions").create({
        "id": {"type": "integer"},
        "dataset_id": {"type": "integer"},
        "predicted_label": {"type": "integer"},
        "true_label": {"type": "integer"}
    }).execute()