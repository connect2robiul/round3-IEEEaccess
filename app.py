import streamlit as st
import pandas as pd
import numpy as np
import torch
from eeg_processing import EEG
from model import RobiulModel
from supabase_utils import get_supabase_client
from config import Config

st.title("Brain Hemisphere Analysis App")

# Sidebar
st.sidebar.header("Settings")
participant = st.sidebar.selectbox("Select Participant", ["Participant 1", "Participant 2", "Participant 3", "Participant 4", "Participant 5"])
dataset_type = st.sidebar.selectbox("Select Dataset Type", ["Mona Lisa", "Necker cube"])

# Load EEG data
eeg_data = EEG(participant, dataset_type)

# Train model
if st.button("Train Model"):
    st.info("Training model...")
    model = RobiulModel(model_name="Small")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = torch.nn.BCELoss()

    X = torch.tensor(eeg_data.alpha[0.5], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.float32)  # Example labels

    for epoch in range(Config.EPOCHS):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        st.write(f"Epoch {epoch+1}/{Config.EPOCHS}, Loss: {loss.item()}")

    st.success("Model trained!")

# Evaluate model
if st.button("Evaluate Model"):
    st.info("Evaluating model...")
    # Add evaluation logic here
    st.success("Evaluation complete!")

# Visualize results
st.subheader("Visualizations")
st.line_chart(eeg_data.alpha[0.5][:, :10])  # Example visualization

# Save results to Supabase
client = get_supabase_client()
create_tables(client)

# Example: Save metrics to Supabase
metrics = {
    "kappa": 0.0,
    "roc_auc": 0.0,
    "t_test": 0.0,
    "p_value": 0.00,
    "f1_score": 0.00,
    "tp": 00,
    "tn": 00,
    "fp": 00,
    "fn": 00
}
client.table("metrics").insert(metrics).execute()

st.write("Metrics saved to Supabase:")
st.json(metrics)