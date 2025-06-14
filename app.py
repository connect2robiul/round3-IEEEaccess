import streamlit as st
import pandas as pd
import numpy as np
#import torch 
from eeg_processing import EEG, createdataframe , add 
from model import RobiulModel
from supabase_utils import get_supabase_client
from config import Config
import pandas as pd 
from tqdm import trange
from datasets import Dataset, DatasetDict
from tqdm import trange, tqdm


st.title("Brain Hemisphere Analysis App")

def safe_torch_import():
    global torch
    import torch


st.write("Add function output:", add(2, 3))


import os
st.write("Current working directory:", os.getcwd())




df = pd.DataFrame()

for experiment in Config.DATASETS:
    for rhythm in Config.RHYTHMS:
        participants = trange(1, 6, leave=True)
        for user in participants:
            df = pd.concat([df, createdataframe(user, experiment, rhythm)], ignore_index=True)
            participants.set_postfix(rhythm=rhythm, user=user, counts=len(df), experiment=experiment, refresh=True)

dataset = DatasetDict()
splits = df['frequency'].unique()

for split in tqdm(splits):
    dataset[split] = Dataset.from_pandas(df[df['frequency'] == split])
    st.write(dataset)


st.stop() 






# df = pd.DataFrame()

# for experiment in Config.EXPERIMENTS:
#     for rhythm in Config.RHYTHMS:
#         participants = trange(1, 6, leave=True)
#         for user in participants:
#             df = pd.concat([df, createdataframe(user, experiment, rhythm)], ignore_index=True)
#             participants.set_postfix(rhythm=rhythm, user=user, counts=len(df),  experiment=experiment, refresh=True)



# dataset = DatasetDict()
# splits = df['frequency'].unique()

# for split in tqdm(splits):
#     # dataset = DatasetDict(
#     #     {split: Dataset.from_pandas(df[df['frequency'] == split])}
#     # )
#     dataset[split] = Dataset.from_pandas(df[df['frequency'] == split], split=split)
#     print(dataset) 



















# # Sidebar
# st.sidebar.header("Settings")
# participant = st.sidebar.selectbox("Select Participant", ["Participant 1", "Participant 2", "Participant 3", "Participant 4", "Participant 5"])
# dataset_type = st.sidebar.selectbox("Select Dataset Type", ["Mona Lisa", "Necker cube"])

# # Load EEG data
# eeg_data = EEG(participant, dataset_type)

# # Train model
# if st.button("Train Model"):
#     st.info("Training model...")
#     model = RobiulModel(model_name="Small")
#     optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
#     criterion = torch.nn.BCELoss()

#     X = torch.tensor(eeg_data.alpha[0.5], dtype=torch.float32)
#     y = torch.tensor([0, 1], dtype=torch.float32)  # Example labels

#     for epoch in range(Config.EPOCHS):
#         optimizer.zero_grad()
#         outputs = model(X)
#         loss = criterion(outputs, y)
#         loss.backward()
#         optimizer.step()
#         st.write(f"Epoch {epoch+1}/{Config.EPOCHS}, Loss: {loss.item()}")

#     st.success("Model trained!")

# # Evaluate model
# if st.button("Evaluate Model"):
#     st.info("Evaluating model...")
#     # Add evaluation logic here
#     st.success("Evaluation complete!")

# # Visualize results
# st.subheader("Visualizations")
# st.line_chart(eeg_data.alpha[0.5][:, :10])  # Example visualization

# # Save results to Supabase
# client = get_supabase_client()
# create_tables(client)

# # Example: Save metrics to Supabase
# metrics = {
#     "kappa": 0.0,
#     "roc_auc": 0.0,
#     "t_test": 0.0,
#     "p_value": 0.00,
#     "f1_score": 0.00,
#     "tp": 00,
#     "tn": 00,
#     "fp": 00,
#     "fn": 00
# }
# client.table("metrics").insert(metrics).execute()

# st.write("Metrics saved to Supabase:")
# st.json(metrics)