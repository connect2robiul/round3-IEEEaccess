import torch.nn as nn
from config import Config
import os
from dotenv import load_dotenv

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from scipy.stats import ttest_ind, f_oneway
from sqlalchemy import create_engine, Table, Column, String, Float, DateTime, MetaData
from datetime import datetime
import uuid # Import uuid for a more robust ID generation

class RobiulModel(nn.Module):
    def __init__(self, model_name='Small'):
        super(RobiulModel, self).__init__()
        self.model_name = model_name
        self.in_features = 15000
        self.out_features = 6400
        self.activation = nn.ReLU()

        if model_name == 'Small':
            self.linear_stack = nn.Sequential(
                nn.Linear(self.in_features, self.out_features),
                self.activation,
                nn.BatchNorm1d(self.out_features),
                nn.Linear(self.out_features, 10),
                self.activation,
                nn.Linear(10, 1),
                nn.Sigmoid()
            )
        elif model_name == 'Big':
            self.linear_stack = nn.Sequential(
                nn.Linear(self.in_features, self.out_features),
                self.activation,
                nn.BatchNorm1d(self.out_features),
                nn.Linear(self.out_features, 1000),
                self.activation,
                nn.Linear(1000, 500),
                self.activation,
                nn.Linear(500, 2),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        return self.linear_stack(x)
    

load_dotenv()


engine = create_engine(os.getenv("DATABASE_URL"))
metadata = MetaData() 

matrix_table = Table(
    'matrix', metadata,
    Column('id', String, primary_key=True), # Use String for UUID
    Column('rhythm', String),
    Column('dataset', String),
    Column('model', String),
    Column('timestamp', DateTime),
    Column('accuracy', Float),
    Column('precision', Float),
    Column('recall', Float),
    Column('specificity', Float),
    Column('f1_score', Float),
    Column('roc_auc', Float),
    Column('kappa', Float),
    Column('ttest_pvalue', Float),
    Column('anova_pvalue', Float)
)

# Create table if it doesn't exist
metadata.create_all(engine)

