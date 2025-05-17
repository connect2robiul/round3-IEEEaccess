import torch.nn as nn
from config import Config

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