import torch
import random
import numpy as np

class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-4
    LR_PATIENCE = 3
    INTENSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    DATASETS = ["Mona Lisa", "Necker cube"]
    RHYTHMS = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    MODELS = ['Big', 'Small', 'CNN']
    OPTIMIZER_LIST = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RMSprop', 'SGD']

    @staticmethod
    def set_seed(seed=SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False