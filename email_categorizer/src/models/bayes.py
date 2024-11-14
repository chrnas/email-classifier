import numpy as np
from sklearn.naive_bayes import GaussianNB
import random
from .base import BaseModel

num_folds = 0
seed = 0
# Data
np.random.seed(seed)
random.seed(seed)

# This file already contain the code for implementing randomforest model
# Carefully observe the methods below and try calling them in modelling.py


class Bayes(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = GaussianNB()
        self.predictions = None
        self.data_transform()
        print("bayes")

    def data_transform(self) -> None:
        ...

    def use_model(self, email: str):
        ...