import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base import BaseModel

class Bayes(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super().__init__(model_name, embeddings, y)
        self.mdl = GaussianNB()
