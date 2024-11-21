import numpy as np
from sklearn.svm import SVC
from .base import BaseModel

class SVC(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super().__init__(model_name, embeddings, y)
        self.mdl = SVC()

