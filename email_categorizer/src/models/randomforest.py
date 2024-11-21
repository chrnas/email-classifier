import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel

class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super().__init__(model_name, embeddings, y)
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
