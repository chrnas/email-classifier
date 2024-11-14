
import numpy as np
from .bayes import Bayes
from .randomforest import RandomForest
from .svc import SVC


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, model_name: str, embeddings: np.ndarray, y: np.ndarray):

        if model_type == "bayes":
            return Bayes(model_name, embeddings, y)
        elif model_type == "randomforest":
            return RandomForest(model_name, embeddings, y)
        elif model_type == "svc":
            return SVC(model_name, embeddings, y)
        else:
            raise ValueError(f"Unknown algorithm: {model_type}.")
