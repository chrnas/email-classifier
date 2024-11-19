
import numpy as np
from .bayes import Bayes
from .randomforest import RandomForest
from .svc import SVC
from .random_forest_old import RandomForestOld


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, embeddings: np.ndarray, y: np.ndarray):

        if model_type == "bayes":
            return Bayes(model_type, embeddings, y)
        elif model_type == "randomforest":
            return RandomForest(model_type, embeddings, y)
        elif model_type == "svc":
            return SVC(model_type, embeddings, y)
        elif model_type == "test_forest":
            return RandomForestOld(model_type, embeddings, y)
        else:
            raise ValueError(f"Unknown algorithm: {model_type}.")
