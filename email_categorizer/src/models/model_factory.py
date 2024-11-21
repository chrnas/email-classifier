import numpy as np
from .bayes import Bayes
from .randomforest import RandomForest
from .svc import SVCModel


class ModelFactory:
    @staticmethod
    def create_model(model_type: str):

        if model_type == "bayes":
            return Bayes(model_type)
        elif model_type == "randomforest":
            return RandomForest(model_type)
        elif model_type == "svc":
            return SVCModel(model_type)
        else:
            raise ValueError(f"Unknown algorithm: {model_type}.")
