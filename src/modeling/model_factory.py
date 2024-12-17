from src.modeling.bayes import Bayes
from src.modeling.randomforest import RandomForest
from src.modeling.svc import SVCModel


class ModelFactory:
    @staticmethod
    def create_model(model_type: str):
        """Creates a model based on the specified type ('bayes', 'randomforest', or 'svc')."""
        if model_type == "bayes":
            return Bayes()
        elif model_type == "randomforest":
            return RandomForest()
        elif model_type == "svc":
            return SVCModel()
        else:
            raise ValueError(f"Unknown algorithm: {model_type}.")
