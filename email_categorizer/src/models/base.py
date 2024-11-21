from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix
from training_data import TrainingData
import numpy as np

class BaseModel(ABC):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.predictions = None

    def train(self, data: TrainingData) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, data) -> None:
        predictions = self.mdl.predict(data.X_test)
        self.predictions = predictions

    def classification_report(self, data) : 
        report = classification_report(data.get_y_test(), self.predictions, output_dict=True)
        return report

    def print_results(self, data):
        print(self.predictions)
        print(classification_report(data.get_y_test(), self.predictions))
        print(confusion_matrix(data.get_y_test(), self.predictions))

    def predict_emails(self, email_embeddings, email_contents):
        predictions = self.mdl.predict(email_embeddings)
        results = []
        for email, prediction in zip(email_contents, predictions):
            results.append((prediction, email))
        return results
    