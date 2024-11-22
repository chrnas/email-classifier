from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix
from training_data import TrainingData


class BaseModel(ABC):
    def __init__(self) -> None:
        self.predictions = None

    def train(self, data: TrainingData) -> None:
        """Train the model using the provided training data."""
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, data: TrainingData) -> None:
        """Generate predictions for the test data and store them in the predictions attribute."""
        predictions = self.mdl.predict(data.X_test)
        self.predictions = predictions

    def classification_report(self, data: TrainingData):
        """Generate and return a classification report based on the test data and model predictions."""
        report = classification_report(
            data.y_test, self.predictions, output_dict=True,zero_division=0)
        return report


    def predict_emails(self, email_embeddings, email_contents):
        """Predict the class for each email based on its embeddings and return a list of (prediction, email) pairs."""
        predictions = self.mdl.predict(email_embeddings)
        results = []
        for email, prediction in zip(email_contents, predictions):
            results.append((prediction, email))
        return results
