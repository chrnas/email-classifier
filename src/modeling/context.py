from src.modeling.base import BaseModel
from src.modeling.training_data import TrainingData


class ContextClassifier():
    """Manages model strategies and notifies observers about events."""

    def __init__(self, data: TrainingData) -> None:
        self.modelstrat = None
        self.data = data
        self._observers = []

    def subscribe(self, observer):
        """Add an observer to the list."""
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe(self, observer):
        """Remove an observer from the list."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event_type: str, smth):
        """Notify all observers of an event."""
        for observer in self._observers:
            observer.update(event_type, smth)

    def choose_strat(self, modelstrat: BaseModel):
        """Set the model strategy."""
        self.modelstrat = modelstrat

    def train(self):
        """Train the selected model strategy."""
        self.modelstrat.train(self.data)

    def predict(self):
        """Make predictions using the selected model strategy."""
        self.modelstrat.predict(self.data)

    def classification_report(self):
        """Generate a classification report and notify observers."""
        report = self.modelstrat.classification_report(self.data)
        self.notify("evaluating", report)

    def predict_emails(self, email_embeddings, email_content):
        """Make predictions for email content and notify observers."""
        predictions = self.modelstrat.predict_emails(
            email_embeddings, email_content)
        self.notify("predicting", predictions)
        return predictions
