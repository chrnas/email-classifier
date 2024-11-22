from models.base import BaseModel
from training_data import TrainingData


class ContextClassifier():
    def __init__(self, data: TrainingData) -> None:
        self.modelstrat = None
        self.data = data
        self._observers = []

    # Observer management methods
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
        self.modelstrat = modelstrat

    def train(self):
        self.modelstrat.train(self.data)

    def predict(self):
        self.modelstrat.predict(self.data)

    def classification_report(self):
        report = self.modelstrat.classification_report(self.data)
        self.notify("evaluating", report)

    def print_results(self):
        self.modelstrat.print_results(self.data)

    def predict_emails(self, email_embeddings, email_content):
        predictions = self.modelstrat.predict_emails(
            email_embeddings, email_content)
        self.notify("predicting", predictions)
