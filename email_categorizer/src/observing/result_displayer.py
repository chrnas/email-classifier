from .observe import Observer
from context_classification.context import ContextClassifier
from pandas import DataFrame

class ResultDisplayer(Observer):
    
    def __init__(self):
        self.predictions = []

    def update(self, event_type, predictions):
        if event_type != 'predicting':
            return  # Ignore irrelevant events
        # Handle relevant event
        self.predictions = predictions
        self.display()

    def display(self):
        # Print collected results
        print("\n")
        for prediction, email_content in self.predictions:
            print(f"Prediction: {prediction} | Email: {email_content}")
            print("\n")