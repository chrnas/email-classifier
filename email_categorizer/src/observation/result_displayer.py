from .observer import Observer

class ResultDisplayer(Observer):
    
    def __init__(self):
        self.predictions = []

    def update(self, event_type, predictions):
        """Updates predictions when a relevant event occurs and triggers display."""
        if event_type != 'predicting':
            return
        # Handle relevant event
        self.predictions = predictions
        self.display()

    def display(self):
        """Prints predictions along with email content."""
        print("\n")
        for prediction, email_content in self.predictions:
            print(f"Prediction: {prediction} | Email: {email_content}")
            print("\n")