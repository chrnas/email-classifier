import csv
from src.observation.observer import Observer
from src.classifier_config_singleton import ClassifierConfigSingleton


class ResultDisplayer(Observer):

    def __init__(self):
        self.predictions = []
        self.config = ClassifierConfigSingleton()

    def update(self, event_type, predictions):
        """Updates predictions when a relevant event occurs and triggers display."""
        if event_type != 'predicting':
            return
        # Handle relevant event
        self.predictions = predictions
        self.display()
        self.save_output()

    def display(self):
        """Prints predictions along with email content."""
        print("\n")
        for prediction, email_content in self.predictions:
            print(f"Prediction: {prediction} | Email: {email_content}")
            print("\n")

    def save_output(self):
        rows = [["Prediction", "Email Content"]]
        file_path = self.config.out_folder_path + "predictions.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for prediction, email_content in self.predictions:
                rows.append([prediction, email_content])
            writer.writerows(rows)
