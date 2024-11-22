from .observe import Observer
from context_classification.context import ContextClassifier


class StatCollector(Observer):

    def __init__(self):
        self.statistics = None

    def update(self, event_type, statistics):
        """Updates the statistics when the 'evaluating' event occurs."""
        if event_type != 'evaluating':
            return
        # Handle relevant event
        self.statistics = statistics
        self.display()

    def display(self):
        """Displays the collected statistics."""
        print("\n")
        for email_class, met in self.statistics.items():
            if email_class not in ["accuracy", "macro avg", "weighted avg"]:
                print(f"Email Type: {email_class}")
                print(f"Precision: {met['precision']}")
                print(f"Recall: {met['recall']}")
                print(f"F1-Score: {met['f1-score']}")
                print(f"Support: {met['support']}")
                print("\n")
            elif email_class is "accuracy":
                print("Overall Metrics:")
                print(f"Accuracy: {met:.3f}")
                print("\n")
