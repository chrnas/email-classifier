from src.observation.observer import Observer


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
        print(self.statistics)
