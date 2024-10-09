from preprocessing import DataProcessor
from dataset_loader import DatasetLoader
from training_data import TrainingData
from randomforest import RandomForest


class EmailClassifier():

    def __init__(self) -> None:

        self.data_processor = None
        self.data_set_loader = DatasetLoader()
        self.model = None
        self.df = None
        self.data = None

    def classify_email(self, email: str) -> str:
        # This method will classify the email using the model, no idea how this will be used
        classification = self.model.use_model(email)
        print("Email classified")
        return classification

    def train_model(self, path: str) -> None:
        self.df = self.data_set_loader.get_input_data(path)
        self.data_processor = DataProcessor(self.df)
        self.data_processor.create_tfidf_embd()
        X = self.data_processor.get_tfidf_embd()
        self.data = TrainingData(X, self.df)
        self.model = RandomForest(
            'RandomForest', self.data.get_X_test(), self.data.get_type())
