from preprocessing import DataProcessor
from dataset_loader import DatasetLoader
from training_data import TrainingData
from randomforest import RandomForest
import pandas as pd


class EmailClassifier():

    def __init__(self) -> None:

        self.data_processor: DataProcessor = None
        self.data_set_loader: DatasetLoader = DatasetLoader()
        self.model: RandomForest = None
        self.df: pd.DataFrame = None
        self.data: TrainingData = None

    def getModel(self):
        return self.model

    def classify_email(self, email: str) -> str:
        # This method will classify the email using the model, no idea how this will be used
        classification = self.model.use_model(email)
        print("Email classified")
        return classification

    def train_model(self, path: str) -> None:
        self.df = self.data_set_loader.read_data(path)
        self.df = self.data_set_loader.renameColumns(self.df)
        self.data_processor = DataProcessor(self.df)
        self.data_processor.create_tfidf_embd()
        X = self.data_processor.get_tfidf_embd()
        self.data = TrainingData(X, self.df)
        self.model = RandomForest(
            'RandomForest', self.data.get_X_test(), self.data.get_type())
        self.model.train(self.data)
        self.model.predict(self.data)

    def printModelEvaluation(self):
        self.model.print_results(self.data)