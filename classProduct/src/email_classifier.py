from DataPreparation.preprocessing import DataProcessor
from DataPreparation.dataset_loader import DatasetLoader
from FeatureEngineering.feature_engineering import FeatureEngineer
from training_data import TrainingData
from Models.randomforest import RandomForest
import pandas as pd


class EmailClassifier():

    def __init__(self) -> None:
        self.data_set_loader: DatasetLoader = DatasetLoader()
        self.data_processor: DataProcessor = None
        self.feature_engineer: FeatureEngineer = None
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
        # load the data
        self.df = self.data_set_loader.read_data(path)
        self.df = self.data_set_loader.renameColumns(self.df)

        # preproccess the data
        self.data_processor = DataProcessor(self.df)
        self.dataset_processor.de_duplication()
        self.dataset_processor.translate_to_en()
        self.dataset_processor.noise_remover()
        self.dataset_processor.convert_to_unicode()
        self.df = self.dataset_processor.get_df()

        # feature engineering
        self.feature_engineer = FeatureEngineer(self.df)
        self.feature_engineer.create_tfidf_embd()
        X = self.feature_engineer.get_tfidf_embd()

        # modelling
        self.data = TrainingData(X, self.df)
        self.model = RandomForest(
            'RandomForest', self.data.get_X_test(), self.data.get_type())
        self.model.train(self.data)
        self.model.predict(self.data)

    def printModelEvaluation(self):
        self.model.print_results(self.data)
