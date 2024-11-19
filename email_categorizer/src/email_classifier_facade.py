from data_preparation.simple_data_preprocessor_decorator_factory import SimpleDataPreProcessorDecoratorFactory
from feature_engineering.base_embeddings import BaseEmbeddings
from context_classification.context import ContextClassifier
from data_preparation.dataset_loader import DatasetLoader
import pandas as pd
from context_classification.context import ContextClassifier
from feature_engineering.sentence_transformer import SentenceTransformerEmbeddings
from data_preparation.data_preprocessor_factory import DataPreProcessorFactory
from data_preparation.data_processor_flex import DataProcessor, DataProcessorDecorator, DeDuplicationDecorator, NoiseRemovalDecorator, TranslatorDecorator, UnicodeConversionDecorator
from training_data import TrainingData
from models.randomforest import RandomForest
from models.model_factory import ModelFactory
from models.base import BaseModel
import random
from feature_engineering.simple_embeddings_factory import SimpleEmbeddingsFactory


class EmailClassifierFacade():
    df: pd.DataFrame
    emails: list[str]
    data_preprocessor: DataProcessor
    base_embeddings: BaseEmbeddings
    model_context: ContextClassifier
    data: TrainingData
    name: str
    df: pd.DataFrame
    emails: list[str]

    def __init__(self,
                 base_embeddings: BaseEmbeddings,
                 data_preprocessor: DataProcessor,
                 model_context: ContextClassifier,
                 training_data: TrainingData,
                 name):
        self.data_preprocessor = data_preprocessor
        self.base_embeddings = base_embeddings
        self.model_context = model_context
        self.data = training_data
        self.name = name
        self.df = None
        self.emails: list[str] = []

    def __eq__(self, other):
        # Define equality based on name and age
        return self.name == other.name

    def __str__(self) -> str:
        return f"Email classifier name: {self.name}"

    def add_emails(self, path):
        data_set_loader = DatasetLoader()
        self.emails = data_set_loader.read_data(path)
        self.emails = data_set_loader.renameColumns(self.df)
        print(self.data)

    def classify_emails(self):
        self.data_preprocessor.process(self.emails)
        self.base_embeddings.create_embeddings()
        self.model_context.classify(self.emails)

    def change_strategy(self, model_type: str):
        model = ModelFactory().create_model(
            model_type, self.data.get_X_test(), self.data.get_type())
        self.model_context.choose_strat(model)

    def add_preprocessing(self, feature: str):
        self.data_preprocessor = SimpleDataPreProcessorDecoratorFactory().create_data_preprocessor(
            self.data_preprocessor, feature)

        print("Processor:", self.data_preprocessor)

    def train_model(self, path: str):
        # load the data
        data_set_loader = DatasetLoader()
        self.df = data_set_loader.read_data(path)
        self.df = data_set_loader.renameColumns(self.df)
        self.df = self.data_preprocessor.process(self.df)
        # print(self.data_preprocessor)
        # self.df = self.data_preprocessor.process(df)  # This doesnt work currently for unknown reason

        # feature engineering
        self.feature_engineer = SimpleEmbeddingsFactory().create_embeddings(
            "sentence_transformer", self.df
        )
        self.feature_engineer.create_embeddings()
        X = self.feature_engineer.get_embeddings()

        # modelling
        self.data = TrainingData(X, self.df)
        #use model name here or something similar
        model = ModelFactory().create_model(
            "randomforest", self.data.get_X_test(), self.data.get_type())
        self.model_context.choose_strat(model)
        self.model_context.train(self.data)
        self.model_context.predict(self.data)
        self.model_context.print_results(self.data)

    def display_evaluation(self):
        self.model_context.print_results(self.data)
