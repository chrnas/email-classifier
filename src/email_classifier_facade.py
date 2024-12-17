import pandas as pd
from src.data_preparation.data_processor_decorator_factory import DataProcessorDecoratorFactory
from src.feature_engineering.base_embeddings import BaseEmbeddings
from src.modeling.context import ContextClassifier
from src.data_preparation.dataset_loader import DatasetLoader
from src.modeling.context import ContextClassifier
from src.data_preparation.data_processor import DataProcessor
from src.modeling.training_data import TrainingData
from src.modeling.model_factory import ModelFactory


class EmailClassifierFacade():
    df: pd.DataFrame
    emails: pd.DataFrame
    data_preprocessor: DataProcessor
    base_embeddings: BaseEmbeddings
    model_context: ContextClassifier
    data: TrainingData
    name: str
    df: pd.DataFrame

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
        self.emails: pd.DataFrame = None

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self) -> str:
        return f"\nName: {self.name}\nPreprocess Features: {self.data_preprocessor}\nEmbeddings: {self.base_embeddings}\nModel: {self.model_context.modelstrat}\n\n"

    def add_emails(self, path):
        data_set_loader = DatasetLoader()
        self.emails = data_set_loader.read_data(path)
        self.emails = data_set_loader.renameColumns(self.emails)

    def classify_emails(self):
        if self.emails is None:
            print("Add emails first to classify them.")
            return
        df = self.data_preprocessor.process(self.emails)
        X = self.base_embeddings.create_classification_embeddings(df)
        char_limit = 200
        predictions = self.model_context.predict_emails(X, df["Interaction content"].apply(
            lambda x: x[:char_limit] + ' ...' if len(x) > char_limit else x))
        return predictions

    def change_strategy(self, model_type: str):
        model = ModelFactory().create_model(model_type)
        self.model_context.choose_strat(model)

    def add_preprocessing(self, feature: str):
        self.data_preprocessor = DataProcessorDecoratorFactory().create_data_preprocessor(
            self.data_preprocessor, feature)

        print("Processor updated to:", self.data_preprocessor)

    def train_model(self, path: str):
        # Preprocess data
        data_set_loader = DatasetLoader()
        self.df = data_set_loader.read_data(path)
        self.df = data_set_loader.renameColumns(self.df)
        self.df = self.data_preprocessor.process(self.df)
        X = self.base_embeddings.create_training_embeddings(self.df)

        # modelling
        self.data = TrainingData(X, self.df)
        self.model_context.data = self.data
        self.model_context.train()
        self.model_context.predict()
        self.model_context.classification_report()

    def display_evaluation(self):
        self.model_context.classification_report()
