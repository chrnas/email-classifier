from feature_engineering.base_embeddings import BaseEmbeddings
from context_classification.context import ContextClassifier
from data_preparation.dataset_loader import DatasetLoader
import pandas as pd
from context_classification.context import ContextClassifier
from feature_engineering.sentence_transformer import SentenceTransformerEmbeddings
from data_preparation.data_preprocessor_factory import DataPreProcessorFactory
from data_preparation.data_processor_flex import DataProcessor, DataProcessorDecorator
from training_data import TrainingData
from models.randomforest import RandomForest
from models.model_factory import ModelFactory
from models.base import BaseModel
import random


class EmailClassifierFacade():
    df: pd.DataFrame
    emails: list[str]
    data_preprocessor: DataProcessor
    base_embeddings: BaseEmbeddings
    model_context: ContextClassifier
    data_set_loader: DatasetLoader
    name: str

    def __init__(self,
                 base_embeddings: BaseEmbeddings,
                 data_preprocessor: DataProcessor,
                 model_context: ContextClassifier):
        self.df = None
        self.emails: list[str] = []
        self.data_preprocessor = data_preprocessor,
        self.base_embeddings = base_embeddings,
        self.model_context = model_context
        self.data_set_loader = DatasetLoader()
        self.name = str(random.randint(0, 1000000))

    def __eq__(self, other):
        # Define equality based on name and age
        return self.name == other.name

    def __str__(self) -> str:
        return f"Email classifier name: {self.name}"

    def add_emails(self, path):
        self.emails = self.data_set_loader.read_data(path)
        self.emails = self.data_set_loader.renameColumns(self.df)

    def classify_emails(self):
        self.data_preprocessor.process(self.emails)
        self.base_embeddings.create_embeddings()
        self.model_context.classify(self.emails)

    def change_strategy(self, model: BaseModel):
        self.model_context.choose_strat(model)

    def add_preprocessing(self, data_processor_decorator: DataProcessorDecorator):
        self.data_preprocessor = data_processor_decorator

    def train_model(self, path):
        self.df = self.data_set_loader.read_data(path)
        self.df = self.data_set_loader.renameColumns(self.df)
        # load the data
        print(self.data_preprocessor)
        self.df = self.data_preprocessor.process(self.df)
        # preproccess the data
        # processor = DataProcessor(self.df)
        # processor = DeDuplicationDecorator(processor)
        # processor = NoiseRemovalDecorator(processor)
        # processor = UnicodeConversionDecorator(processor)
        # self.df = processor.process()

        # feature engineering
        self.base_embeddings = SentenceTransformerEmbeddings(self.df)
        self.base_embeddings.create_embeddings()
        X = self.base_embeddings.get_embeddings()

        # modelling
        self.data = TrainingData(X, self.df)
        model = ModelFactory().create_model(
            'random_forest', self.data.get_X_test(), self.data.get_type())
        self.model_context.choose_strat(model)
        self.model_context.train(self.data)
        self.model_context.predict(self.data)
        self.model_context.print_results(self.data)

    def displayEvaluation(self):
        self.model.print_results(self.data)
