from data_preparation.data_processor import *
from data_preparation.dataset_loader import DatasetLoader
from feature_engineering.base_embeddings import BaseEmbeddings
from feature_engineering.tfidf import TfidfEmbeddings
from feature_engineering.word2vec import Word2VecEmbeddings
from feature_engineering.sentence_transformer import SentenceTransformerEmbeddings
from feature_engineering.wordcount import WordcountEmbeddings
from training_data import TrainingData
from models.randomforest import RandomForest
from models.bayes import Bayes
from context_classification.context import ContextClassifier
import pandas as pd


class EmailClassifier():

    def __init__(self) -> None:
        self.data_set_loader: DatasetLoader = DatasetLoader()
        self.data_processor: DataProcessor = None
        self.base_embeddings: BaseEmbeddings = None

        self.context: ContextClassifier = None

        self.df: pd.DataFrame = None
        self.data: TrainingData = None

    def getModel(self):
        return self.model

    def classify_email(self, email: str) -> str:
        # This method will classify the email using the model, no idea how this will be used
        classification = self.model.use_model(email)
        print("Email classified")
        return classification
    
    def transform_text (self, config,csv_file):
        embedding_t = config["embedding_types"]
        preprocessing_t = config["preprocessing_steps"]
        df_input = pd.read_csv(csv_file)

    def train_model(self, path: str) -> None:
        # load the data
        self.df = self.data_set_loader.read_data(path)
        self.df = self.data_set_loader.renameColumns(self.df)

        # preproccess the data
        processor = DataProcessor(self.df)
        processor = DeDuplicationDecorator(processor)
        # processor = TranslatorDecorator(processor)
        processor = NoiseRemovalDecorator(processor)
        processor = UnicodeConversionDecorator(processor)
        self.df = processor.process()

        # feature engineering
        self.base_embeddings = SentenceTransformerEmbeddings()
        X = self.base_embeddings.create_embeddings(self.df)

        # modelling
        self.data = TrainingData(X, self.df)
        context = ContextClassifier(self.data)
        context.choose_strat(RandomForest(
            'RandomForest', self.data.X_test, self.data.y))
        context.train()
        #self.model.train(self.data)
        context.predict(self.data)
        #self.model.predict(self.data)
        self.context = context



        input_path= "/Users/patrickvorreiter/Documents/Studium/2024 Wintersemester/Systems Analysis and Design/email-categorizer/email_categorizer/data/TestEmails.csv"
        df_input = self.data_set_loader.read_data(input_path)
        df_input = self.data_set_loader.renameColumns(df_input)
        # preproccess the data
        processor = DataProcessor(df_input)
        processor = DeDuplicationDecorator(processor)
        # processor = TranslatorDecorator(processor)
        processor = NoiseRemovalDecorator(processor)
        processor = UnicodeConversionDecorator(processor)
        df_input = processor.process()
        X_input = self.base_embeddings.create_embeddings(df_input)
        print("_____BEFORE________")
        self.context.predict_emails(X_input)
        print("_____AFTER________")
        

    def printModelEvaluation(self):
        self.context.print_results(self.data)
