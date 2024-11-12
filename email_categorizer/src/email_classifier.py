from data_preparation.data_processor import DataProcessor
from data_preparation.dataset_loader import DatasetLoader
from feature_engineering.base_embeddings import BaseEmbeddings
from feature_engineering.tfidf import TfidfEmbeddings
from feature_engineering.word2vec import Word2VecEmbeddings
from feature_engineering.sentence_transformer import SentenceTransformerEmbeddings
from training_data import TrainingData
from models.randomforest import RandomForest
<<<<<<< Updated upstream
=======
from models.bayes import Bayes
from models.svm import Svm
from feature_engineering.wordcount import Wordcount
>>>>>>> Stashed changes
import pandas as pd


class EmailClassifier():

    def __init__(self) -> None:
        self.data_set_loader: DatasetLoader = DatasetLoader()
        self.data_processor: DataProcessor = None
        self.base_embeddings: BaseEmbeddings = None
<<<<<<< Updated upstream
        self.model: RandomForest = None
=======
        self.context: ContextClassifier = None
>>>>>>> Stashed changes
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
        self.data_processor.de_duplication()
        self.data_processor.translate_to_en()
        self.data_processor.remove_noise()
        self.data_processor.convert_to_unicode()
        self.df = self.data_processor.get_df()

        # feature engineering
<<<<<<< Updated upstream
        BaseEmbeddings
        self.base_embeddings = SentenceTransformerEmbeddings(self.df)
=======
        self.base_embeddings = Wordcount(self.df)
>>>>>>> Stashed changes
        self.base_embeddings.create_embeddings()
        X = self.base_embeddings.get_embeddings()

        # modelling
        self.data = TrainingData(X, self.df)
<<<<<<< Updated upstream
        self.model = RandomForest(
            'RandomForest', self.data.get_X_test(), self.data.get_type())
        self.model.train(self.data)
        self.model.predict(self.data)
=======
        
        context = ContextClassifier(RandomForest(
            'RandomForest', self.data.get_X_test(), self.data.get_type()))
        
        context.train(self.data)

        context.choose_strat(Svm(
            'Svm', self.data.get_X_test(), self.data.get_type()))
        context.train(self.data)

        #self.model.train(self.data)
        context.predict(self.data)
        self.context =context
        #self.model.predict(self.data)
>>>>>>> Stashed changes

    def printModelEvaluation(self):
        self.context.print_results(self.data)
