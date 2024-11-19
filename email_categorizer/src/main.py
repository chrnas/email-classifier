from context_classification.context import ContextClassifier
from data_preparation.data_processor import DataProcessor, TranslatorDecorator
from email_classifier import EmailClassifier
from feature_engineering.tfidf import TfidfEmbeddings
from data_preparation.dataset_loader import DatasetLoader
from feature_engineering.wordcount import WordcountEmbeddings
from models.randomforest import RandomForest
from training_data import TrainingData
from data_preparation.simple_data_preprocessor_decorator_factory import SimpleDataPreProcessorDecoratorFactory
from feature_engineering.sentence_transformer import SentenceTransformerEmbeddings

# Code will start executing from following line
if __name__ == '__main__':
    # train email classifier
    """
     email_classifier = EmailClassifier()
     email_classifier.train_model("../data/AppGallery.csv")
     # Print results
     email_classifier.printModelEvaluation()
     """
    # classify email
    # email_classifier.classify_email("email")
    data_set_loader = DatasetLoader()
    # load the data
    path = "../data/AppGallery.csv"
    df = data_set_loader.read_data(path)
    df = data_set_loader.renameColumns(df)

    feature = "deduplication"
    processor = DataProcessor()
    processor = SimpleDataPreProcessorDecoratorFactory().create_data_preprocessor(
        processor, feature)
    # preproccess the data
    df = processor.process(df)

    # feature engineering
    base_embeddings = SentenceTransformerEmbeddings()
    X = base_embeddings.create_embeddings(df)

    # modelling
    data = TrainingData(X, df)
    context = ContextClassifier(data)

    context.choose_strat(RandomForest(
        'RandomForest', data.get_X_test(), data.get_type()))
    context.train()
    # model.train(data)
    context.predict()
    # model.predict(data)
    context.print_results()
