from context_classification.context import ContextClassifier
from data_preparation.data_processor_flex import DataProcessor, TranslatorDecorator
from email_classifier import EmailClassifier
from feature_engineering.tfidf import TfidfEmbeddings
from data_preparation.dataset_loader import DatasetLoader
from feature_engineering.wordcount import WordcountEmbeddings
from models.randomforest import RandomForest
from training_data import TrainingData
from data_preparation.simple_data_preprocessor_decorator_factory import SimpleDataPreProcessorDecoratorFactory

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

    feature = "translation"
    processor = DataProcessor()
    processor = SimpleDataPreProcessorDecoratorFactory().create_data_preprocessor(
        processor, feature)
    # preproccess the data
    df = processor.process(df)

    # feature engineering
    base_embeddings = TfidfEmbeddings(df)
    base_embeddings.create_embeddings()
    X = base_embeddings.get_embeddings()

    # modelling
    data = TrainingData(X, df)
    context = ContextClassifier(RandomForest(
        'RandomForest', data.get_X_test(), data.get_type()))

    context.train(data)

    context.choose_strat(RandomForest(
        'RandomForest', data.get_X_test(), data.get_type()))
    context.train(data)

    # model.train(data)
    context.predict(data)
    # model.predict(data)
    context.print_results(data)
