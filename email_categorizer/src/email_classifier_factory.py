from context_classification.context import ContextClassifier
from data_preparation.dataset_loader import DatasetLoader
from email_classifier_facade import EmailClassifierFacade
from data_preparation.data_processor_flex import DataProcessor
from data_preparation.data_preprocessor_factory import DataPreProcessorFactory
from feature_engineering.base_embeddings import BaseEmbeddings
from models.model_factory import ModelFactory
from feature_engineering.simple_embeddings_factory import SimpleEmbeddingsFactory
import pandas as pd
from data_preparation.simple_data_preprocessor_decorator_factory import SimpleDataPreProcessorDecoratorFactory
from training_data import TrainingData


class EmailClassifierFactory:

    @staticmethod
    def create_email_classifier(
        df: pd.DataFrame,
        embedding_type: str,
        model_type: str,
        name: str
    ):
        # data_processor = DataPreProcessorFactory().create_data_preprocessor(
        #    df, pre_processing_features)
        data_processor = DataProcessor()
        df = data_processor.process(df)
        feature_engineer = SimpleEmbeddingsFactory().create_embeddings(
            embedding_type, df
        )
        feature_engineer.create_embeddings()
        X = feature_engineer.get_embeddings()
        data = TrainingData(X, df)
        model = ModelFactory().create_model(
            model_type, data.get_X_test(), data.get_type())
        strategy_context = ContextClassifier(model)
        strategy_context.train(data)
        strategy_context.predict(data)
        strategy_context.print_results(data)
        email_classifier = EmailClassifierFacade(
            feature_engineer,
            data_processor,
            strategy_context,
            data,
            name
        )
        return email_classifier
