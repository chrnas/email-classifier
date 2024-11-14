from email_classifier_facade import EmailClassifierFacade
from data_preparation.data_processor_flex import DataProcessor
from data_preparation.data_preprocessor_factory import DataPreProcessorFactory
from feature_engineering.base_embeddings import BaseEmbeddings
from models.model_factory import ModelFactory
from feature_engineering.simple_embeddings_factory import SimpleEmbeddingsFactory
import pandas as pd
from training_data import TrainingData


class EmailClassifierFactory:

    @staticmethod
    def create_email_classifier(
        df: pd.DataFrame,
        embedding_type: list[str],
        pre_processing_features: list[str],
        model_type: str
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
            model_type, "model_name(notneeded!)", data.get_X_test(), data.get_type())
        email_classifier = EmailClassifierFacade(
            feature_engineer,
            data_processor,
            model
        )
        return email_classifier
