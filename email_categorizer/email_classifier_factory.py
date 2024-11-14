from email_classifier_facade import EmailClassifierFacade
from data_preparation.data_processor_flex import DataProcessor
from data_preparation.data_preprocessor_factory import DataPreProcessorFactory
from feature_engineering.base_embeddings import BaseEmbeddings
from models.classification_factory import ClassificationFactory
from feature_engineering.simple_embeddings_factory import SimpleEmbeddingsFactory
import pandas as pd


class EmailClassifierFactory:

    @staticmethod
    def create_email_classifier(
        df: pd.DataFrame,
        embeddings: list[str],
        pre_processing_features: list[str],
        classification_algorithm: str
    ):
        feature_engineer = SimpleEmbeddingsFactory().create_embeddings(
            embeddings, df
        )
        # data_processor = DataPreProcessorFactory().create_data_preprocessor(
        #    df, pre_processing_features)
        data_processor = DataProcessor()
        classification_strategy = ClassificationFactory().create_classification_algorithm(
            classification_algorithm)
        email_classifier = EmailClassifierFacade(
            feature_engineer,
            data_processor,
            classification_strategy
        )
        return email_classifier
