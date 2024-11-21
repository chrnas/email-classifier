from context_classification.context import ContextClassifier
from data_preparation.dataset_loader import DatasetLoader
from email_classifier_facade import EmailClassifierFacade
from data_preparation.data_processor import DataProcessor
from feature_engineering.base_embeddings import BaseEmbeddings
from models.model_factory import ModelFactory
from feature_engineering.embeddings_factory import EmbeddingsFactory
import pandas as pd
from training_data import TrainingData
from observing.statistics_collector import StatCollector
from observing.result_displayer import ResultDisplayer


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
        feature_engineer = EmbeddingsFactory().create_embeddings(
            embedding_type, df
        )
        X = feature_engineer.create_training_embeddings(df)
        data = TrainingData(X, df)
        model = ModelFactory().create_model(
            model_type, data.X_test, data.y)
        strategy_context = ContextClassifier(data)
        stat_collector = StatCollector()
        strategy_context.subscribe(stat_collector)
        result_displayer = ResultDisplayer()
        strategy_context.subscribe(result_displayer)
        strategy_context.choose_strat(model)
        strategy_context.train()
        strategy_context.predict()
        strategy_context.classification_report()
        # strategy_context.print_results()
        email_classifier = EmailClassifierFacade(
            feature_engineer,
            data_processor,
            strategy_context,
            data,
            name
        )
        return email_classifier
