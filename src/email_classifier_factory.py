from src.modeling.context import ContextClassifier
from src.email_classifier_facade import EmailClassifierFacade
from src.data_preparation.data_processor import DataProcessor
from src.modeling.model_factory import ModelFactory
from src.feature_engineering.embeddings_factory import EmbeddingsFactory
import pandas as pd
from src.modeling.training_data import TrainingData
from src.observation.statistics_collector import StatCollector
from src.observation.result_displayer import ResultDisplayer


class EmailClassifierFactory:

    @staticmethod
    def create_email_classifier(
        df: pd.DataFrame,
        embedding_type: str,
        model_type: str,
        name: str
    ):
        data_processor = DataProcessor()
        df = data_processor.process(df)
        feature_engineer = EmbeddingsFactory().create_embeddings(
            embedding_type
        )
        X = feature_engineer.create_training_embeddings(df)
        data = TrainingData(X, df)
        model = ModelFactory().create_model(model_type)
        strategy_context = ContextClassifier(data)
        stat_collector = StatCollector()
        strategy_context.subscribe(stat_collector)
        result_displayer = ResultDisplayer()
        strategy_context.subscribe(result_displayer)
        strategy_context.choose_strat(model)
        strategy_context.train()
        strategy_context.predict()
        strategy_context.classification_report()
        email_classifier = EmailClassifierFacade(
            feature_engineer,
            data_processor,
            strategy_context,
            data,
            name
        )
        return email_classifier
