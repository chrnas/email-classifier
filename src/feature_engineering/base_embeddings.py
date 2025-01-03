import pandas as pd
from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):

    @abstractmethod
    def create_training_embeddings(self, df: pd.DataFrame):
        """Creates the embeddings for the columns "Interaction content" and "Ticket Summary" of the dataframe."""
        ...

    @abstractmethod
    def create_classification_embeddings(self, df: pd.DataFrame):
        """Creates the embeddings for the columns "Interaction content" and "Ticket Summary" of the dataframe."""
        ...
