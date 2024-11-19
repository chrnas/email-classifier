from abc import ABC, abstractmethod
import pandas as pd


class BaseEmbeddings(ABC):

    @abstractmethod
    def create_embeddings(self):
        """Creates the embeddings for the columns "Interaction content" and "Ticket Summary" of the dataframe."""
        ...