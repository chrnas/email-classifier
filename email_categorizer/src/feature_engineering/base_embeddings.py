from abc import ABC, abstractmethod
import pandas as pd


class BaseEmbeddings(ABC):
    def __init__(self,
             df: pd.DataFrame) -> None:
        self.df = df
        self.X = None

    @abstractmethod
    def create_embeddings(self) -> None:
        """Creates the embeddings for the columns "Interaction content" and "Ticket Summary" of the dataframe."""
        ...

    def get_embeddings(self):
        """Returns the combined matrix as a NumPy array."""
        return self.X