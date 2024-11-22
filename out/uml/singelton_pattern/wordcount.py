from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from .base_embeddings import BaseEmbeddings
import pandas as pd


class WordcountEmbeddings(BaseEmbeddings):

    model: CountVectorizer

    def __init__(self):
        self.model = CountVectorizer()

    def create_training_embeddings(self, df: pd.DataFrame):
        """Create training embeddings by fitting the CountVectorizer model on the data and transforming it into feature vectors."""
        self.model.fit(df["Interaction content"].tolist() +
                       df["Ticket Summary"].tolist())

        x1 = self.model.transform(df["Interaction content"].tolist()).toarray()
        x2 = self.model.transform(df["Ticket Summary"].tolist()).toarray()

        X = np.concatenate((x1, x2), axis=1)

        return X

    def create_classification_embeddings(self, df: pd.DataFrame):
        """Create classification embeddings by transforming the data using the fitted CountVectorizer model."""
        x1 = self.model.transform(df["Interaction content"].tolist()).toarray()
        x2 = self.model.transform(df["Ticket Summary"].tolist()).toarray()

        X = np.concatenate((x1, x2), axis=1)

        return X
