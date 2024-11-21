from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from .base_embeddings import BaseEmbeddings
import pandas as pd


class WordcountEmbeddings(BaseEmbeddings):

    model: CountVectorizer

    def __init__(self):
        self.model = CountVectorizer()

    def create_training_embeddings(self, df: pd.DataFrame):
        # Vokabular auf beiden Textspalten fitten
        self.model.fit(df["Interaction content"].tolist() +
                       df["Ticket Summary"].tolist())

        # Embeddings für beide Spalten generieren
        x1 = self.model.transform(df["Interaction content"].tolist()).toarray()
        x2 = self.model.transform(df["Ticket Summary"].tolist()).toarray()

        # Embeddings entlang der zweiten Achse verketten
        X = np.concatenate((x1, x2), axis=1)

        return X

    def create_classification_embeddings(self, df: pd.DataFrame):

        # Embeddings für beide Spalten generieren
        x1 = self.model.transform(df["Interaction content"].tolist()).toarray()
        x2 = self.model.transform(df["Ticket Summary"].tolist()).toarray()

        # Embeddings entlang der zweiten Achse verketten
        X = np.concatenate((x1, x2), axis=1)

        return X
