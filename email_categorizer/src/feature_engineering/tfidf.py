import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embeddings import BaseEmbeddings


class TfidfEmbeddings(BaseEmbeddings):

    vectorizer: TfidfVectorizer

    def __init__(self):
        self.vectorizer_interaction = TfidfVectorizer(
            max_features=2000, min_df=4, max_df=0.90)
        self.vectorizer_summary = TfidfVectorizer(
            max_features=2000, min_df=4, max_df=0.90)

    def __str__(self) -> str:
        return "tf-idf"

    def create_training_embeddings(self, df):
        """Converts text data from "Interaction content" and "Ticket Summary" columns into TF-IDF feature vectors, then concatenates them into a single feature matrix."""
        x1 = self.vectorizer_interaction.fit_transform(df["Interaction content"]).toarray()
        x2 = self.vectorizer_summary.fit_transform(df["Ticket Summary"]).toarray()
        X = np.concatenate((x1, x2), axis=1)
        return X

    def create_classification_embeddings(self, df):
        """Converts text data from "Interaction content" and "Ticket Summary" columns into TF-IDF feature vectors, then concatenates them into a single feature matrix."""
        x1 = self.vectorizer_interaction.transform(df["Interaction content"]).toarray()
        x2 = self.vectorizer_summary.transform(df["Ticket Summary"]).toarray()
        X = np.concatenate((x1, x2), axis=1)
        return X
