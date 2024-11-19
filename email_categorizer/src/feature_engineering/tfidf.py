import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embeddings import BaseEmbeddings


class TfidfEmbeddings(BaseEmbeddings):
        def create_embeddings(self, df):
            """Converts text data from "Interaction content" and "Ticket Summary" columns into TF-IDF feature vectors, then concatenates them into a single feature matrix."""
            tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
            x1 = tfidfconverter.fit_transform(df["Interaction content"]).toarray()
            x2 = tfidfconverter.fit_transform(df["Ticket Summary"].toarray())
            X = np.concatenate((x1, x2), axis=1)
            return X
