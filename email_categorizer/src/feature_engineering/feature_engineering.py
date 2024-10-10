import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineer():
        def __init__(self,
                 df: pd.DataFrame) -> None:
            self.df = df
            self.X = None

        def create_tfidf_embd(self):
            """Converts text data from "Interaction content" and "Ticket Summary" columns into TF-IDF feature vectors, then concatenates them into a single feature matrix."""
            df = self.df
            tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
            x1 = tfidfconverter.fit_transform(df["Interaction content"]).toarray()
            x2 = tfidfconverter.fit_transform(df["Ticket Summary"]).toarray()
            X = np.concatenate((x1, x2), axis=1)
            self.X = X

        def get_tfidf_embd(self):
            """Returns the combined matrix as a NumPy array."""
            return self.X