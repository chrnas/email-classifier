import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from .base_embeddings import BaseEmbeddings


class SentenceTransformerEmbeddings(BaseEmbeddings):

    def __str__(self) -> str:
        return "sentence_transformer"

    def create_training_embeddings(self, df: pd.DataFrame):
        """Converts text data from "Interaction content" and "Ticket Summary" columns into SentenceTransformer embeddings, then concatenates them into a single feature matrix."""
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for both columns
        x1 = model.encode(df["Interaction content"].tolist())
        x2 = model.encode(df["Ticket Summary"].tolist())

        # Concatenate embeddings along the second axis
        X = np.array(np.concatenate((x1, x2), axis=1))

        return X

    def create_classification_embeddings(self, df: pd.DataFrame):
        """Converts text data from "Interaction content" and "Ticket Summary" columns into SentenceTransformer embeddings, then concatenates them into a single feature matrix."""
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for both columns
        x1 = model.encode(df["Interaction content"].tolist())
        x2 = model.encode(df["Ticket Summary"].tolist())

        # Concatenate embeddings along the second axis
        X = np.array(np.concatenate((x1, x2), axis=1))

        return X
