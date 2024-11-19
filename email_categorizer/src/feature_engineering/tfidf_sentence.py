import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embeddings import BaseEmbeddings


class TfidfSentenceEmbeddings(BaseEmbeddings):
    def create_embeddings(self):
        """Converts text data from "Interaction content" and "Ticket Summary" columns into TF-IDF feature vectors, then concatenates them into a single feature matrix."""
        ...
