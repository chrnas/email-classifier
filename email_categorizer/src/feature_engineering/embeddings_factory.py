from .sentence_transformer import SentenceTransformerEmbeddings
from .tfidf import TfidfEmbeddings
from .word2vec import Word2VecEmbeddings
import pandas as pd

class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(embedding_types, df: pd.DataFrame):
        # Check if embedding_types is a string, if so, convert to a list
        if isinstance(embedding_types, str):
            embedding_types = [embedding_types]

        embeddings = []

        for embedding_type in embedding_types:
            if embedding_type == "sentence_transformer":
                embeddings.append(SentenceTransformerEmbeddings(df))
            elif embedding_type == "tfidf":
                embeddings.append(TfidfEmbeddings(df))
            elif embedding_type == "word2vec":
                embeddings.append(Word2VecEmbeddings(df))
            else:
                raise ValueError(f"Unknown embedding type: {embedding_type}. Choose between tfidf, word2vec or sentence_transformer.")

        return embeddings