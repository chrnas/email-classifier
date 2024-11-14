from .sentence_transformer import SentenceTransformerEmbeddings
from .tfidf import TfidfEmbeddings
from .word2vec import Word2VecEmbeddings
import pandas as pd


class SimpleEmbeddingsFactory:
    @staticmethod
    def create_embeddings(embedding_type, df: pd.DataFrame):

        if embedding_type == "sentence_transformer":
            return SentenceTransformerEmbeddings(df)
        elif embedding_type == "tfidf":
            return TfidfEmbeddings(df)
        elif embedding_type == "tfidf-sentence_transformer":
            raise TfidfSentenceEmbeddings(df)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}. Choose between 'tfidf', 'word2vec' or 'sentence_transformer'.")
