from .sentence_transformer import SentenceTransformerEmbeddings
from .tfidf import TfidfEmbeddings
from .word2vec import Word2VecEmbeddings
import pandas as pd

class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(embedding_types, df: pd.DataFrame):
        """
        Creates embedding instances based on specified embedding types.

        This method allows the user to create one or more types of embeddings 
        from a given DataFrame. The supported embedding types include 
        'tfidf', 'sentence_transformer', and 'word2vec'.

        Args:
            embedding_types (str or list): A string or list of strings specifying 
                                            the types of embeddings to create. 
                                            Valid options are 'tfidf', 
                                            'sentence_transformer', and 'word2vec'.
            df (pd.DataFrame): The DataFrame containing text data from which 
                               the embeddings will be generated.

        Returns:
            list: A list of instantiated embedding..

        Raises:
            ValueError: If an unknown embedding type is specified.
        """
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
                raise ValueError(f"Unknown embedding type: {embedding_type}. Choose between 'tfidf', 'word2vec' or 'sentence_transformer'.")

        return embeddings