from .sentence_transformer import SentenceTransformerEmbeddings
from .tfidf import TfidfEmbeddings
from .tfidf_sentence import TfidfSentenceEmbeddings
from .base_embeddings import BaseEmbeddings


class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(embedding_type: str) -> BaseEmbeddings:
        """Create and return the appropriate embeddings class based on the specified embedding type."""
        if embedding_type == "sentence_transformer":
            return SentenceTransformerEmbeddings()
        elif embedding_type == "tfidf":
            return TfidfEmbeddings()
        elif embedding_type == "tfidf-sentence_transformer":
            return TfidfSentenceEmbeddings()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}. Choose between 'tfidf', 'word2vec' or 'sentence_transformer'.")
