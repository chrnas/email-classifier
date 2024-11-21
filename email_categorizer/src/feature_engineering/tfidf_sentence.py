import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embeddings import BaseEmbeddings


class TfidfSentenceEmbeddings(BaseEmbeddings):

    def __init__(self):
        self.vectorizer_interaction = TfidfVectorizer(
            max_features=2000, min_df=4, max_df=0.90)
        self.vectorizer_summary = TfidfVectorizer(
            max_features=2000, min_df=4, max_df=0.90)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def __str__(self) -> str:
        return "tf-idf-sentence_transformer"

    def create_training_embeddings(self, df):
        """Combines TF-IDF and SentenceTransformer embeddings for both columns."""

        # Generate SentenceTransformer embeddings
        st_emb_interaction = self.model.encode(
            df["Interaction content"].tolist())
        st_emb_summary = self.model.encode(df["Ticket Summary"].tolist())

        # Generate TF-IDF embeddings
        tfidf_emb_interaction = self.vectorizer_interaction.fit_transform(
            df["Interaction content"]).toarray()
        tfidf_emb_summary = self.vectorizer_summary.fit_transform(
            df["Ticket Summary"]).toarray()

        # Concatenate TF-IDF and SentenceTransformer embeddings for each column
        interaction_embeddings = np.concatenate(
            (tfidf_emb_interaction, st_emb_interaction), axis=1)
        summary_embeddings = np.concatenate(
            (tfidf_emb_summary, st_emb_summary), axis=1)

        # Combine the columns into one feature matrix
        X = np.concatenate(
            (interaction_embeddings, summary_embeddings), axis=1)
        return X

    def create_classification_embeddings(self, df):
        """Combines TF-IDF and SentenceTransformer embeddings for both columns."""

        # Generate sentenceTransformer embeddings
        st_emb_interaction = self.model.encode(
            df["Interaction content"].tolist())
        st_emb_summary = self.model.encode(df["Ticket Summary"].tolist())

        # Generate tfidf embeddings
        tfidf_emb_interaction = self.vectorizer_interaction.transform(
            df["Interaction content"]).toarray()
        tfidf_emb_summary = self.vectorizer_summary.transform(
            df["Ticket Summary"]).toarray()

        interaction_embeddings = np.concatenate(
            (tfidf_emb_interaction, st_emb_interaction), axis=1)
        summary_embeddings = np.concatenate(
            (tfidf_emb_summary, st_emb_summary), axis=1)

        X = np.concatenate(
            (interaction_embeddings, summary_embeddings), axis=1)
        return X
