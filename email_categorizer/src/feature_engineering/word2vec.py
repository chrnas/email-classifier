from gensim.models import Word2Vec
import numpy as np
from .base_embeddings import BaseEmbeddings
import pandas as pd


class Word2VecEmbeddings(BaseEmbeddings):
    def tokenize_sentence(self, sentence):
        """Simple tokenizer: split sentence into words. You can enhance this for more sophisticated tokenization."""
        return sentence.split()

    def create_classification_embeddings(self, df: pd.DataFrame):
        """Converts text data from 'Interaction content' and 'Ticket Summary' columns into Word2Vec embeddings, then concatenates them into a single feature matrix."""

        # Tokenize sentences into words
        sentences = df["Interaction content"].apply(self.tokenize_sentence).tolist() + \
            df["Ticket Summary"].apply(self.tokenize_sentence).tolist()

        # Train Word2Vec model
        word2vec_model = Word2Vec(
            sentences, vector_size=100, window=5, min_count=1)

        # Create sentence embeddings by averaging word vectors
        def get_sentence_embedding(sentence, model):
            words = self.tokenize_sentence(sentence)
            return np.mean([model.wv[word] for word in words if word in model.wv], axis=0)

        x1 = np.array([get_sentence_embedding(sentence, word2vec_model)
                      for sentence in df["Interaction content"]])
        x2 = np.array([get_sentence_embedding(sentence, word2vec_model)
                      for sentence in df["Ticket Summary"]])

        # Concatenate sentence embeddings along the second axis
        X = np.array(np.concatenate((x1, x2), axis=1))

        return X

    def create_training_embeddings(self, df: pd.DataFrame):
        """Converts text data from 'Interaction content' and 'Ticket Summary' columns into Word2Vec embeddings, then concatenates them into a single feature matrix."""

        # Tokenize sentences into words
        sentences = df["Interaction content"].apply(self.tokenize_sentence).tolist() + \
            df["Ticket Summary"].apply(self.tokenize_sentence).tolist()

        # Train Word2Vec model
        word2vec_model = Word2Vec(
            sentences, vector_size=100, window=5, min_count=1)

        # Create sentence embeddings by averaging word vectors
        def get_sentence_embedding(sentence, model):
            words = self.tokenize_sentence(sentence)
            return np.mean([model.wv[word] for word in words if word in model.wv], axis=0)

        x1 = np.array([get_sentence_embedding(sentence, word2vec_model)
                      for sentence in df["Interaction content"]])
        x2 = np.array([get_sentence_embedding(sentence, word2vec_model)
                      for sentence in df["Ticket Summary"]])

        # Concatenate sentence embeddings along the second axis
        X = np.array(np.concatenate((x1, x2), axis=1))

        return X