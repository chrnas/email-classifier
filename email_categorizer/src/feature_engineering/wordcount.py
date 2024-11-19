from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from .base_embeddings import BaseEmbeddings

class WordcountEmbeddings(BaseEmbeddings):
    def create_embeddings(self, df):
        model = CountVectorizer()
        
        # Vokabular auf beiden Textspalten fitten
        model.fit(df["Interaction content"].tolist() + df["Ticket Summary"].tolist())

        # Embeddings für beide Spalten generieren
        x1 = model.transform(df["Interaction content"].tolist()).toarray()
        x2 = model.transform(df["Ticket Summary"].tolist()).toarray()
        
        # Embeddings entlang der zweiten Achse verketten
        X = np.concatenate((x1, x2), axis=1)

        return X