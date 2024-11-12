from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from .base_embeddings import BaseEmbeddings

class Wordcount(BaseEmbeddings):
    def create_embeddings(self):
        df = self.df
        model = CountVectorizer()
        
        
        # Generate embeddings for both columns
        x1 = model.fit_transform(df["Interaction content"].tolist())
        x2 = model.fit_transform(df["Ticket Summary"].tolist())
        
        # Concatenate embeddings along the second axis
        self.X = np.array(np.concatenate((x1, x2), axis=1))
       
