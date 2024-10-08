#Methods related to converting text in into numeric representation and then returning numeric representation may go here
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Text representation in Numeric Form
def get_tfidf_embd(df):
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidfconverter.fit_transform(df["Interaction content"]).toarray()
    x2 = tfidfconverter.fit_transform(df["Ticket Summary"]).toarray()
    X = np.concatenate((x1, x2), axis=1)
    return X