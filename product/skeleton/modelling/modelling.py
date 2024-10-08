from model.randomforest import RandomForest
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

def model_predict(data, df, name):
    # Here we need to call the methods related to the model e.g., random forest 
    classifier = RandomForest(model_name="Random Forest", embeddings=data.get_X_train(), y=data.get_type())
    #rf.train(data)
    classifier.train(data)
    # Testing
    classifier.predict(data)
    # Result Display
    classifier.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)