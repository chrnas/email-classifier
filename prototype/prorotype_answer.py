from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

df = pd.read_csv("AppGallery_done.csv")
# convert the dtype object to unicode string
df['Interaction content'] = df['Interaction content'].values.astype('U')
df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')
# Optional: rename variable names for remebering easily
df["y1"] = df["Type 1"]
df["y2"] = df["Type 2"]
df["y3"] = df["Type 3"]
df["y4"] = df["Type 4"]
df["x"] = df['Interaction content']
df["y"] = df["y2"]
# remove empty y
df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
temp = df
# Text representation in Numeric Form
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
x1 = tfidfconverter.fit_transform(temp["Interaction content"]).toarray()
x2 = tfidfconverter.fit_transform(temp["Ticket Summary"]).toarray()
X = np.concatenate((x1, x2), axis=1)
# Data preparation
y = temp.y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Model Selection
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
# Training
classifier.fit(X_train, y_train)
# Testing
y_pred = classifier.predict(X_test)
# Result Display
p_result = pd.DataFrame(classifier.predict_proba(X_test))
p_result.columns = classifier.classes_
print(p_result)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
