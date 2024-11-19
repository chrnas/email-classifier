
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle

### Training the model

# Load and preprocess data
df = pd.read_csv("AppGallery_done.csv")
df['Interaction content'] = df['Interaction content'].values.astype('U')
df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

# Create TF-IDF vectorizers for each column
tfidf_interaction = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
tfidf_summary = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)

# Fit and transform the training data
x1 = tfidf_interaction.fit_transform(df["Interaction content"]).toarray()
x2 = tfidf_summary.fit_transform(df["Ticket Summary"]).toarray()

# Concatenate features from both columns
X = np.concatenate((x1, x2), axis=1)
y = df['y'].to_numpy()

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

# Test the classifier
y_pred = classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


### Predicting new data

# Load and preprocess new data
df_emails = pd.read_csv("email.csv")
df_emails['Interaction content'] = df_emails['Interaction content'].values.astype('U')
df_emails['Ticket Summary'] = df_emails['Ticket Summary'].values.astype('U')

# Transform new data using the saved vectorizers
x1_new = tfidf_interaction.transform(df_emails["Interaction content"]).toarray()
x2_new = tfidf_summary.transform(df_emails["Ticket Summary"]).toarray()

# Concatenate features
X_new = np.concatenate((x1_new, x2_new), axis=1)

# Predict using the trained classifier
y_pred_new = classifier.predict(X_new)
index = 0
for pred in y_pred_new:
    index += 1
    print(f"Prediction for email {index}: {pred}")
