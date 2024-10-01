#Methods related to data loading and all pre-processing steps will go here

import numpy as np
import pandas as pd

def get_input_data():
    df = pd.read_csv("/Users/patrickvorreiter/Documents/Studium/2024 Wintersemester/Systems Analysis and Design/AppGallery.csv")

    # convert the dtype object to unicode string
    df['Interaction content'] = df['Interaction content'].values.astype('U')
    df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

    #Optional: rename variable names for remebering easily
    df["y1"] = df["Type 1"]
    df["y2"] = df["Type 2"]
    df["y3"] = df["Type 3"]
    df["y4"] = df["Type 4"]
    df["x"] = df['Interaction content']

    df["y"] = df["y2"]
    # use all the types and not just type 2
    # df["y"] = df[["y1", "y2", "y3", "y4"]].fillna('').apply(lambda row: '|'.join(filter(None, row)), axis=1)

    return df

# remove empty y
def de_duplication(df):
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return df




# Data preparation
from sklearn.model_selection import train_test_split
y = temp.y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
