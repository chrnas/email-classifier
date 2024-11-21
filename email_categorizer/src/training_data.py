import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class TrainingData():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        # This method will create the model for data
        # This will be performed in second activity
        self.y = df.y.to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, df.y, test_size=0.2, random_state=0)
