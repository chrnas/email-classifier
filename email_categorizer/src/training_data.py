import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TrainingData():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        self.y = df.y.to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, df.y, test_size=0.2, random_state=0)
