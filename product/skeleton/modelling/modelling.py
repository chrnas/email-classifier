from model.randomforest import RandomForest
from sklearn.model_selection import train_test_split
import pandas as pd

def model_predict(data, df, name):
    # Here we need to call the methods related to the model e.g., random forest 
    rf = RandomForest(model_name="Random Forest", embeddings=data.get_X_train(), y=data.get_type())
    rf.train(data)
    x_test_series = pd.Series(data.get_X_test())
    rf.predict(x_test_series)
    rf.print_results(data)

def model_evaluate(model, data):
    model.print_results(data)