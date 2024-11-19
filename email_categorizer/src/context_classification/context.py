import pandas as pd
from models.base import BaseModel
from models.randomforest import RandomForest
from training_data import TrainingData



class ContextClassifier():
    def __init__(self, data: TrainingData) -> None:
        self.modelstrat = None
        self.data =data

    def choose_strat(self, modelstrat: BaseModel):
        self.modelstrat = modelstrat
        
    def train(self):
        self.modelstrat.train(self.data)
    
<<<<<<< HEAD
    def predict(self):
        self.modelstrat.predict(self.data)
=======
    def predict(self, data) :
        self.modelstrat.predict(data)
>>>>>>> 2fa899b6374f26ebec90937f449cd6d90a2cf7ae
      
    def print_results(self,data):
        self.modelstrat.print_results(data)
    
    def predict_emails(self, emails_to_predict):
        self.modelstrat.predict_emails(emails_to_predict)  
  