import pandas as pd
from models.base import BaseModel
from models.randomforest import RandomForest


class ContextClassifier:
    def __init__(self) -> None:
        self.modelstrat =RandomForest(
            'RandomForest', self.data.get_X_test(), self.data.get_type())
    
    def choose_strat(self,modelstrat: BaseModel):
        self.modelstrat = modelstrat
        
    def train(self,data) :
        self.modelstrat.train(data)

    
    def predict(self,data) :
        self.modelstrat.predict(data)
      
    def print_results(self, data):
        self.modelstrat.print_results(data)
        

  
