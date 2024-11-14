import pandas as pd
from models.base import BaseModel
from models.randomforest import RandomForest
from training_data import TrainingData



class ContextClassifier():
    def __init__(self, data: TrainingData) -> None:
        self.modelstrat = None
        self.data =data

    def choose_strat(self,modelstrat: BaseModel):
        self.modelstrat = modelstrat
        
    def train(self) :
        self.modelstrat.train(self.data)
    
    def predict(self) :
        self.modelstrat.predict(self.data)
      
    def print_results(self):
        self.modelstrat.print_results(self.data)
    

  