import pandas as pd
from models.base import BaseModel


class ContextClassify:
    def __init__(self,modelstrat: BaseModel) -> None:
        self.modelstrat = modelstrat
    
    def choose_strat(self,modelstrat: BaseModel):
        self.modelstrat = modelstrat
        
    def train(self,data) :
        self.modelstrat.train(data)

    
    def predict(self,data) :
        self.modelstrat.predict(data)
      
    def print_results(self, data):
        self.modelstrat.print_results(data)
        

  
