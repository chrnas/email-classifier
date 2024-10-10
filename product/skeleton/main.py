#This is a main file: The controller. All methods will directly on directly be called here
import numpy as np
import pandas as pd
from preprocess import get_input_data, de_duplication, noise_remover, translate_to_english
from Config import Config
from embeddings import get_tfidf_embd
from modelling.modelling import model_predict
from modelling.data_model import Data
import random
from data_processing.dataset_loader import DatasetLoader
from data_processing.preprocessing import DataProcessor

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    df = get_input_data()
    return df


def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_english(df[Config.TICKET_SUMMARY].to_list())
    #df = translate_to_en(df)
    # remove noise in input data
    #df = noise_remover(df)
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)


# Code will start executing from following line
if __name__ == '__main__':
    # load the data
    dataset_loader = DatasetLoader()
    df = dataset_loader.get_input_data("./data/AppGallery.csv")
    
    # preproccess the data
    dataset_processor = DataProcessor(df)
    dataset_processor.de_duplication()
    #dataset_processor.translate_to_en()
    dataset_processor.noise_remover()
    dataset_processor.convert_to_unicode()

    # feature engineering
    dataset_processor.create_tfidf_embd()

    X = dataset_processor.get_tfidf_embd()
    df = dataset_processor.get_df()

    # data modelling
    data = get_data_object(X, df)

    # modelling
    perform_modelling(data, df, 'name')
    
