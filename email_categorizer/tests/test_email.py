import pytest
from unittest.mock import MagicMock
from src.data_preparation.dataset_loader import DatasetLoader



def test_1():
    # Create an instance of EmailClassifierFacade using the mocked dependencies
    data_set_loader = DatasetLoader()
    #base_embeddings = EmbeddingsFactory('tfidf')
    assert 1 == 1
