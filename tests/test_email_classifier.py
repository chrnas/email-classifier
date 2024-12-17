import pytest
import pandas as pd
import csv
from src.email_classifier_factory import EmailClassifierFactory
from src.data_preparation.dataset_loader import DatasetLoader
from src.email_classifier_facade import EmailClassifierFacade


@pytest.fixture(scope="module")
def test_data_path():
    path = "./tests/test_data/AppGallery.csv"
    return path

@pytest.fixture(scope="module")
def test_email_path():
    path = "./tests/test_data/TestEmails.csv"
    return path


@pytest.fixture(scope="module")
def test_predictions_path():
    path = "./tests/test_data/predictions.csv"
    return path


@pytest.fixture(scope="module")
def email_classifier(test_data_path):
    df = DatasetLoader().read_data(test_data_path)
    df = DatasetLoader().renameColumns(df)
    email_classifier = EmailClassifierFactory().create_email_classifier(
        df=df,
        embedding_type='tfidf',
        model_type='svc',
        name='classifier1'
    )
    return email_classifier


def test_email_cassifier_creation(email_classifier):
    assert isinstance(email_classifier, EmailClassifierFacade)
    assert email_classifier.name == 'classifier1'
    assert email_classifier.base_embeddings.__str__() == 'tf-idf'
    assert email_classifier.data_preprocessor.__str__() == 'DataProcessor: features: base'
    assert email_classifier.model_context.modelstrat.__str__() == 'SVC'
    assert email_classifier.emails is None


def test_add_emails(email_classifier, test_email_path):
    email_classifier.add_emails(test_email_path)
    assert isinstance(email_classifier.emails, pd.DataFrame)


def test_classify_emails(email_classifier, test_email_path, test_predictions_path):
    expected_predictions = []
    with open(test_predictions_path, "r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            prediction, email_content = row  # Unpack the row into two variables
            expected_predictions.append((prediction, email_content))  # Append the tuple
    expected_predictions = expected_predictions[1:]
    actual_predictions = email_classifier.classify_emails()
    print(expected_predictions)
    print("\n")
    print(actual_predictions)
    assert expected_predictions == actual_predictions


def test_dispaly_evaluation(email_classifier):
    email_classifier.display_evaluation()
    assert isinstance(email_classifier, EmailClassifierFacade)


def test_change_strategy_bayes(email_classifier, test_data_path):
    email_classifier.change_strategy('bayes')
    email_classifier.train_model(test_data_path)
    assert email_classifier.model_context.modelstrat.__str__() == "Bayes"


def test_change_strategy_randomforest(email_classifier, test_data_path):
    email_classifier.change_strategy('randomforest')
    email_classifier.train_model(test_data_path)
    assert email_classifier.model_context.modelstrat.__str__() == "RandomForest"


def test_change_strategy_svc(email_classifier, test_data_path):
    email_classifier.change_strategy('svc')
    email_classifier.train_model(test_data_path)
    assert email_classifier.model_context.modelstrat.__str__() == 'SVC'


def test_add_preprocessing_noise_removal(email_classifier, test_data_path):
    email_classifier.add_preprocessing('noise_removal')
    email_classifier.train_model(test_data_path)
    assert email_classifier.data_preprocessor.__str__(
    ) == 'DataProcessor: features: base, noise_removal'


def test_add_preprocessing_unicode_conversion(email_classifier, test_data_path):
    email_classifier.add_preprocessing('unicode_conversion')
    email_classifier.train_model(test_data_path)
    assert email_classifier.data_preprocessor.__str__(
    ) == 'DataProcessor: features: base, noise_removal, unicode_conversion'


def test_add_preprocessing_translation(email_classifier):
    email_classifier.add_preprocessing('translation')
    assert email_classifier.data_preprocessor.__str__() == 'DataProcessor: features: base, noise_removal, unicode_conversion, translation'


def test_train_model(email_classifier, test_data_path):
    email_classifier.train_model(test_data_path)
    assert isinstance(email_classifier, EmailClassifierFacade)