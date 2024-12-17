import pytest
from src.client import Client
from types import SimpleNamespace


@pytest.fixture(scope="module")
def client():
    client = Client()
    args = {'command': 'create_email_classifier', 'path': 'AppGallery.csv',
            'embedding': 'tfidf', 'model': 'svc', 'name': 'classifier1'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    return client


def test_add_email_classifier_command(client):
    args = {'command': 'create_email_classifier', 'path': 'AppGallery.csv',
            'embedding': 'tfidf', 'model': 'svc', 'name': 'classifier2'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert len(client.email_classifiers) == 2


def test_list_email_classifiers_command(capsys, client):
    args = {'command': 'list_email_classifiers'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert capsys.readouterr().out is not None


def test_choose_email_classifier_command(client):
    args = {'command': 'choose_email_classifier', 'name': 'classifier1'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert client.email_classifiers[0].name == 'classifier1'


def test_remove_classifier_command(client):
    args = {'command': 'remove_email_classifier', 'name': 'classifier2'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert len(client.email_classifiers) == 1


def test_add_emails_command(client):
    args = {'command': 'add_emails', 'path': 'TestEmails.csv'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert client.email_classifiers[0].emails is not None


def test_classify_emails_command(capsys, client):
    args = {'command': 'classify_emails'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert capsys.readouterr().out is not None


def test_change_strategy_command(client):
    args = {'command': 'change_strategy', 'strategy': 'bayes'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert client.email_classifiers[0].model_context.modelstrat.__str__(
    ) == 'Bayes'


def test_add_preprocessing_command(client):
    args = {'command': 'add_preprocessing', 'feature': 'noise_removal'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert client.email_classifiers[0].data_preprocessor.__str__(
    ) == 'DataProcessor: features: base, noise_removal'


def test_train_model_command(capsys, client):
    args = {'command': 'add_preprocessing', 'feature': 'noise_removal'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    args = {'command': 'train_model', 'path': 'AppGallery.csv'}
    args = SimpleNamespace(**args)
    client.handle_input(args)
    assert capsys.readouterr().out is not None
