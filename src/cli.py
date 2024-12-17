import argparse
import os
import pickle
from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.history import InMemoryHistory
from src.client import Client
from src.classifier_config_singleton import ClassifierConfigSingleton


class Cli:

    client: Client
    config_manager: ClassifierConfigSingleton

    def __init__(self, client: Client):
        self.client = client
        self.config_manager = ClassifierConfigSingleton()

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="CLI for estimating positions based on sound files.")
        subparsers = parser.add_subparsers(dest='command')

        subparsers.add_parser(
            'test', help='Execute a test command to verify CLI functionality.')

        add_email_parser = subparsers.add_parser(
            'add_emails', help='Classify email comand.')
        add_email_parser.add_argument('path', help='Path to the email files.')

        subparsers.add_parser(
            'classify_emails', help='Classify email comand.')

        create_email_classifier_parser = subparsers.add_parser(
            'create_email_classifier', help='Create an email classifier.')
        create_email_classifier_parser.add_argument(
            'path', help='Path to the email files.')
        create_email_classifier_parser.add_argument(
            'embedding', help='Embeddings type.')
        create_email_classifier_parser.add_argument(
            'model', help='Classification algorithm.')
        create_email_classifier_parser.add_argument(
            'name', help='Name to the email classifier.')

        choose_email_classifier = subparsers.add_parser(
            'choose_email_classifier', help='Choose email classifier.')
        choose_email_classifier.add_argument(
            'name', help='Name to the email classifier.')

        remove_email_classifier = subparsers.add_parser(
            'remove_email_classifier', help='Remove email classifier.')
        remove_email_classifier.add_argument(
            'name', help='Name to the email classifier to be removed.')

        change_strategy_parser = subparsers.add_parser(
            'change_strategy', help='Change strategy.')
        change_strategy_parser.add_argument(
            'strategy', help='Change strategy to bayes.')

        add_pre_processing_parser = subparsers.add_parser(
            'add_preprocessing', help='Add preprocessing.')
        add_pre_processing_parser.add_argument(
            'feature', help='Add translation feature.')

        train_model_parser = subparsers.add_parser(
            'train_model', help='Train model.')
        train_model_parser.add_argument(
            'path', help='Path for training data.')

        subparsers.add_parser(
            'display_evaluation', help='Display the evaluation of the current positioning method.')

        subparsers.add_parser(
            'list_email_classifiers', help='List email classifiers.')

        subparsers.add_parser(
            'undo', help='Undo the last command.')

        subparsers.add_parser('exit', help='Exit the CLI.')

        return parser

    def create_completer(self) -> NestedCompleter:

        features_dict = self.config_manager.preprocessing_features
        model_dict = self.config_manager.models
        embeddings_dict = self.config_manager.embeddings
        completer_dict = {
            'test': None,
            'add_emails': {paths: None for paths in os.listdir(self.config_manager.data_folder_path)},
            'classify_emails': None,
            'create_email_classifier': {paths: {embedding: {model: {"enter_a_classifier_name"} for model in model_dict} for embedding in embeddings_dict} for paths in os.listdir(self.config_manager.data_folder_path)},
            'list_email_classifiers': None,
            'choose_email_classifier': {email_classifier.name: None for email_classifier in self.client.email_classifiers},
            'remove_email_classifier': {email_classifier.name: None for email_classifier in self.client.email_classifiers},
            'change_strategy': {model: None for model in model_dict},
            'add_preprocessing': {feature: None for feature in features_dict},
            'train_model': {paths: None for paths in os.listdir(self.config_manager.data_folder_path)},
            'display_evaluation': None,
            'undo': None,
            'exit': None
        }
        return NestedCompleter.from_nested_dict(completer_dict)

    def run(self):
        parser = self.create_parser()
        history = InMemoryHistory()
        completer = self.create_completer()

        print("Startup complete")
        print("Welcome to Email Classifier application.")

        file_path = self.config_manager.state_folder_path + "/client_state.pkl"
        try:
            with open(file_path, 'rb') as file:
                self.client = pickle.load(file)
        except Exception as e:
            print(f"Error: {e}")
            print("Error loading client, continuing with a new client.")

        while True:
            completer = self.create_completer()
            try:
                input_str = prompt("> ", completer=completer, history=history)
                if input_str.strip().lower() == 'exit':
                    print("Exiting CLI.")
                    file_path = self.config_manager.state_folder_path + "/client_state.pkl"
                    pickle.dump(self.client, open(file_path, "wb"))
                    break
                # update completer to update automatic completion
                completer = self.create_completer()
                args = parser.parse_args(input_str.split())
                self.client.handle_input(args)
            except SystemExit:
                # argparse throws a SystemExit exception if parsing fails, we'll catch it to keep the loop running
                print("Exiting CLI.")
                file_path = self.config_manager.state_folder_path + "/client_state.pkl"
                pickle.dump(self.client, open(file_path, "wb"))
                continue
            except KeyboardInterrupt:
                print("Exiting CLI.")
                file_path = self.config_manager.state_folder_path + "/client_state.pkl"
                pickle.dump(self.client, open(file_path, "wb"))
                break
            except Exception as e:  # catch errror and display it
                print(f"Error: {e}")
                file_path = self.config_manager.state_folder_path + "/client_state.pkl"
                pickle.dump(self.client, open(file_path, "wb"))
