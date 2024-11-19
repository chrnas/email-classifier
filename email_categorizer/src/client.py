import argparse
import os
from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.history import InMemoryHistory
from email_classifier_facade import EmailClassifierFacade
from data_preparation.dataset_loader import DatasetLoader
from command import ChangeStrategyCommand, ChooseEmailClassifierCommand, \
    AddEmailsCommand, CommandInvoker, \
    CreateEmailClassifierCommand, DisplayEvaluationCommand, \
    ListEmailClassifiersCommand, AddPreprocessingCommand, \
    TrainModelCommand, ClassifyEmailCommand
from email_classifier_factory import EmailClassifierFactory


class Client:

    email_classifiers: list[EmailClassifierFacade]
    data_set_loader: DatasetLoader
    command_invoker: CommandInvoker

    def __init__(self):
        self.data_set_loader = DatasetLoader()
        self.command_invoker = CommandInvoker()
        self.email_classifiers: list[EmailClassifierFacade] = []

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="CLI for estimating positions based on sound files.")
        subparsers = parser.add_subparsers(dest='command')

        # Test command
        subparsers.add_parser(
            'test', help='Execute a test command to verify CLI functionality.')

        # PosList command
        add_email_parser = subparsers.add_parser(
            'add_emails', help='Classify email comand.')
        add_email_parser.add_argument('path', help='Path to the email files.')
        classify_email_parser = subparsers.add_parser(
            'classify_emails', help='Classify email comand.')
        # Create email classifier command
        create_email_classifier_parser = subparsers.add_parser(
            'create_email_classifier', help='Create an email classifier.')
        create_email_classifier_parser.add_argument(
            'path', help='Path to the email files.')
        create_email_classifier_parser.add_argument(
            'embedding', help='Embeddings type.')
        create_email_classifier_parser.add_argument(
            'model', help='Classification algorithm.')
        create_email_classifier_parser.add_argument('name', help='Name to the email classifier.')

        # Choose email classifier command
        choose_email_classifier = subparsers.add_parser(
            'choose_email_classifier', help='Choose email classifier.')
        choose_email_classifier.add_argument(
            'name', help='Name to the email classifier.')

        # Change Strategy command
        change_strategy_parser = subparsers.add_parser(
            'change_strategy', help='Change strategy.')
        change_strategy_parser.add_argument(
            'strategy', help='Change strategy to bayes.')
        # Add preprocessing command
        add_pre_processing_parser = subparsers.add_parser(
            'add_preprocessing', help='Add preprocessing.')
        add_pre_processing_parser.add_argument(
            'feature', help='Add translation feature.')
        # Add preprocessing command
        train_model = subparsers.add_parser(
            'train_model', help='Train model.')
        train_model.add_argument(
            'path', help='Path for training data.')

        # Display Evaluation command
        display_evaluation_parser = subparsers.add_parser(
            'display_evaluation', help='Display the evaluation of the current positioning method.')
        # List email calssifiers command
        list_email_classifiers_parser = subparsers.add_parser(
            'list_email_classifiers', help='List email classifiers.')
        # Exit command
        undo_parser = subparsers.add_parser('undo', help='Undo the last command.')
        
        exit_parser = subparsers.add_parser('exit', help='Exit the CLI.')

        return parser

    def create_completer(self) -> NestedCompleter:

        # Extract method names and their possible settings
        features_dict = ['deduplication', 'unicode_conversion', 'noise_removal', 'translation']
        model_dict = ['bayes', 'randomforest', 'svc']
        embeddings_dict = ['tfidf', 'wordcount', 'sentence_transformer']
        completer_dict = {
            'create_email_classifier AppGallery.csv sentence_transformer randomforest emailclassifiertest': None,
            'test': {emailclassifier.name: None for emailclassifier in self.email_classifiers},
            'add_emails': {paths: None for paths in os.listdir("../data")},
            'classify_emails': None,
            'create_email_classifier': {paths: {embedding: {model: None for model in model_dict} for embedding in embeddings_dict} for paths in os.listdir("../data")},
            'list_email_classifiers': {email_classifier.name: None for email_classifier in self.email_classifiers},
            'choose_email_classifier': None,
            'change_strategy': {model: None for model in model_dict},
            'add_preprocessing': {feature: None for feature in features_dict},
            'train_model': {paths: None for paths in os.listdir("../data")},
            'display_evaluation': None,
            'undo': None,
            'exit': None
        }
        return NestedCompleter.from_nested_dict(completer_dict)

    def run_cli(self):

        parser = self.create_parser()
        history = InMemoryHistory()
        completer = self.create_completer()

        print("Startup complete")
        print("Welcome to Email Classifier application.")

        while True:
            completer = self.create_completer()
            try:            
                input_str = prompt("> ", completer=completer, history=history)
                if input_str.strip().lower() == 'exit':
                    print("Exiting CLI.")
                    break
                completer = self.create_completer()
                args = parser.parse_args(input_str.split())
                self.handle_input(args)
            except SystemExit:
                # argparse throws a SystemExit exception if parsing fails, we'll catch it to keep the loop running
                continue
            except Exception as e:
                print(f"Error: {e}")

    def handle_input(self, args) -> bool:
        match args.command:
            case "test":
                print("Args are:")
                for arg in vars(args):
                    if arg != 'command':
                        print(f"{arg}: {getattr(args, arg)}")
                print("test command executed")
            case "create_email_classifier":
                command = CreateEmailClassifierCommand(
                    email_classifiers=self.email_classifiers,
                    path=args.path,
                    embedding=args.embedding,
                    model=args.model,
                    name=args.name
                )
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "list_email_classifiers":
                command = ListEmailClassifiersCommand(
                    email_classifiers=self.email_classifiers)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "choose_email_classifier":
                command = ChooseEmailClassifierCommand(
                    email_classifiers=self.email_classifiers, name=args.name)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "add_emails":
                command = AddEmailsCommand(
                    email_classifier=self.email_classifiers[0])
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "classify_emails":
                command = ClassifyEmailCommand(
                    email_classifier=self.email_classifiers[0])
                self.command_invoker.set_command(command)
                self.command_invoker.execute()

            case "change_strategy":
                command = ChangeStrategyCommand(
                    email_classifier=self.email_classifiers[0], model_type=args.strategy)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "add_preprocessing":
                command = AddPreprocessingCommand(
                    email_classifier=self.email_classifiers[0], feature=args.feature)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "train_model":
                command = TrainModelCommand(
                    email_classifier=self.email_classifiers[0], path=args.path)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "display_evaluation":
                command = DisplayEvaluationCommand(
                    email_classifier=self.email_classifiers[0])
                self.command_invoker.set_command(command)
                self.command_invoker.execute()
            case "undo":
                self.command_invoker.undo()
            case "exit":
                print("Exiting CLI.")
                exit(0)
            case _:
                print("Unknown command")
        return False


if __name__ == "__main__":
    client = Client()
    client.run_cli()
