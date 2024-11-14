import argparse
from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.history import InMemoryHistory
from email_classifier_facade import EmailClassifierFacade
from email_classifier_factory import EmailClassifierFactory
from data_preparation.dataset_loader import DatasetLoader
from command import ChangeStrategyCommand, ChooseEmailClassifierCommand, \
    CommandExecutor, AddEmailsCommand, \
    CreateEmailClassifierCommand, DisplayEvaluationCommand, ListEmailClassifiersCommand, \
    AddPreprocessingCommand, TrainModelCommand, ClassifyEmailCommand


class Client:

    email_classifiers: list[EmailClassifierFacade]
    data_set_loader: DatasetLoader
    command_executor: CommandExecutor

    def __init__(self):
        self.data_set_loader = DatasetLoader()
        self.command_executor = CommandExecutor()
        self.email_classifiers: list[EmailClassifierFacade] = []

    def print_startup(self):
        print("Startup complete")
        print("Welcome to Email Classifier application.")

    def run_cli(self):
        parser = create_parser()
        history = InMemoryHistory()
        completer = create_completer()

        self.print_startup()

        while True:
            try:
                input_str = prompt("> ", completer=completer, history=history)
                if input_str.strip().lower() == 'exit':
                    print("Exiting CLI.")
                    break
                args = parser.parse_args(input_str.split())
                self.handle_command(args)
            except SystemExit:
                # argparse throws a SystemExit exception if parsing fails, we'll catch it to keep the loop running
                continue
            except Exception as e:
                print(f"Error: {e}")

    def handle_command(self, args) -> bool:
        match args.command:
            case "test":
                print("Args are:")
                for arg in vars(args):
                    if arg != 'command':
                        print(f"{arg}: {getattr(args, arg)}")
                print("test command executed")
            case "add_emails":
                command = AddEmailsCommand(
                    email_classifier=self.email_classifiers[0])
            case "classify_emails":
                command = ClassifyEmailCommand(
                    email_classifier=self.email_classifiers[0])
                self.command_executor.execute_command(command)
            case "create_email_classifier":
                config = {
                    "embeddings": "tfidf",
                    "pre_processing_features": ["noise_removal", "deduplication"],
                    "classification_algorithm": "rainforest"
                }
                # print(args.path)
                # df = self.data_set_loader.read_data(args.path)
                # df = self.data_set_loader.renameColumns(df)
                command = CreateEmailClassifierCommand(
                    email_classifiers=self.email_classifiers,
                    config=config,
                    path=args.path
                )
                self.command_executor.execute_command(command)
                print(self.email_classifiers)
                """data_set_loader = DatasetLoader()
                df = data_set_loader.read_data(args.path)
                df = data_set_loader.renameColumns(df)
                self.email_classifier = EmailClassifierFactory().create_email_classifier(
                    df=df,
                    embeddings=config["embeddings"],
                    pre_processing_features=config["pre_processing_features"],
                    classification_algorithm=config["classification_algorithm"]
                )
                self.email_classifier.train_model(args.path)
                self.email_classifiers.append(self.email_classifier)"""
            case "list_email_classifiers":
                print(self.email_classifiers)
                command = ListEmailClassifiersCommand(
                    email_classifiers=self.email_classifiers
                )
                self.command_executor.execute_command(command)
            case "choose_email_classifier":
                command = ChooseEmailClassifierCommand(
                    self.email_classifiers, args.name)
                self.command_executor.execute_command(command)
            case "change_strategy":
                command = ChangeStrategyCommand(
                    email_classifier=self.email_classifiers[0], strategy=args.strategy)
                self.command_executor.execute_command(command)
            case "add_preprocessing":
                command = AddPreprocessingCommand(
                    email_classifier=self.email_classifiers[0], feature=args.feature)
                self.command_executor.execute_command(command)
                print("Preprocessing {args.command} added")
            case "train_model":
                command = TrainModelCommand(
                    email_classifier=self.email_classifiers[0], path=args.path)
                self.command_executor.execute_command(command)
            case "display_evaluation":
                command = DisplayEvaluationCommand(
                    email_classifier=self.email_classifiers[0])
                self.command_executor.execute_command(command)
            case "exit":
                print("Exiting CLI.")
                exit(0)
            case _:
                print("Unknown command")
        return False


def create_parser() -> argparse.ArgumentParser:
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

    subparsers.add_parser(
        'classify_emails', help='Classify email comand.')

    # Create email classifier command
    create_email_classifier_parser = subparsers.add_parser(
        'create_email_classifier', help='Create an email classifier.')
    create_email_classifier_parser.add_argument(
        'path', help='Path to the email files.')
     # PosList command
    choose_email_classifier = subparsers.add_parser(
        'choose_email_classifier', help='Choose email classifier.')
    choose_email_classifier.add_argument('name', help='Name to the email classifier.')

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
        'path', help='Path for trainng data.')

    # Display Evaluation command
    subparsers.add_parser(
        'display_evaluation', help='Display the evaluation of the current positioning method.')

    # List email calssifiers command
    subparsers.add_parser(
        'list_email_classifiers', help='List email classifiers.')

    # Exit command
    subparsers.add_parser('exit', help='Exit the CLI.')

    return parser


def create_completer() -> NestedCompleter:

    # Extract method names and their possible settings
    completer_dict = {
        'test': None,
        'add_emails': None,
        'classify_emails': None,
        'create_email_classifier': {"../data/AppGallery.csv": None},
        'list_email_classifiers': None,
        'choose_email_classifier': None,
        'change_strategy': {'bayes', 'rainforest'},
        'add_preprocessing': {'deduplication': None, 'unicode_conversion': None, 'noise_removal':None, 'translation': None},
        'train_model': {"../data/AppGallery.csv": None},
        'display_evaluation': None,
        'exit': None
    }
    return NestedCompleter.from_nested_dict(completer_dict)


if __name__ == "__main__":
    client = Client()
    client.run_cli()
