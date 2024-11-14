from abc import ABC, abstractmethod
from data_preparation.simple_data_preprocessor_decorator_factory import SimpleDataPreProcessorDecoratorFactory
from email_classifier_facade import EmailClassifierFacade
from email_classifier_factory import EmailClassifierFactory
from data_preparation.dataset_loader import DatasetLoader
from models.model_factory import ModelFactory

# Abstract Command interface


class CommandExecutor:
    def __init__(self):
        self.history = []  # to keep track of commands executed

    def execute_command(self, command):
        command.execute()
        self.history.append(command)  # keep a history of executed commands


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


class ClassifyEmailCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade):
        self.email_classifier = email_classifier

    def execute(self):
        pass


class AddEmailsCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade):
        self.email_classifier = email_classifier

    def execute(self):
        self.email_classifier.add_emails()


class CreateEmailClassifierCommand(Command):
    def __init__(
        self,
        email_classifiers: list[EmailClassifierFacade],
        config: dict,
        path: str,
    ):
        self.config = config
        self.email_classifiers = email_classifiers
        self.path = path
        self.email_classifier = None

    def execute(self):
        data_set_loader = DatasetLoader()
        df = data_set_loader.read_data(self.path)
        df = data_set_loader.renameColumns(df)
        email_classifier_factory = EmailClassifierFactory()
        self.email_classifier = email_classifier_factory.create_email_classifier(
            df,
            self.config["embeddings"],
            self.config["pre_processing_features"],
            self.config["classification_algorithm"]
        )
        self.email_classifiers.append(self.email_classifier)
        print('Added Email classifier:\n', self.email_classifier)


class ListEmailClassifiersCommand(Command):
    def __init__(self, email_classifiers: list[EmailClassifierFacade]):
        self.email_classifiers = email_classifiers

    def execute(self):
        print("Email classifiers:")
        for email_classifier in self.email_classifiers:
            print(email_classifier.name)


class ChooseEmailClassifierCommand(Command):

    def __init__(
        self,
        email_classifiers: list[EmailClassifierFacade],
        name: str
    ):
        self.email_classifiers = email_classifiers
        self.name = name

    def execute(self):
        found = False
        index = 0
        for email_classifier in self.email_classifiers:
            if email_classifier.name == self.name:
                self.email_classifiers.insert(
                    0, self.email_classifiers.pop(index))
                print(f"Active email classifier changed to {self.name}")
                found = True
                break
            index += 1

        if not found:
            print(f"Email classifier '{self.name}' not found.")


class ChangeStrategyCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade, model: str):
        self.email_classifier = email_classifier
        self.model = model

    def execute(self):
        model = ModelFactory().create_classification_strategy(
            self.strategy)
        self.email_classifier.classification_strategy_contex.change_strategy(
            model)
        print("Model changed to", self.model)


class AddPreprocessingCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade, feature: str):
        self.email_classifier = email_classifier
        self.feature = feature
        self.pre_processing_feature = None

    def execute(self):
        self.pre_processing_feature_decorator = SimpleDataPreProcessorDecoratorFactory().create_data_preprocessor(
            self.email_classifier.data_preprocessor, self.feature)
        print(f"self pre processing feature: {
              self.pre_processing_feature_decorator}")
        self.email_classifier.add_preprocessing(
            self.pre_processing_feature_decorator)
        print(f"Preprocessing {self.feature} added")


class TrainModelCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade, path: str):
        self.email_classifier = email_classifier
        self.path = path

    def execute(self):
        print(self.email_classifier.data_preprocessor)
        self.email_classifier.train_model(self.path)
        print(f"Model trained with data from {self.path}")


class DisplayEvaluationCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade):
        self.email_classifier = email_classifier

    def execute(self):
        self.email_classifier.display_evaluation()
        print("Model evaluation displayed")
