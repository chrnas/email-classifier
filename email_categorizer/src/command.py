import os
from abc import ABC, abstractmethod
from data_preparation.simple_data_preprocessor_decorator_factory import SimpleDataPreProcessorDecoratorFactory
from email_classifier_facade import EmailClassifierFacade
from email_classifier_factory import EmailClassifierFactory
from data_preparation.dataset_loader import DatasetLoader
from models.model_factory import ModelFactory

# Abstract Command interface


class CommandInvoker:
    def __init__(self):
        self._command: Command
        self._history = []

    def set_command(self, command):
        self._command = command
        self._history.append(command)  # keep a history of executed commands

    def execute(self):
        self._command.execute()
        self._history.append(self._command)

    def undo(self):
        self._command.undo()


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    def undo(self):
        pass


class ClassifyEmailCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade):
        self.email_classifier = email_classifier

    def execute(self):
        pass

    def undo(self):
        print("There is no undo operation for the classify email command.")


class AddEmailsCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade, path: str):
        self.email_classifier = email_classifier
        self.path = path

    def execute(self):
        path = "../data/"+str(self.path)
        self.email_classifier.add_emails(path)
        print(f'Added emails to email classifier:{self.email_classifier.name}')

    def undo():
        print("There is no undo operation for the add emails command.")


class CreateEmailClassifierCommand(Command):
    def __init__(
        self,
        email_classifiers: list[EmailClassifierFacade],
        path: str,
        embedding: str,
        model: str,
        name: str,
    ):
        self.email_classifiers = email_classifiers
        self.path = path
        self.embedding = embedding
        self.model = model
        self.name = name
        self.created_email_classifier = None

    def execute(self):
        email_classifier_factory = EmailClassifierFactory()
        data_set_loader = DatasetLoader()
        path = "../data/"+str(self.path)
        df = data_set_loader.read_data(path)
        df = data_set_loader.renameColumns(df)
        self.email_classifier = email_classifier_factory.create_email_classifier(
            df,
            self.embedding,
            self.model,
            self.name
        )
        self.email_classifiers.append(self.email_classifier)
        self.created_email_classifier = self.email_classifier
        print('Added Email classifier:\n', self.created_email_classifier)

    def undo(self):
        self.email_classifiers.remove(self.created_email_classifier)
        print("Removed previous created email classifier to undo command.")


class ListEmailClassifiersCommand(Command):
    def __init__(self, email_classifiers: list[EmailClassifierFacade]):
        self.email_classifiers = email_classifiers

    def execute(self):
        print("Email classifiers:")
        for email_classifier in self.email_classifiers:
            print(email_classifier)

    def undo(self):
        os.system('cls')
        print(
            "The terminal has been cleared to no undo operation to list email classifiers.")


class ChooseEmailClassifierCommand(Command):

    def __init__(
        self,
        email_classifiers: list[EmailClassifierFacade],
        name: str
    ):
        self.email_classifiers = email_classifiers
        self.name = name
        self.previoius_index = None

    def execute(self):
        found = False
        index = 0
        for email_classifier in self.email_classifiers:
            if email_classifier.name == self.name:
                self.previoius_index = index
                self.email_classifiers.insert(
                    0, self.email_classifiers.pop(index))
                print(f"Active email classifier changed to {self.name}")
                found = True
                break
            index += 1

        if not found:
            print(f"Email classifier '{self.name}' not found.")

    def undo(self):
        email_classifier = self.email_classifiers.pop(0)
        self.email_classifiers.insert(self.previous_index, email_classifier)


class ChangeStrategyCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade, model_type: str):
        self.email_classifier = email_classifier
        self.model_type = model_type
        #self.old_model = email_classifier.model_context.modelstrat

    def execute(self):
        self.email_classifier.change_strategy(self.model_type)
        print("Model changed to", self.model_type)

    def undo(self):
        print("Undoing the previous change strategy command.")
        #self.email_classifier.model_context.change_strategy(
        #    self.email_classifier.old_model)


class AddPreprocessingCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade, feature: str):
        self.email_classifier = email_classifier
        self.feature = feature
        self.pre_processing_feature = None
        self.old_pre_processor = email_classifier.data_preprocessor

    def execute(self):
        #self.pre_processing_feature_decorator = SimpleDataPreProcessorDecoratorFactory().create_data_preprocessor(
        #    self.email_classifier.data_preprocessor, self.feature)
        #print(f"self pre processing feature: {
        #      self.pre_processing_feature_decorator}")
        self.email_classifier.add_preprocessing(self.feature)
        print(f"Preprocessing {self.feature} added")

    def undo(self):
        self.email_classifier.data_preprocessor = self.old_pre_processor


class TrainModelCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade, path: str):
        self.email_classifier = email_classifier
        self.path = path

    def execute(self):
        path = "../data/"+str(self.path)
        self.email_classifier.train_model(path)

    def undo():
        print("There is no undo operation for the train model command.")


class DisplayEvaluationCommand(Command):
    def __init__(self, email_classifier: EmailClassifierFacade):
        self.email_classifier = email_classifier

    def execute(self):
        self.email_classifier.display_evaluation()
        print("Model evaluation displayed")

    def undo():
        os.system('cls')
        print("The terminal has been cleared to no undo operation to display evaluations.")
