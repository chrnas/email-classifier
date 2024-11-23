from email_classifier_facade import EmailClassifierFacade
from command import ChangeStrategyCommand, ChooseEmailClassifierCommand, \
    AddEmailsCommand, CommandInvoker, \
    CreateEmailClassifierCommand, DisplayEvaluationCommand, \
    ListEmailClassifiersCommand, \
    AddPreprocessingCommand, \
    RemoveEmailClassifierCommand, \
    TrainModelCommand, ClassifyEmailCommand
from classifier_config_singleton import ClassifierConfigSingleton


class Client:

    email_classifiers: list[EmailClassifierFacade]
    command_invoker: CommandInvoker
    config_manager: ClassifierConfigSingleton

    def __init__(self):
        self.command_invoker = CommandInvoker()
        self.email_classifiers: list[EmailClassifierFacade] = []
        self.config_manager = ClassifierConfigSingleton()

    def handle_input(self, args) -> bool:
        match args.command:

            case "test":
                print("Args are:")
                for arg in vars(args):
                    if arg != 'command':
                        print(f"{arg}: {getattr(args, arg)}")
                print("test command executed")

            case "create_email_classifier":
                path = self.config_manager.data_folder_path + args.path
                command = CreateEmailClassifierCommand(
                    email_classifiers=self.email_classifiers,
                    path=path,
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

            case "remove_email_classifier":
                command = RemoveEmailClassifierCommand(
                    email_classifiers=self.email_classifiers, name=args.name)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()

            case "add_emails":
                path = self.config_manager.data_folder_path + args.path
                command = AddEmailsCommand(
                    email_classifier=self.email_classifiers[0], path=path)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()

            case "classify_emails":
                command = ClassifyEmailCommand(
                    email_classifier=self.email_classifiers[0])
                self.command_invoker.set_command(command)
                self.command_invoker.execute()

            case "change_strategy":
                command = ChangeStrategyCommand(
                    email_classifier=self.email_classifiers[0],
                    model_type=args.strategy)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()

            case "add_preprocessing":
                command = AddPreprocessingCommand(
                    email_classifier=self.email_classifiers[0],
                    feature=args.feature)
                self.command_invoker.set_command(command)
                self.command_invoker.execute()

            case "train_model":
                path = self.config_manager.data_folder_path + args.path
                command = TrainModelCommand(
                    email_classifier=self.email_classifiers[0],
                    path=path)
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
