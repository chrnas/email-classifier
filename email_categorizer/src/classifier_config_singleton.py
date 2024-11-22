

class ClassifierConfigSingleton(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassifierConfigSingleton, cls).__new__(cls)

            # Configuration using dictionaries and variables
            cls._instance.preprocessing_features = [
                'unicode_conversion',
                'noise_removal',
                'translation']

            cls._instance.embeddings = [
                'tfidf',
                'wordcount',
                'sentence_transformer']

            cls._instance.models = ['bayes', 'randomforest', 'svc']

            cls._instance.default_classifier_config = {
                'preprocessing': [],
                'embedding': 'tfidf',
                'model': 'randomforest'
            }

            cls._instance.data_folder_path = "../data/"

            cls._instance.input_columns = {
                'ticket_summary': "Ticket Summary",
                'interaction_content': "Interaction content"
            }

            cls._instance.ticket_summary = "Ticket Summary"

            cls._instance.interaction_content = "Interaction content"

            cls.classification_column = "y1"

            cls._instance.type_columns = [
                "y1", "y2", "y3", "y4"]

            cls._instance.type_columns_names = [
                "Type 1", "Type 2", "Type 3", "Type 4"]

        return cls._instance
