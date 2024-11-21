

class ClassifierConfigSingleton(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassifierConfigSingleton, cls).__new__(cls)

            # Configuration using dictionaries and variables
            cls._instance.preprocessing_features = [
                'deduplication',
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

            cls._instance.appgallery_input_columns = {
                'ticket_summary': "Ticket Summary",
                'interaction_content': "Interaction content"
            }

            cls._instance.appgallery_type_columns = {
                'test_columns': ["y2", "y3", "y4"],
                'classification_column': "y2",
                'grouped_column': "y1"
            }

        return cls._instance
