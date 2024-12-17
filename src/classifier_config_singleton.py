

class ClassifierConfigSingleton(object):
    _instance = None

    def __new__(cls):
        """Singleton instance creation."""
        if cls._instance is None:
            cls._instance = super(ClassifierConfigSingleton, cls).__new__(cls)

            # configurations for the cli and client
            cls._instance.data_folder_path = "./data/"

            cls._instance.state_folder_path = "./state/"

            cls._instance.state_file_name = "client_state.pkl"

            cls._instance.out_folder_path = "./out/"

            cls._instance.preprocessing_features = [
                'unicode_conversion',
                'noise_removal',
                'translation']

            cls._instance.embeddings = [
                'tfidf',
                'sentence_transformer',
                'tfidf_sentence_transformer']

            cls._instance.models = ['bayes', 'randomforest', 'svc']

            # Configurations for the email classifier
            cls._instance.ticket_summary = "Ticket Summary"

            cls._instance.interaction_content = "Interaction content"

            cls.classification_column = "y2"

            cls._instance.type_columns = [
                "y1", "y2", "y3", "y4"]

            cls._instance.type_columns_names = [
                "Type 1", "Type 2", "Type 3", "Type 4"]

        return cls._instance
