import threading

class ClassifierConfig:
    _instance = None
    _lock = threading.Lock()  # Lock to ensure thread safety

    def __init__(self):
        if ClassifierConfig._instance is not None:
            raise Exception("This class is a singleton! Use get_instance() method to get the instance.")
        
        # Initialize your configuration settings here
        self.config = {
            "embedding_types": ["tfidf"],
            "model_type": "RandomForest",
            "preprocessing_steps": ["de_duplication", "translate_to_en", 
                                    "remove_noise", "convert_to_unicode"]
        }

    @staticmethod
    def get_instance():
        with ClassifierConfig._lock:  # Ensure thread safety
            if ClassifierConfig._instance is None:
                print("Creating unique instance of ClassifierConfig")
                ClassifierConfig._instance = ClassifierConfig()
            else:
                print("Returning instance of ClassifierConfig")
        return ClassifierConfig._instance

    def get_config(self):
        return self.config
    
    def update_config(self, key, value):
        self.config[key] = value
    
        # Methods to add embeddings and preprocessing steps
    def add_embedding(self, embedding_type):
        if embedding_type not in self.config["embedding_types"]:
            self.config["embedding_types"].append(embedding_type)

    def add_preprocessing_step(self, step):
        if step not in self.config["preprocessing_steps"]:
            self.config["preprocessing_steps"].append(step)

    # Methods to remove embeddings and preprocessing steps
    def remove_embedding(self, embedding_type):
        if embedding_type in self.config["embedding_types"]:
            self.config["embedding_types"].remove(embedding_type)

    def remove_preprocessing_step(self, step):
        if step in self.config["preprocessing_steps"]:
            self.config["preprocessing_steps"].remove(step)
