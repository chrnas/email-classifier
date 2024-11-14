

class ClassificationFactory:
    @staticmethod
    def create_classification_algorithm(classification_algorithm: str):

        if classification_algorithm == "bayes":
            return "bayes"
        elif classification_algorithm == "rainforest":
            return "rainforest"
        else:
            raise ValueError(f"Unknown algorithm: {classification_algorithm}.")