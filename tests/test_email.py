from src.email_classifier_facade import EmailClassifierFacade
from src.email_classifier_factory import EmailClassifierFactory
from src.data_preparation.dataset_loader import DatasetLoader


def test_setup():
    df = DatasetLoader().read_data("../email_categorizer/data/AppGallery.csv")
    email_categorizer: EmailClassifierFacade = EmailClassifierFactory.create_email_classifier(
        df, "tfidf", "bayes", "email1"
    )
    email_categorizer.add_emails("../email_categorizer/data/AppGallery.csv")
    email_categorizer.classify_emails()

