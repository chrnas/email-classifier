from email_classifier import EmailClassifier

# Code will start executing from following line
if __name__ == '__main__':
    # Classify email
    email_classifier = EmailClassifier()
    email_classifier.train_model("../data/AppGallery.csv")
    email_classifier.classify_email("email")
    
    # Print results
    email_classifier.printModelEvaluation()
