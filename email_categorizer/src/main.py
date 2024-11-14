from email_classifier import EmailClassifier

# Code will start executing from following line
if __name__ == '__main__':
    # train email classifier
    email_classifier = EmailClassifier()
    email_classifier.train_model("/Users/patrickvorreiter/Documents/Studium/2024 Wintersemester/Systems Analysis and Design/email-categorizer/email_categorizer/data/AppGallery.csv")
    
    # Print results
    email_classifier.printModelEvaluation()

    # classify email
    # email_classifier.classify_email("email")
