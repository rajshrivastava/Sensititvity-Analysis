# Sensititvity-Analysis
This module uses a feed forward neural network to detect a sensitive email. It is a classifier where the context of sensitivity is learned based on the emails which are labelled as sensitive. It has an important application in email editors where confidential emails can be checked before being sent ahead.

The code of Sensitivity classifier.py has now been divided into three files:
- skipgram_generator.py : It trains word embeddings for the words of the entire data-set before training the classifier. It runs on the generated_data.csv (entire data-set)
- training.py: It trains a neural network to classify the mails as sensitive or non-sensitive. It runs on training_data.csv .
- testing.py: It evaluates the trained network and generates various performance metrics including the confusion matrix. It runs on testing_data.csv .


Next part: [Recipient Analysis](https://github.com/rajshrivastava/Recipient-Analysis)
