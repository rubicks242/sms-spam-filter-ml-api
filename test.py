import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re

nltk.download('punkt')
nltk.download('stopwords')

# Load the models
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
svc_model = joblib.load('svc_model.pkl')
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')
cv = joblib.load('count_vectorizer.pkl')

le = LabelEncoder()
le.classes_ = joblib.load('label_encoder_classes.pkl')

stem = PorterStemmer()

# Testing
sms_message = input("Enter the message: ")

test_sentence = []
senti = re.sub('[^A-Za-z]', ' ', sms_message)
senti = senti.lower()
words = word_tokenize(senti)
word = [stem.stem(i) for i in words if i not in stopwords.words('english')]
senti = ' '.join(word)
test_sentence.append(senti)

test_features = cv.transform(test_sentence).toarray()

naive_bayes_prediction = naive_bayes_model.predict(test_features)[0]
svc_prediction = svc_model.predict(test_features)[0]
logistic_regression_prediction = logistic_regression_model.predict(test_features)[0]
decision_tree_prediction = decision_tree_model.predict(test_features)[0]

# Convert predictions back to original labels
naive_bayes_class = le.inverse_transform([naive_bayes_prediction])[0]
svc_class = le.inverse_transform([svc_prediction])[0]
logistic_regression_class = le.inverse_transform([logistic_regression_prediction])[0]
decision_tree_class = le.inverse_transform([decision_tree_prediction])[0]

print("Naive Bayes Prediction:", naive_bayes_class)
print("SVC Prediction:", svc_class)
print("Logistic Regression Prediction:", logistic_regression_class)
print("Decision Tree Prediction:", decision_tree_class)
