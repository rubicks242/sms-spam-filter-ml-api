from flask import Flask, request, jsonify, make_response
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re

app = Flask(__name__)

# Load the models
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
svc_model = joblib.load('svc_model.pkl')
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')
le = LabelEncoder()
le.classes_ = joblib.load('label_encoder_classes.pkl')
stem = PorterStemmer()

# Load the CountVectorizer
cv = joblib.load('count_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sms_message = data['message']

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

    results = {
        'naive_bayes': le.classes_[naive_bayes_prediction],
        'svc': le.classes_[svc_prediction],
        'logistic_regression': le.classes_[logistic_regression_prediction],
        'decision_tree': le.classes_[decision_tree_prediction]
    }

    response = make_response(jsonify(results))
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
