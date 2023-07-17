import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import re

nltk.download('punkt')
nltk.download('stopwords')

stem = PorterStemmer()

# Training
dataset = pd.read_csv('spam.csv', encoding='latin-1')
sent = dataset.iloc[:, [1]]['v2']
label = dataset.iloc[:, [0]]['v1']

le = LabelEncoder()
label = le.fit_transform(label)

sentences = []
for sen in sent:
    senti = re.sub('[^A-Za-z]', ' ', sen)
    senti = senti.lower()
    words = word_tokenize(senti)
    word = [stem.stem(i) for i in words if i not in stopwords.words('english')]
    senti = ' '.join(word)
    sentences.append(senti)

cv = CountVectorizer(max_features=5000)
features = cv.fit_transform(sentences).toarray()

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size=0.2, random_state=7)

# Train the models
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(feature_train, label_train)

svc_model = SVC(kernel='linear')
svc_model.fit(feature_train, label_train)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(feature_train, label_train)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(feature_train, label_train)

# Save the trained models
joblib.dump(naive_bayes_model, 'naive_bayes_model.pkl')
joblib.dump(svc_model, 'svc_model.pkl')
joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')
joblib.dump(decision_tree_model, 'decision_tree_model.pkl')
joblib.dump(le.classes_, 'label_encoder_classes.pkl')
joblib.dump(cv, 'count_vectorizer.pkl')