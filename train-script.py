# train-script.py
# Grab data from movie_data.csv and train a ML model.
# Kelly Fesler (c) Nov 2020
# Modified from Soumya Gupta (c) Jan 2020

# STEP 1: import -------------------------------------------

# Import libraries
import urllib.request
import os
import pandas as pd
import numpy as np
import nltk
import sklearn
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# STEP 2: read ---------------------------------------------

# Read in the large movie review dataset; display the first 3 lines
df = pd.read_csv('movie_data.csv', encoding='utf-8')
print("Loading data...\n")
data_top = df.head(3)
print(data_top)

# STEP 3: clean --------------------------------------------

# prepare tokenizer, stopwords, stemmer objects
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

# set up helper function to clean data:
def getStemmedReview(review):

    # turn to lowercase
    review = review.lower()
    review = review.replace("<br /><br />", " ")

    # tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]

    # stem
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    clean_review = ' '.join(stemmed_tokens)
    return clean_review

# tokenize & clean all reviews
print("")
print("Tokenizing & cleaning...")
df['review'].apply(getStemmedReview)

# STEP 4: split --------------------------------------------

print("Splitting...")

# split: 35k rows for training
X_train = df.loc[:35000, 'review'].values
Y_train = df.loc[:35000, 'sentiment'].values

# split: 15k rows for testing
X_test = df.loc[35000:, 'review'].values
Y_test = df.loc[35000:, 'sentiment'].values

# STEP 5: transform to feature vectors ---------------------

# set up vectorizer from sklearn
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8')

# train on the training data
print("Training...")
vectorizer.fit(X_train)

# after learning from training data, transform the test data
print("Transforming...")
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# STEP 6: create the ML model ------------------------------

print("Creating the model...")
model = LogisticRegression(solver='liblinear')
model.fit(X_train,Y_train)

print("ok!")

# print scores
print("")
print("Score on training data is: " + str(model.score(X_train,Y_train)))
print("Score on testing data is:" + str(model.score(X_test,Y_test)))
