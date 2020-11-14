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
from sklearn.feature_extraction_text import TfidfVectorizer

# STEP 2: read ---------------------------------------------

# Read in the large movie review dataset; display the first 3 lines
df = pd.read_csv('movie_data.csv', encoding='utf-8')
data_top = df.head(3)
print(data_top)

# STEP 3: clean --------------------------------------------

# init Objects
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

# STEP 4: split --------------------------------------------

# clean all reviews
df['review'].apply(getStemmedReview)

# split: 35k rows for training
X_train = df.loc[:35000, 'review'].values
Y_train = df.loc[:35000, 'sentiment'].values

# split: 15k rows for testing
X_test = df.loc[35000:, 'review'].values
Y_test = df.loc[35000:, 'sentiment'].values

# STEP 5: transform to feature vectors ---------------------

# set up vectorizer from sklearn
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8')
