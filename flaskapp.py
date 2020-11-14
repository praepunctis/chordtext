# flaskapp.py
# Kelly Fesler (c) Nov 2020
# Modified from Soumya Gupta (c) Jan 2020

# Import libraries
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import joblib

# Load .pkl files
loaded_model = joblib.load("./pkl_objects/model.pkl")
loaded_stop = joblib.load("./pkl_objects/stopwords.pkl")
loaded_vec = joblib.load("./pkl_objects/vectorizer.pkl")

app = Flask(__name__)

# classify(): uses the ML model to judge mood of input text
def classify(document):
    label = {0: 'sad', 1: 'happy'}
    X = loaded_vec.transform([document])
    Y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[Y], proba

# TextForm(): establishes the TextAreaField for input text
class TextForm(Form):
    text_field = TextAreaField('', [validators.DataRequired(),validators.length(min=15)])

# default route: display the text input form
@app.route('/')
def index():
    form = TextForm(request.form)
    return render_template('textform.html',form=form)

# results page: use classify() & the input to do some magic maths
@app.route('/results',methods=['POST'])
def results():
    form = TextForm(request.form)
    if request.method == 'POST' and form.validate():
        text_in = request.form['text_field']
        y, proba = classify(text_in)
        return render_template('results.html',content=text_in,prediction=y,probability=round(proba*100,2))
    return render_template('textform.html', form=form)

# do the thing
if __name__ == '__main__':
    app.run(debug=True)
