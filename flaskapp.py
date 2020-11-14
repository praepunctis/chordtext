# flaskapp.py
# Kelly Fesler (c) Nov 2020
# Modified from Soumya Gupta (c) Jan 2020

from flask import Flask

app = Flask(__name__)
@app.route("/")

def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)
