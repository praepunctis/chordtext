# chordtext
*Turn text into chords with a little ML magic.*

---

> *This project is still under construction, but stay tuned for updates!*


Have you written lyrics for a song, but don't know what tune to set it to? 



tl;dr -- I'm creating a web app that uses sentiment analysis to **turn user-input lyrics into musical chords.**


### Here's how it works so far:
* I leveraged the **Large Movie Review Dataset,** made publicly available by [Andrew Maas et al](https://ai.stanford.edu/~amaas/data/sentiment/). It contains 50,000 movie reviews mapped to scores based on the general tone of the review. This dataset was introduced in their 2011 ACL paper and is cited at the footer.
* I used some **Python** libraries (`pandas`, `numpy`, `scikit-learn`, `nltk`) to train a machine learning model off of this dataset. Given user-input text, the model is capable of judging its general sentiment with about 90% accuracy.
* I have a **Flask** app (and some HTML/CSS) that handles collecting the user-input text and using the ML model I trained to make an informed estimate of the lyrics' mood.
* This app is [hosted on PythonAnywhere](http://praepunctis.pythonanywhere.com/) and can be accessed anywhere :)


---
### Acknowledgements

* Citation for the 2011 ACL paper about the Large Movie Review Dataset:
  * *Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).*

* The machine learning portion of this project was guided by a tutorial called [Embedding Sentiment Analysis Model into a Web Application](https://medium.com/analytics-vidhya/embedding-sentiment-analysis-model-into-a-web-application-93b76ab6348c) by Soumya Gupta. `train-script.py` is cited appropriately. While carefully transcribing and cleaning the code, I fixed some syntax errors, import statements (after many hours of fighting Python dependencies), and authored the console outputs and all annotations within the program.
  * Starter code © Soumya Gupta, 2019. Used in compliance with the [MIT license](https://choosealicense.com/licenses/mit/).

* I got some last-minute CSS inspiration from an open-source program by [Mohammad Abdul Mohaiman](https://codepen.io/mohaiman/pen/MQqMyo). Without this, Chordtext would look a little less exciting. 
