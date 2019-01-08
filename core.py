#imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import dask.dataframe as dd
import os

#storage
from google.cloud import storage
from sklearn.externals import joblib

#modules
from core import TextClassifier

#loading buckets
GCS_BUCKET = os.environ['GCS_BUCKET']
GCS_DATA_BLOB = os.environ['GCS_DATA_BLOB']


class TextClassifier(object):
    """
    Wrapper class for Quora insincerity prediction
    """

    def __init__(self):
        
        self.stopwords = ['the','be','to','of','and','a','an']
        
        tfidfvectorizer = TfidfVectorizer(ngram_range=(1,4), top_words=self.stopwords)
        classifier = SVC(kernel='linear', probability=True)

        self.pipeline = Pipeline([('vectorizer', tfidfvectorizer), ('classifier', classifier)])
        

    def fit(self):
        """
        Fetch train.csv from bucket and fit
    	"""
    	
        df = dd.read_csv('gs://{}'.format(os.path.join(GCS_BUCKET, GCS_MODEL_BLOB)))
        df_pd = df.compute()
        
        self.pipeline.fit(df_pd.question_text, df_pd.target)
        

    def predict(self, input_texts):
        """
    	Args:
      		- input_texts: ['input_str']
    	Returns:
      		- predictions: [[prob_0, prob_1], ...]
    	"""

        probas = self.pipeline.predict_proba(input_texts)
        
        returned_probas = []

        for zero,one in probas:
            returned_probas.append({'label_0': zero, 'label_1': one})
            
        return returned_probas