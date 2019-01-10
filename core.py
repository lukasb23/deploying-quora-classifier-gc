#imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import dask.dataframe as dd
import os

#storage
from google.cloud import storage
from sklearn.externals import joblib

#loading buckets
GCS_BUCKET = os.environ['GCS_BUCKET']
GCS_DATA_BLOB = os.environ['GCS_DATA_BLOB']


class TextClassifier(object):
    """
    Wrapper class for Quora insincerity prediction
    """

    def __init__(self):
        
        self.stopwords = ['the','be','to','of','and','a','an']
        
        tfidfvectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words=self.stopwords)
        classifier = LinearSVC()

        self.pipeline = Pipeline([('vectorizer', tfidfvectorizer), ('classifier', classifier)])
        

    def fit_model(self):
        """
        Fetch train.csv and train, 
        illustration for on first 100000 rows
    	"""

        print('Fetching gs://{}'.format(os.path.join(GCS_BUCKET, GCS_DATA_BLOB)))
        df = dd.read_csv('gs://{}'.format(os.path.join(GCS_BUCKET, GCS_DATA_BLOB)))
        print('Df reading...')
        df_pd = df.compute()
        df_prd = df_pd.head(100000)
        print('Columns:', df_pd.columns, ', Length:', len(df_pd))

        model = self.pipeline.fit(df_pd.question_text, df_pd.target)
        print('Fitted model...')
        return model
