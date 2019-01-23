# deploying-quora-classifier-gc

Deployed a simple Quora Insincere Question Detector as RestAPI, in the Google Cloud Environment, using Flask/Gunicorn for app deployment and scikit-learn for model building.

Background on the dataset: see [Quora Insicere Questions Kaggle Challenge](https://www.kaggle.com/c/quora-insincere-questions-classification).

### Remarks 

Followed the very handy [tutorial of Sylvain Truong](https://towardsdatascience.com/https-towardsdatascience-com-deploying-machine-learning-has-never-been-so-easy-bbdb500a39a). 
Adapted requirements.txt slightly for deployment (dask[complete] and gcsfs).


### Some basic commands: 
- Fit Model: curl --request GET https://quora-classifier-dot-green-wares-224816.appspot.com/fit
- Predict Model: 

	content_type="Content-Type: application/json" <br>
	request="POST" <br>
	data='{"text":["Why has Einstein become such a genius?", "My only intent is to make a stupid statement on Quora. How shall I proceed?"]}' <br>
	http="https://quora-classifier-dot-green-wares-224816.appspot.com/predict" <br>
	curl --header "$content_type" \ --request POST \ --data "$data" \ $http

- gcloud app logs tail -s quora-classifier

Models >500MB might result in the Joblib Memory Error, see [Joblib.dump Memory error](https://github.com/joblib/joblib/issues/66).
