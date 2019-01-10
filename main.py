#imports
import os, io
from flask import request, jsonify, Flask
from google.cloud import storage
from sklearn.externals import joblib

from core import TextClassifier

#loading buckets
GCS_BUCKET = os.environ['GCS_BUCKET']
GCS_MODEL_BLOB = os.environ['GCS_MODEL_BLOB']

app = Flask(__name__)

#loading model
@app.before_first_request
def _load_func():
    global model

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_BLOB)

    if blob.exists():
        f = io.BytesIO()
        blob.download_to_file(f)

        model = joblib.load(f)

    else:
        model = None

#fit model
@app.route('/fit', methods=['GET'])
def fit_func():
    tmp_filename = 'model.tmp'

    print('Fitting model...')
    model = TextClassifier().fit_model()

    print('Dumping model...')
    joblib.dump(model, tmp_filename)

    print('Prepare saving model...')
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    if not bucket.exists():
        bucket = client.create_bucket(GCS_BUCKET)

    blob = bucket.blob(GCS_MODEL_BLOB)

    print('Saving model...')
    with open(tmp_filename, 'rb') as f:
        blob.upload_from_file(f)

    return 'Model successfully fitted and dumped to gs://{}'.format(os.path.join(GCS_BUCKET, GCS_MODEL_BLOB))

#predict
@app.route('/predict', methods=['POST'])
def predict_func():
    if not model:
        _load_func()
        if not model:
            return 'Model not found at gs://{}'.format(os.path.join(GCS_BUCKET, GCS_MODEL_BLOB))

    print('Predicting...')
    in_text = request.get_json()['text']

    prediction_values = model.predict(in_text).tolist()

    return jsonify({"predictions": prediction_values})

#error handling
@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


#main
if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
