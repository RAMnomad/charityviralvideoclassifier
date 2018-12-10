import pandas as pd
import pickle
import boto3


s3 = boto3.client('s3')
def lambda_handler(event, context):
    # load model
    with open('/tmp/model.sav', 'wb') as model:
        s3.download_fileobj('charityviralreach', 'trainingdata/finalized_model.sav', model)
    loaded_model = pickle.load(model)

    # get final rekognition results which triggered the lambda
    key = event["Records"][0]["object"]["key"]
    with open('/tmp/rekogOutput.csv') as rekog:
        s3.download_fileobj('charityviralreach', key, rekog)

    # get tfidf results
    with open('/tmp/tfidf.csv') as tfidf:
        s3.download_fileobj('charityviralreach', key.replace("rekognitionResults", "srtResults"), tfidf)

    rekognition_results = pd.read_csv(rekog, delimiter=',', index_col=0)
    srt_results = pd.read_csv(tfidf, delimiter=',', index_col=0)

    combined_results = pd.concat([rekognition_results, srt_results], axis=1)

    result = loaded_model.predict_proba(combined_results.iloc[[0]])
    print(str(result[0][1] * 100) + '% chance of becoming viral content')