import boto3
import csv
import json
import collections

# Gathers rekognition output and cleans and aggregates it for random forest classifier.
s3 = boto3.client('s3')
collected_labels = ['Human', 'Adventure', 'Beard', 'Animal', 'Back', 'Art', 'Brochure', 'Audience', 'Asleep', 'Arecaceae']
emotion_list = ['HAPPY', 'SAD', 'ANGRY', 'CONFUSED', 'DISGUSTED', 'SURPRISED', 'CALM', 'UNKNOWN']
rekog = boto3.client('rekognition')


def one_hot_labels(labels):
    return [labels.count(label) for label in collected_labels]


def process_labels(job_id):
    labelJob = rekog.get_label_detection(
        JobId=job_id,
        MaxResults=30,
        SortBy='NAME')
    labels = []
    for label in labelJob['Labels']:
        if label['Label']['Confidence'] > 80:
            labels.append(label['Label']['Name'])
    return one_hot_labels(labels)


def process_celebrities(job_id):
    celebJob = rekog.get_celebrity_recognition(
        JobId=job_id,
        SortBy='ID'
    )
    celebrities = []
    for celebrity in celebJob['Celebrities']:
        if celebrity['Celebrity']['Confidence'] > 60:
            celebrities.append(celebrity['Celebrity']['Name'])
    return [len(set(celebrities))]


def one_hot_emotions(emotions):
    sorted_emotions_array = []
    for emotion in emotion_list:
        if hasattr(emotions, 'emotion'):
            sorted_emotions_array.append(emotions[emotion])
        else: sorted_emotions_array.append(0)
    return sorted_emotions_array


def process_faces(job_id):
    faceJob = rekog.get_face_detection(
        JobId=job_id,
        MaxResults=100)
    filtered_faces = []
    for face in faceJob['Faces']:
        new_face = {k: v for k, v in face['Face'].items() if k in ['Emotions', 'Gender', 'AgeRange', 'Smile']}
        filtered_faces.append(new_face)
    emotions = collections.Counter(
        face['Emotions'][0]['Type'] for face in filtered_faces if face['Emotions'][0]['Confidence'] > 75)
    genders = collections.Counter(face['Gender']['Value'] for face in filtered_faces)
    women = genders['Female']
    men = genders['Male']
    average_ages = [(face['AgeRange']['High'] + face['AgeRange']['Low']) / 2 for face in filtered_faces]
    children = len([age for age in average_ages if age < 18])
    young_adults = len([age for age in average_ages if 18 < age < 30])
    adults = len([age for age in average_ages if 30 < age < 55])
    elderly = len([age for age in average_ages if age > 55])
    smiles = len([face for face in filtered_faces if face['Smile']['Value']])
    return one_hot_emotions(emotions).extend([women, men, children, young_adults, adults, elderly, smiles])


def gather_output(sqs_message):
    api_invocation_list = {
        "StartLabelDetection": process_labels,
        "StartCelebrityRecognition": process_celebrities,
        "StartFaceDetection": process_faces
    }
    job_id = sqs_message["JobId"]
    row = [sqs_message["JobTag"]]
    return row.extend(api_invocation_list[sqs_message["API"]](job_id))


def strip_prefix_suffix(key):
    return key.replace("processing/rekognitionResults/", "").replace(".csv", "")


def get_list_of_keys(tag):
    key = 'processing/retrievalKeys/' + tag + ".csv"
    response = s3.get_object(Bucket='charityviralreach', key=key)
    lines = response['Body'].read().splitlines(True)
    return lines[0].split(',')[1:]


def check_for_job_completion(tag):
    line = get_list_of_keys(tag)
    object_list = s3.list_objects(Bucket="charityviralreach", prefix="processing/rekognitionResults")
    key_list = [strip_prefix_suffix(obj["Contents"]["Key"]) for obj in object_list]
    return line.issublist(key_list)


def aggregate_rekog_data(tag):
    keys = get_list_of_keys(tag)
    row = []
    for key in keys:
        response = s3.get_object(Bucket='charityviralreach', key="processing/rekognitionResults/" + key + ".csv")
        lines = response['Body'].read().splitlines(True)
        line = lines[0].split(',')
        row.extend(line)

    header_row = ['file']
    header_row.extend(collected_labels)
    header_row.extend(['file', 'celebrity_count', 'HAPPY', 'SAD', 'ANGRY', 'CONFUSED', 'DISGUSTED', 'SURPRISED', 'CALM',
                       'UNKNOWN', 'women', 'men', 'children', 'young_adults', 'adults', 'elderly', 'smile_count'])
    output_file = open('/tmp/rekognitionOutput.csv', 'w')
    writer = csv.writer(output_file)
    writer.writerow(header_row)
    writer.writerow(row)
    output_file.close()
    s3.upload_file(Bucket='charityviralreach', Filename='/tmp/rekognitionOutput.csv',
                   key='processing/rekognitionResults/final' + tag + '.csv')


def lambda_handler(event, context):
    body = json.loads(event["Records"][0]["body"])
    if "Message" in body and "JobId" in body["Message"] and body["Message"]["Status"] == 'SUCCEEDED':
        output_file = open('/tmp/rekognitionOutput.csv', 'w')
        writer = csv.writer(output_file)
        data = gather_output(body["Message"])
        print(data)
        writer.writerow(data)
        output_file.close()
        s3.upload_file(Bucket='charityviralreach', Filename='/tmp/rekognitionOutput.csv',
                                  key='processing/rekognitionResults/' + body["Message"]["JobId"] + '.csv')
        if check_for_job_completion(body["Message"]["JobTag"]):
            aggregate_rekog_data(body["Message"]["JobTag"])
