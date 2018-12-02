import boto3
import uuid

# TODO: A bit sloppy, needs refactoring,
# Initializes feature extraction from video using rekognition


s3 = boto3.resource('s3')


def write_csv_to_s3(job_tag):
    s3.upload_file(Bucket='charityviralreach', Filename="/tmp/results.csv", Key='processing/retrievalKeys/'+job_tag+'.csv')


def lambda_handler(event, context):
    rekog = boto3.client('rekognition')
    records = event["Records"]
    result = open('/tmp/results.csv', 'w')
    for record in records:
        key = record["S3"]["Object"]["Key"]
        if key == 'video/':
            continue
        job = key.replace('video/', '').replace(' ', '-').replace(',', '').replace("'", '').replace('&', '').replace('.mp4', '')
        label = rekog.start_label_detection(
            Video={
                'S3Object': {
                    'Bucket': 'charityviralreach',
                    'Name': key,
                }
            },
            MinConfidence=75,
            ClientRequestToken=str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': 'arn:aws:sns:us-west-2:561454936176:AmazonRekognition-charity-viral-video-processing',
                'RoleArn': 'arn:aws:iam::561454936176:role/Rekognition'
            },
            JobTag= job
        )

        celeb = rekog.start_celebrity_recognition(
            Video={
                'S3Object': {
                    'Bucket': 'charityviralreach',
                    'Name': key,
                }
            },
            ClientRequestToken= str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': 'arn:aws:sns:us-west-2:561454936176:AmazonRekognition-charity-viral-video-processing',
                'RoleArn': 'arn:aws:iam::561454936176:role/Rekognition'
            },
            JobTag=job
        )

        faces = rekog.start_face_detection(
            Video={
                'S3Object': {
                    'Bucket': 'charityviralreach',
                    'Name': key,
                }
            },
            ClientRequestToken= str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': 'arn:aws:sns:us-west-2:561454936176:AmazonRekognition-charity-viral-video-processing',
                'RoleArn': 'arn:aws:iam::561454936176:role/Rekognition'
            },
            JobTag=job,
            FaceAttributes='ALL',
        )
        result.write(job + ',' + label["JobId"] + ',' + celeb['JobId'] + ',' + faces["JobId"] + '\r\n')
        write_csv_to_s3(job)
