import boto3
import uuid
import time
# TODO: A bit sloppy, needs refactoring,
# TODO: reuse code for processing new videos run against finished model in lambda

# Initial training data creation script:
## Grabs all the videos currently uploaded to the bucket and processes them for label detection and
## celebrity recognition
## Writes jobIds to csv file for ease of retrieval once SNS shows completion.


def check_active_job_count(count):
    # max 20 active jobs on rekognition
    if (count + 1) % 20 == 0:
        time.sleep(1500)
    return count + 1

def main():
    s3 = boto3.resource('s3')
    rekog = boto3.client('rekognition')
    records = s3.Bucket('charityviralreach').objects.filter(Prefix='videos')
    result = open('results.csv', 'w')
    open_job_count = 0
    for record in records:
        if record.key == 'videos/':
            continue
        job = record.key.replace('videos/', '').replace(' ', '-').replace(',', '').replace("'", '').replace('&', '').replace('.mp4', '')
        label = rekog.start_label_detection(
            Video={
                'S3Object': {
                    'Bucket': 'charityviralreach',
                    'Name': record.key,
                }
            },
            MinConfidence=75,
            ClientRequestToken=str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': 'arn:aws:sns:us-west-2:561454936176:AmazonRekognition-charity-viral-video-processing',
                'RoleArn': 'arn:aws:iam::561454936176:role/Rekognition'
            },
            JobTag='labels:' + job
        )
        open_job_count = check_active_job_count(open_job_count)

        celeb = rekog.start_celebrity_recognition(
            Video={
                'S3Object': {
                    'Bucket': 'charityviralreach',
                    'Name': record.key,
                }
            },
            ClientRequestToken= str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': 'arn:aws:sns:us-west-2:561454936176:AmazonRekognition-charity-viral-video-processing',
                'RoleArn': 'arn:aws:iam::561454936176:role/Rekognition'
            },
            JobTag='celebrities:' + job
        )
        open_job_count = check_active_job_count(open_job_count)

        faces = rekog.start_face_detection(
            Video={
                'S3Object': {
                    'Bucket': 'charityviralreach',
                    'Name': record.key,
                }
            },
            ClientRequestToken= str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': 'arn:aws:sns:us-west-2:561454936176:AmazonRekognition-charity-viral-video-processing',
                'RoleArn': 'arn:aws:iam::561454936176:role/Rekognition'
            },
            JobTag='celebrities:' + job,
            FaceAttributes='ALL',
        )
        result.write(job + ',' + label["JobId"] + ',' + celeb['JobId'] + ',' + faces["JobId"] + '\r\n')
        open_job_count = check_active_job_count(open_job_count)


if __name__ == "__main__":
    main()
