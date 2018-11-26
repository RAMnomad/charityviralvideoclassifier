import pytube
import boto3
import os


s3 = boto3.client('s3')



def upload_subtitles_to_s3(data, key):
    s3.put_object(Bucket='charityviralreach', Body=data, Key='processing/srt/'+key+'.srt')


def handler(event, context):
    os.chdir('/tmp')
    link = event['queryStringParameters']['link']
    key = pytube.extract.video_id(link)

    def upload_video_to_s3(stream, filehandler):
        print(stream)
        print(filehandler)
        s3.upload_file(Bucket='charityviralreach', Filename=filehandler.name, Key='processing/video/' + key + '.mp4')

    yt = pytube.YouTube(link, on_complete_callback=upload_video_to_s3)
    stream = yt.streams.filter(progressive=True).filter(subtype='mp4').first()
    stream.download(filename=key+'.mp4')
    captions = yt.captions.get_by_language_code('en')
    upload_subtitles_to_s3(captions.generate_srt_captions(), key)
