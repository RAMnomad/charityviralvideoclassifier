from pytube import YouTube
import boto3

s3 = boto3.client('s3')
bucket = s3.Bucket('charityviralreach')


def upload_video_to_s3(data):
    bucket.putObject(Body=data, Key='processing/video/'+key+'.mp4')


def upload_subtitles_to_s3(data, key):
    bucket.putObject(Body=data, Key='processing/srt/'+key+'.srt')


def trim_link(link):
    return link[link.index('?v=')+1:]


def handler(event, context):
    yt = YouTube(event.queryStringParameters.link, on_complete_callback=upload_video_to_s3)
    yt.streams.filter(progressive=True).filter(subtype='mp4').first().download()
    captions = yt.captions.get_by_language_code('en')
    upload_subtitles_to_s3(captions)
