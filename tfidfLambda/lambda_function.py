import boto3
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

s3 = boto3.client('s3')

def has_letters(line):
    if re.search('[a-zA-Z]', line):
        return True
    return False


def has_no_text(line):
    l = line.strip()
    if not len(l):
        return True
    if l.isnumeric():
        return True
    if l[0] == '(' and l[-1] == ')':
        return True
    if not has_letters(line):
        return True
    return False


def clear_html(line):
    return re.sub('<[^<]+?>', '', line)


def clean(line):
    if has_no_text(line):
        return ''
    else:
        return clear_html(line) + '\r\n'

def filter_key(key):
    return key.replace('.srt', '').replace('processing/srt/', '').replace(' ', '-').replace(',', '').replace("'", '').replace('&', '')\
        .replace('.en', '').replace('transcribed', '').replace('translated', '').replace('.', '')


def fetch_and_decode_file(key):
    file_encoding = 'utf-8'
    file = s3.get_object(Bucket='charityviralreach', Key=key)
    new_lines = []
    for line in file.read().splitlines():
        new_line = clean(line.decode(file_encoding))
        new_lines.append(new_line)
    return new_lines



def top_mean_feat_counts(row, top_features):
    return [row.count(feature) for feature in top_features]


def top_tfidf_feats(row, features, top_n=1):
    # gets the features of a single srt returning them as an array
    top_n_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [features[i] for i in top_n_ids]
    return top_feats[:top_n]

def top_tfidf_values(row, top_n=1):
    top_n_ids = np.argsort(row)[::-1][:top_n]
    return [row[i] for i in top_n_ids]


def lambda_handler(event, context):
    document = fetch_and_decode_file(event['Records'][0]['object']['key'])
    top_mean_features_file = s3.get_object(Bucket="charityviralreach", Key="trainingdata/top_mean_features.csv")
    top_mean_features = [line.decode('utf-8').split(',') for line in top_mean_features_file.read().splitlines()]
    words = []
    for values in document:
        cleaned_words = re.split(';|,|\s|\.|\?|!', values)
        words.extend(cleaned_words)

    # get pickled model
    tf_file = s3.get_object(Bucket="charityviralreach", Key="trainingdata/tf.sav")
    tf_model = pickle.load(tf_file)

    # transform and merge new data into the matrix
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=tf_model)
    tfidf = transformer.fit_transform(loaded_vec.transform(document))
    print(tfidf)

    # get mean counts
