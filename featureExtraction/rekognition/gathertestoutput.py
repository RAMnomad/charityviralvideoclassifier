import pandas
import boto3
import csv
import collections

# Gathers rekognition output and cleans it for random forest classifier.
# TODO: reuse in lambda for swf for user uploads.


def one_hot_emotions(emotions):
    emotion_list = ['HAPPY', 'SAD', 'ANGRY', 'CONFUSED', 'DISGUSTED', 'SURPRISED', 'CALM', 'UNKNOWN']
    sorted_emotions_array = []
    for emotion in emotion_list:
         if hasattr(emotions, 'emotion'):
             sorted_emotions_array.append(emotions[emotion])
         else: sorted_emotions_array.append(0)
    return sorted_emotions_array


def one_hot_labels(label_row, collected_labels):
    return [label_row.count(label) for label in collected_labels]


def main():
    rekog = boto3.client('rekognition')
    results = pandas.read_csv('results.csv', delimiter=',')
    tuples = [tuple(x) for x in results.values]

    outputFile = open('rekognitionOutput.csv', 'w')
    writer = csv.writer(outputFile)
    all_labels = []
    rows = []
    for jobIdTuple in tuples:
        row = []
        row.append(jobIdTuple[0])
        labelJob = rekog.get_label_detection(
            JobId=jobIdTuple[1],
            MaxResults=30,
            SortBy='NAME')
        celebJob = rekog.get_celebrity_recognition(
            JobId=jobIdTuple[2],
            SortBy='ID'
        )
        faceJob = rekog.get_face_detection(
            JobId=jobIdTuple[3],
            MaxResults=100)
# Columns for final csv:
#    labels with confidence level above 60%
#    harvest top 3-10 by frequency

        labels = []
        for label in labelJob['Labels']:
            if label['Label']['Confidence'] > 80:
                labels.append(label['Label']['Name'])
        all_labels.extend(labels)
        celebrities = []
        for celebrity in celebJob['Celebrities']:
            if celebrity['Celebrity']['Confidence'] > 60:
                celebrities.append(celebrity['Celebrity']['Name'])
        celebrity_count = len(set(celebrities))
        row.append(celebrity_count)
    # Columns for final csv:
    #    Celebrity count

        if faceJob['JobStatus'] == 'SUCCEEDED':
            filtered_faces = []
            for face in faceJob['Faces']:
                new_face = {k: v for k, v in face['Face'].items() if k in ['Emotions', 'Gender', 'AgeRange', 'Smile']}
                filtered_faces.append(new_face)
            emotions = collections.Counter(face['Emotions'][0]['Type'] for face in filtered_faces if face['Emotions'][0]['Confidence'] > 75)
            genders = collections.Counter(face['Gender']['Value'] for face in filtered_faces)
            women = genders['Female']
            men = genders['Male']
            average_ages = [(face['AgeRange']['High'] + face['AgeRange']['Low'])/2 for face in filtered_faces]
            children = len([age for age in average_ages if age < 18])
            young_adults = len([age for age in average_ages if 18 < age < 30])
            adults = len([age for age in average_ages if 30 < age < 55])
            elderly = len([age for age in average_ages if age > 55])
            smiles = len([face for face in filtered_faces if face['Smile']['Value']])
            row.extend(one_hot_emotions(emotions))
            row.extend([women, men, children, young_adults, adults, elderly, smiles])
# Columns for final csv:
#    emotion rankings, total number of faces, number of children, young adults, middle aged adults, and older adults,
#    number of men and women, number of smiles
        rows.append(row)

    collect_labels = [item[0] for item in collections.Counter(all_labels).most_common(10)]
    header_row = (['file', 'celebrity_count', 'HAPPY', 'SAD', 'ANGRY', 'CONFUSED', 'DISGUSTED', 'SURPRISED', 'CALM',
                   'UNKNOWN', 'women', 'men', 'children', 'young_adults', 'adults', 'elderly', 'smile_count'])
    header_row.extend(collect_labels)
    writer.writerow(header_row)
    for row in rows:
        row.extend(one_hot_labels(row, collect_labels))
        writer.writerow(row)


if __name__ == "__main__":
    main()
