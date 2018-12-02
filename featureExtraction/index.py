from rekognitionResultsLambda.package import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    rekognition_results = pd.read_csv('./rekognition/rekognitionOutput.csv', delimiter=',', index_col=0)
    srt_results = pd.read_csv('./srtProcessing/tfidfOutput.csv', delimiter=',', index_col=0)

    combined_results = pd.concat([rekognition_results, srt_results], axis=1)

    labels = pd.read_csv('labels.csv', index_col=0, header=None)

    train_features, test_features, train_labels, test_labels = train_test_split(combined_results, labels, test_size=0.25)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    # # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


if __name__ == "__main__":
    main()
