import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import pickle


def main():
    rekognition_results = pd.read_csv('./rekognition/rekognitionOutput.csv', delimiter=',', index_col=0)
    srt_results = pd.read_csv('./srtProcessing/tfidfOutput.csv', delimiter=',', index_col=0)

    combined_results = pd.concat([rekognition_results, srt_results], axis=1)

    labels = pd.read_csv('labels.csv', index_col=0, header=None)

    train_features, test_features, train_labels, test_labels = train_test_split(combined_results, labels, test_size=0.25)

    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(train_features, train_labels.values.ravel())
    predictions = rf.predict(test_features)
    print(predictions)
    print(test_labels.values.ravel())
    # Calculate the absolute errors
    errors = abs(predictions - test_labels.values.ravel())
    # # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    probability = rf.predict_proba(test_features)
    print(probability)
    accuracy = accuracy_score(test_labels.values.ravel(), predictions) * 100
    print(accuracy)
    filename = 'finalized_model.sav'
    file = open(filename, 'wb')
    final_model = rf.fit(combined_results, labels.values.ravel())
    pickle.dump(final_model, file )
    file.close()



if __name__ == "__main__":
    main()
