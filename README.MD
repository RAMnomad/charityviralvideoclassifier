Youtube uploading:
 - Leveraging pytube for both subtitle and mp4 downloads.
 - The cleanup lambda will delete after processing.

Feature Extraction: 
 - Amazon Rekognition Label detection, face detection, and celebrity detection
 - SciKit Learn - Bag of Words and TF-IDF scoring on subtitles
 - Merge features into one dataframe and pass into a Random Forest classifier

Trained Model will be deployed on Sagemaker

Application flow: 
 1) The UI will allow users to enter youtube video upload link, and receive an id for that video.
 2) Lambda function uploads, and attaches the id as a tag and sends an SQS message.
 3) UI listens on SQS and displays progress bar.
 4) Upload completion triggers second lambda to run Rekognition analysis, which receives a job id, and sends a job started message to SQS.
 5) Rekog finishes and publishes to an SNS topic which passes the message to SQS.
 6) SQS triggers another lambda which fetches the results and then calls the model endpoint and updates the queue with job finished and the resulting output.
 7) UI updates with the results passed to the message queue and displays a Snack Bar to let the user know the job is done.


