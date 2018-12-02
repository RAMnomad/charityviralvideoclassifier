Charity Viral Reach Project
--
**Purpose**

In gathering research on what drives nonprofit engagement and attempting to improve my own outreach on nonprofit programs I manage, 
I find that much depends on getting the right message to the right stakeholder or influencer.  In charity outreach, 
emphasis is often placed on blanketing every social media outlet to get as many eyes on your message as possible. 
In recent years a little more thought has gone into the quality of that content/message. Some organizations have taken the time 
to show both good and bad practices through efforts like the Rusty Radiator awards. 

This project is an attempt at pulling together some beginner machine learning concepts I've been studying in a way which looks at the 
content itself in an attempt to find common themes which resonate with wide audiences and encourage sharing.

This is a stepping stone to future projects that will seek to target specific influencers with messages that resonate
with their primary motivations in charitable engagement by analyzing their social media feeds and networks. Before 
reaching more people, I want to ensure that my content is going to be as impactful as possible.

This repo could be used to classify other types of videos with different training data provided. Please see the dev setup notes below.
Keep in mind that this is a demo project in the very early stages and accuracy may not be optimal.

**Methods**

Youtube uploading:
 - Leveraging pytube for both subtitle and mp4 downloads.
 - The cleanup lambda will delete after processing.

Feature Extraction: 
 - Amazon Rekognition Label detection, face detection, and celebrity detection
 - SciKit Learn - Bag of Words and TF-IDF scoring on subtitles
 - Merge features into one dataframe and pass into a Random Forest classifier 

Trained Model will be deployed on Sagemaker and will predict probability of going viral

Application flow: 
 1) The UI will allow users to enter a youtube video upload link, and receive an id for that video.
 2) Lambda function uploads, and attaches the id as a tag and sends an SQS message.
 3) UI listens on SQS and displays progress bar.
 4) Upload completion triggers second lambda to run Rekognition analysis, which receives a job id, and sends a job started message to SQS.
 5) Rekog finishes and publishes to an SNS topic which passes the message to SQS.
 6) SQS triggers another lambda which fetches the results and then calls the model endpoint and updates the queue with job finished and the resulting output.
 7) UI updates with the results passed to the message queue.


**Lessons and Takeaways and Future Enhancements**

Currently the training dataset is far too small for much of the features extracted to be truly telling.
The dataset needs to be built up in one way or another. Charity video campaigns are a relatively niche subject,
so this might be a valid scenario for a transfer learning implementation, since resources are limited. 

Viewer reactions are probably the most useful data we could pull in this scenario, but due to time constraints, 
I did not make an attempt at this, since it would basically require being able to weed out troll comments and their responses on YouTube,
or volunteers to label each video with their impressions. 

**Developer Usage Notes**

If you wish to leverage this code for video classification:
- Set up your S3 bucket and edit the bucket names and prefixes to reflect your setup.
- Initial feature extraction and training can be done locally using the scripts in /featureExtraction. All videos and subtitle files should be uploaded
to your S3 bucket prior to running. 
- Each of the lambda folders contain the necessary packages and are zipped and ready to upload to Lambda.
- At minimum bucket notifications need to be set to:
  
  - trigger the Rekognition Start lambda on receiving a video to the S3 bucket with a particular prefix
  - trigger an SQS message when Rekognition results have been processed
  
 - SQS should be: 
 
   - Subscribed to the SNS topic Rekognition automatically sends a message to when the called API has completed its process. 
   - Set up to trigger the Rekognition Results lambda.
   
- The trigger for the downloader is an API Gateway call. 
    - You can set a path as a lambda proxy that passes a link query parameter

