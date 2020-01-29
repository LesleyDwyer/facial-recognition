# facial-recognition

This is coursework for my Computer Vision module at City, University of London in 2018-19. We were given several images of our classmates (individual and group). We were tasked with preparing the image data, trying different feature types and classifier combinations for facial recognition and reporting the results. (I have removed all face images from the report that I submitted). I used HOG and SIFT feature types and CNN, MLP and SVM classifiers. We were also tasked with using OCR to recognise the digits each person was holding in the photo. Full details can be found in the report, and all code is included below. To run the code, you will need MATLAB (version R2018b was used to create this).

All files:
1) Lesley Dwyer Computer Vision Report with Face images removed.pdf - report
2) RecogniseFace.m - function for face recognition
3) detectNum.m - function for digit recognition

trained_models_and_supporting_files folder:
1) bag.mat - bag created for Bag of Fatures; needed to predict with MLP_SURF
2) CNN.mat - CNN model (*Unavailable - exceeded Github's file limit)
3) MLP_HOG.mat - MLP model with HOG features
4) MLP_SURF.mat - MLP model with SURF features
5) SVM_HOG.mat - SVM model with HOG features (*Unavailable - exceeded Github's file limit)
6) SVM_SURF.mat - SVM model with SURF features
7) trainingLabel.mat - lookup to the labels from the indexes; needed to predict with MLPs

other_scripts folder:
1) image_preprocessing.m - code to create image training database; extracts images from videos, crops and resizes images and save into new folders
2) labels_faces - code to produce a figure of all faces and labels; used for testing group images
3) testing_group - code to plot labels of faces onto group image; used for testing group images
4) train_CNN.m - code to train CNN model
5) train_MLP_HOG.m - code to train MLP model with HOG features
6) train_MLP_SURF.m - code to train MLP model with SURF features
7) train_SVM_HOG.m - code to train SVM model with HOG features
8) train_SVM_SURF.m - code to train SVM model with SURF features

TO RUN the RecogniseFace function:
1) Save RecogniseFace.m and all files from the trained_models_and_supporting_files folder to the same folder as the test image file.
2) The function requires the following arguments as inputs*:
	- Image name, e.g. 'IMG_name.jpg'. 
	- Classifier name. Valid values are: 'CNN', 'SVM', 'MLP'
	- Feature type. Valid values are 'HOG', 'SURF', '' (for CNN, this is not required, so enter '')
3) In the Matlab command line, enter this with your arguments in quotes: 
	RecogniseFace('IMG_name.jpg','featureType', 'classifierName') 
4) It returns the following, for each face in the image, where id is the person's label and x-coordinate and y-coordinate is the location of the face:
<br/>	id	x-coordinate	y-coordinate
<br/>
<br/> *NOTE: - CNN and SVM-HOG combination can not be run from here, as the CNN and SVM-HOG model files exceeded Github's file limit.

TO RUN the detectNum function:
1) Save detectNum.m file to the same folder as the test image or video file.
2) The function requires the following arguments as inputs:
	- image or video name, e.g. 'IMG_name.jpg' or 'IMG_name.mov'
3) In the Matlab command line, enter this with your argument: 
	detectNum('IMG_name.jpg') 
4) It returns the number identified.


