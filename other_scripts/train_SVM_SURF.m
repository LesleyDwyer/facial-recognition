% Computer Vision Coursework - Face Recognition and OCR
% Student: Lesley Dwyer
% Train SVM model with SURF features

clc;
clear all;

% Start timer;
tic;

% Limit number of images in each folder to be the same
trainingDB = imageSet('trainingDatabase','recursive'); %For testing
minCount = min([trainingDB.Count]); % find smallest number of images
trainingDB = partition(trainingDB, minCount, 'randomize'); % Use partition
% to reduce the image sets to min number

% Split the cropped images into 80% training and 20% validation
[training, validation] = partition(trainingDB, 0.8, 'randomize');

% Feature extraction
% The code below was adapted from Computer Vision Lab 5.
bag = bagOfFeatures(training); % extracts SURF features

% SVM Classifier
SVM_SURF = trainImageCategoryClassifier(training, bag); 

% Evaluation of SVM Classifier
% First evaluate on the training data:
confMatrixTrain = evaluate(SVM_SURF, training);

% Compute the training accuracy
mean(diag(confMatrixTrain))

% Next, evaluate on the validation set:
confMatrix = evaluate(SVM_SURF, validation);

% Compute the validation accuracy
mean(diag(confMatrix))

% End timer
toc;