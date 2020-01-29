% Computer Vision Coursework - Face Recognition and OCR
% Student: Lesley Dwyer
% Train CNN model 

clc;
clear all;

% Start timer;
tic;

% The code below has been adapted from this Mathworks example:
% https://uk.mathworks.com/help/deeplearning/examples/transfer-learning-using-alexnet.html

% Load images from prepared training database into Matlab
trainingDB = imageDatastore('trainingDatabase',...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Limit number of images in each folder to be the same
T = trainingDB.countEachLabel;
minCount = min(T.Count); % find smallest number of images
% Use splitEachLabel to reduce the image sets to min number
[trainingDB, unused] = splitEachLabel(trainingDB, minCount, 'randomize'); 

% Split the cropped images into 80% training and 20% validation
[training, validation] = splitEachLabel(trainingDB,0.8,'randomized');

% CNN Classifier - Use alexnet
net = alexnet;
inputSize = net.Layers(1).InputSize;

% Replace final 3 layers
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(training.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Train the network. Use an augmented image datastore to handle the
% different image sizes as alexnet requires size of 227x227
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augTraining = augmentedImageDatastore(inputSize(1:2),training, ...
    'DataAugmentation',imageAugmenter);

% Resize the validation images
augValidation = augmentedImageDatastore(inputSize(1:2),validation);

% Specify the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

CNN = trainNetwork(augTraining,layers,options);

% Classify training images
[YTrainPred,train_scores] = classify(CNN,augTraining);

% Classify validation images
[YPred,scores] = classify(CNN,augValidation);

% Evaluation of Classifier
% Evaluate on 5 validation images:
idx = randperm(numel(validation.Files),5);
figure
for i = 1:5
    subplot(1,5,i)
    I = readimage(validation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

% Compute the average training accuracy
YTraining = training.Labels;
training_accuracy = mean(YTrainPred == YTraining);

% Compute the average validation accuracy
YValidation = validation.Labels;
accuracy = mean(YPred == YValidation)

% End timer
toc;