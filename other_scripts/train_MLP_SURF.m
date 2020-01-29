% Computer Vision Coursework - Face Recognition and OCR
% Student: Lesley Dwyer
% Train MLP model with SURF features

clc;
clear all;

% Start timer;
tic;

% Load images from prepared training database into Matlab
trainingDB = imageDatastore('trainingDatabase',...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
trainingDB.ReadFcn = @(loc)rgb2gray(imread(loc)); % change to grayscale

% Limit number of images in each folder to be the same
T = trainingDB.countEachLabel;
minCount = min(T.Count); % find smallest number of images
% Use splitEachLabel to reduce the image sets to min number
[trainingDB, unused] = splitEachLabel(trainingDB, minCount, 'randomize'); 

% Extract SURF features
% The code below was adapted from Computer Vision Lab 5.
bag = bagOfFeatures(trainingDB); % extracts SURF features

% Encode features 
trainingFeatures= encode(bag,trainingDB);

imageCount = 1; % To count images while in loop
 for i=1:size(T,1) % Loop through folders
     for j = 1:minCount %Loop through training images in each folder
         trainingIdx(imageCount, i) = 1;
         trainingLabel(1,i) = string(table2array(T(i,1)));
         imageCount = imageCount + 1;
     end
 end

x = trainingFeatures'; % Transpose for nnet
t = trainingIdx';  % Transpose for nnet

% The code below was adapted from the script generated from the Matlab 
% Neural Network Toolbox.

% Choose a Training Function
trainFcn = 'trainlm'; % Levenberg-Marquardt backpropogation

% Create a Feedforward Network
hiddenLayerSize = 10; % Number of hidden neurons
MLP_SURF = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
MLP_SURF.input.processFcns = {'removeconstantrows','mapminmax'};
MLP_SURF.trainParam.lr = 0.1; % learning rate

% Setup Division of Data for Training, Validation, Testing
MLP_SURF.divideFcn = 'dividerand';  % Divide data randomly
MLP_SURF.divideMode = 'sample';  % Divide up every sample
MLP_SURF.divideParam.trainRatio = 80/100;
MLP_SURF.divideParam.valRatio = 20/100;
MLP_SURF.divideParam.testRatio = 0/100;

% Choose a Performance Function
MLP_SURF.performFcn = 'mse';  % mean-squared error

% Choose Plot Functions
MLP_SURF.plotFcns = {'plotperform','plottrainstate','ploterrhist'};

% Train the Network
[MLP_SURF,tr] = train(MLP_SURF,x,t);

% Test the Network
y = MLP_SURF(x);
e = gsubtract(t,y);
performance = perform(MLP_SURF,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(MLP_SURF,trainTargets,y)
valPerformance = perform(MLP_SURF,valTargets,y)
testPerformance = perform(MLP_SURF,testTargets,y)
 
% Plots
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)

% Test one image
test_col = 800;
y2 = MLP_SURF(x(:,test_col));
yind2 = vec2ind(y2);
id_pred = trainingLabel(yind2);
x2 = (tind(:,test_col));
id_actual = trainingLabel(x2);

% End timer
toc;