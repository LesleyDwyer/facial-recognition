% Computer Vision Coursework - Face Recognition and OCR
% Student: Lesley Dwyer
% Train MLP model with HOG features

clc;
clear all;

% Start timer;
tic;

% Limit number of images in each folder to be the same
trainingDB = imageSet('trainingDatabase','recursive'); 
minCount = min([trainingDB.Count]); % find smallest number of images
trainingDB = partition(trainingDB, minCount, 'randomize'); % Use partition
% to reduce the image sets to min number

% Extract HOG features for all training images
% The code below was adapted from Computer Vision Lab 6 
trainingFeatures = zeros(size(trainingDB,2)*trainingDB(1).Count,1764); 
trainingIdx = zeros(size(trainingDB,2)*trainingDB(1).Count,size(trainingDB,2));
imageCount = 1; % To count images while in loop

for i=1:size(trainingDB,2) % Loop through folders
    for j = 1:trainingDB(i).Count %Loop through training images in each folder
        trainingFeatures(imageCount,:) = extractHOGFeatures(read(trainingDB(i),j));
        trainingIdx(imageCount, i) = 1; % create classes as one-hot encoding
        trainingLabel(1,i) = string(trainingDB(i).Description); % create actual labels
        imageCount = imageCount + 1;
    end
end

x = trainingFeatures'; % Transpose for nnet
t = trainingIdx'; % Transpose for nnet

% The code below was adapted from the script generated from the Matlab 
% Neural Network Toolbox.

% Choose a Training Function
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Feedforward Network
hiddenLayerSize = 10; % Number of hidden neurons
% MLP_HOG = feedforwardnet(hiddenLayerSize, trainFcn);
MLP_HOG = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
MLP_HOG.input.processFcns = {'removeconstantrows','mapminmax'};
MLP_HOG.trainParam.lr = 0.01; % learning rate

% Setup Division of Data for Training, Validation, Testing
MLP_HOG.divideFcn = 'dividerand';  % Divide data randomly
MLP_HOG.divideMode = 'sample';  % Divide up every sample
MLP_HOG.divideParam.trainRatio = 80/100;
MLP_HOG.divideParam.valRatio = 20/100;
MLP_HOG.divideParam.testRatio = 0/100;

% Choose a Performance Function
MLP_HOG.performFcn = 'mse';  % mean-squared error

% Choose Plot Functions
MLP_HOG.plotFcns = {'plotperform','plottrainstate','ploterrhist'};

% Train the Network
[MLP_HOG,tr] = train(MLP_HOG,x,t,'useGPU','yes');

% Test the Network
y = MLP_HOG(x,'useGPU','yes');
e = gsubtract(t,y);
performance = perform(MLP_HOG,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(MLP_HOG,trainTargets,y)
valPerformance = perform(MLP_HOG,valTargets,y)
testPerformance = perform(MLP_HOG,testTargets,y)
 
% Plots
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)

% End timer
toc;