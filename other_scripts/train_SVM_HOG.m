% Computer Vision Coursework - Face Recognition and OCR
% Student: Lesley Dwyer
% Train SVM model with HOG features

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

% Extract HOG features for all images
% The code below was adapted from Computer Vision Lab 6 

trainingFeatures = zeros(size(training,2)*training(1).Count,1764); 
featureCount = 1;

for i=1:size(training,2)
    for j = 1:training(i).Count
    trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
    trainingLabel{featureCount} = training(i).Description;
    featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

% SVM classifier
SVM_HOG = fitcecoc(trainingFeatures, trainingLabel); 

% Cross-validation of training data
CVMdl = crossval(SVM_HOG);
loss = kfoldLoss(CVMdl);
oofLabel = kfoldPredict(CVMdl);
figure;
ConfMat = confusionchart(trainingLabel,oofLabel,'RowSummary','total-normalized');

% Test 5 People from Test Set
figureNum = 1;
for person=1:5
    figure;
    for j = 1:2
    queryImage = read(validation(person),j);
    queryFeatures = extractHOGFeatures(queryImage);
    personLabel = predict(SVM_HOG,queryFeatures);
    
    % Map back to training set to find identity
    booleanIndex = strcmp(personLabel, personIndex);
    integerIndex = find(booleanIndex);
    subplot(2,2,figureNum);imshow(imresize(queryImage,3));title('Test Image');
    subplot(2,2,figureNum+1);imshow(imresize(read(training(integerIndex),1),3));title('Predicted Face');
    figureNum = figureNum+2;
    end

figureNum = 1;
end

% End timer;
toc;