function [P] = RecogniseFace(I, featureType, classifierName)
% 
% RecogniseFace face recognition function 
% Student: Lesley Dwyer
% Built with Matlab version R2018b

% RecogniseFace function takes an image, featureType (HOG or SURF) and 
% classifierName (CNN, MLP, or SVM) as inputs. (For CNN, featureType can be
% null). It uses the featureType specified to extract features from the 
% image. It uses the classifierName specified to predict the person in the
% image. The classifier has been trained on a database of images of all
% people in the class. Each person has been assigned a number as the label.
% The output of this function, when predicted correctly, will produce the
% number and location of the person's face (as x and y coordinates) for
% each person in the image.

% First, detect faces in image I
% The face detection code was adapted from the Mathworks face detection example:
% https://uk.mathworks.com/help/vision/ref/vision.cascadeobjectdetector-system-object.html
A=imread(I);
myFaceDetector = vision.CascadeObjectDetector('MergeThreshold',6,'MinSize',[75 75]); % Use face detector to detect faces
BBOX = myFaceDetector(A); 
N = size(BBOX,1); % Count number of faces

if N == 0 % if no faces are detected, try rotating image
    A = imrotate(A,270);
    myFaceDetector = vision.CascadeObjectDetector('MergeThreshold',6,'MinSize',[75 75]); % Detect faces
    BBOX = myFaceDetector(A); 
    N = size(BBOX,1); % Count number of faces
end
   
P = zeros(N,3); % Create a matrix for the result, P

for i = 1:N % loop through all faces in image
    face = imcrop(A,BBOX(i,:));
    faceResized = imresize(face, [64 NaN]); % Resize image
    y1 = BBOX(i,2);
    y2 = y1 + BBOX(i,4);
    y = (y1 + y2)/2; % y location of face
    x1 = BBOX(i,1);
    x2 = x1 + BBOX(i,3);
    x = (x1 + x2)/2; % x location of face

    % Convert input arguments from char to string
    featureType = string(featureType);
    classifierName = string(classifierName);
    
    % Call classifier 
    if (featureType == 'SURF') & (classifierName == 'SVM')
        load('SVM_SURF.mat')
        idx = predict(SVM_SURF, faceResized); %predict index
        id = double(string(SVM_SURF.Labels(idx))); % map index to label
    elseif (featureType == 'HOG') & (classifierName == 'SVM')
        load('SVM_HOG.mat')
        queryFeatures = extractHOGFeatures(faceResized); % extract features 
        id = double(string(predict(SVM_HOG,queryFeatures))); % predict
    elseif (featureType == 'HOG') & (classifierName == 'MLP')
        load('MLP_HOG.mat')
        load('trainingLabel.mat') % load labels
        queryFeatures = extractHOGFeatures(faceResized)'; % extract features
        YPred = MLP_HOG(queryFeatures); % predict
        idx = vec2ind(YPred); % get index
        id = trainingLabel(idx); % map index to label
     elseif (featureType == 'SURF') & (classifierName == 'MLP')
        load('MLP_SURF.mat')
        load('trainingLabel.mat') % load labels
        load('bag.mat') % load bag of features used for model
        faceResized = rgb2gray(faceResized); % change to grayscale
        queryFeatures = encode(bag,faceResized)'; % encode features
        YPred = MLP_SURF(queryFeatures); % predict
        idx = vec2ind(YPred); % get index 
        id = trainingLabel(idx); % map index to label
    elseif classifierName == 'CNN'
        load('CNN.mat')
        imwrite(faceResized,'faceResized.jpeg'); % save image for datastore
        faceds = imageDatastore('faceResized.jpeg'); % load image to datastore
        inputSize = [227,227,3]; % set input size
        augFaceds = augmentedImageDatastore(inputSize(1:2),faceds); % augment image
        YPred = classify(CNN,augFaceds); % predict
        id = double(string(YPred)); % convert to double
    end
    P(i,:)=[id,round(x),round(y)];
end    

end