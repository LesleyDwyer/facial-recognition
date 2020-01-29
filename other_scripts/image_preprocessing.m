% Computer Vision Coursework - Face Recognition and OCR
% Student: Lesley Dwyer
% Image Pre-processing

% This code creates the image training database for face recognition and
% OCR. It extracts 10 frames from each video, detects faces in all images,
% crops and resizes them and saves them to a new set of folders. It assumes
% the original images are all stored in labelled sub-directories in one 
% main directory.

% BEFORE RUNNING: Save this file to the directory above the directory where 
% your images are stored. Update the imageDir variable to the directory 
% with the images and the numFolders to the number of image folders. 

clc;
clear all;
tic;

% First, extract images from video files and save back to same directory

rootDir = what; % Get current path
imageDir = 'imageDatabase'; % CHANGE to directory where images are stored.
% This should be a sub-directory of rootDir.
numFolders = 69; % CHANGE to number of image folders

% First convert .mov files
entireDir = dir(imageDir); % entire directory contents
dirCount = length(entireDir([entireDir.isdir])); % find number of folders to loop through
for i = 1:dirCount
    cd(entireDir(i).folder);
    videoFiles = dir([entireDir(i).name,'\*.mov']); % list video files
    videoCount = length(videoFiles); % get count of video files in current folder
    if videoCount > 0
        cd(videoFiles(1).folder);
    end
    for j = 1:videoCount
        videoReader = VideoReader(videoFiles(j).name);
        images = read(videoReader);
        for k = 1:10 % extract 10 frames from video
            I = images(:,:,:,k);
            videoNum = num2str(j);
            num = num2str(k);
            name = strcat('IMG_',num,'_fromMov_', videoNum);
            ext = '.jpeg';
            filename = strcat(name, ext) ;
            imwrite(I,filename);
        end
    end
end

% Do the same for .mp4 files
cd(entireDir(1).folder);
for i = 1:dirCount
    cd(entireDir(i).folder);
    videoFiles = dir([entireDir(i).name,'\*.mp4']); % list video files
    videoCount = length(videoFiles); % get count of video files in current folder
    if videoCount > 0
        cd(videoFiles(1).folder);
    end
    for j = 1:videoCount
        videoReader = VideoReader(videoFiles(j).name);
        images = read(videoReader);
        for k = 1:10 % extract 10 frames from video
            I = images(:,:,:,k);
            videoNum = num2str(j);
            num = num2str(k);
            name = strcat('IMG_',num,'_fromMov_', videoNum);
            ext = '.jpeg';
            filename = strcat(name, ext) ;
            imwrite(I,filename);
        end
    end
end

%cd('C:\Users\Lesley\Documents\Data Science\City Data Science MSc\Computer Vis INM460\Coursework');
cd(rootDir.path); % Change directory back to root folder
mkdir trainingDatabase;

% Load images from sub-directories into Matlab
faceDB = imageSet(imageDir,'recursive');
trainingLabels={faceDB.Description}; % Assign all folder names to folderNames

% Detect faces and crop images to specific size
% This code was adapted from the Mathworks face detection example:
% https://uk.mathworks.com/help/vision/ref/vision.cascadeobjectdetector-system-object.html
% It creates new labelled folders. It then loops through each old folder and 
% each image. It crops and resizes each image and saves it to the newly 
% created labelled folder

for i = 1:numFolders % loop through all folders
    numImages = faceDB(i).Count; % number of images in current folder
    if numImages > 0 
        folderId = cell2mat(trainingLabels(i)); % Return name of folder for id
        newDir = strcat('trainingDatabase/', num2str(folderId));
        mkdir(newDir); % create new labelled folder to save edited images 
        for j = 1:numImages % loop through all images
            A=read(faceDB(i),j);
            myFaceDetector = vision.CascadeObjectDetector("MergeThreshold",6); % Use face detector to detect faces
            BBOX = myFaceDetector(A); 
            N = size(BBOX,1); % Count number of faces
            if N == 0 % If no faces are found, try rotating image
                A = imrotate(A,270);
                myFaceDetector = vision.CascadeObjectDetector("MergeThreshold",6);
                BBOX = myFaceDetector(A); 
                N = size(BBOX,1); % Count number of faces
            end
            for k = 1:N % loop through all faces
                face = imcrop(A,BBOX(k,:));
                faceResized = imresize(face, [64 NaN]); % Resize images to all the same size
                name = 'trainingImage_';
                faceNum = num2str(k);
                imageNum = num2str(j);
                ext = '.jpeg';
                path = strcat('trainingDatabase/', num2str(folderId),'/');
                filename = strcat(path, name, '_', num2str(folderId), '_', imageNum, '_',faceNum, ext);
                imwrite(faceResized,filename);
            end    
        end
    end
end

toc;

% NOW MANUALLY REMOVE BAD IMAGES FROM FOLDERS
