function [id] = detectNum(filename)
% detectNum number detection function
% Student: Lesley Dwyer
% Built with Matlab version R2018b

% detectNum takes an image or video as input. It uses MSER to detect 
% candidate regions of text, then uses geometric properties and stroke 
% width variation to determine regions to remove. It then runs OCR on 
% the remaining regions and returns the digit found in the region(s) with 
% confidence higher than 75%.

% Determine if file is video or image
isMovie = strfind(filename,'.mov');
isMovie2 = strfind(filename,'.mp4');
if isMovie > 0 
    % If .mov video, extract image here
    videoReader = VideoReader(filename);
    images = read(videoReader);
    I = images(:,:,:,10);
elseif isMovie2 > 0
    % If .mp4 video, extract image here
    videoReader = VideoReader(filename);
    images = read(videoReader);
    I = images(:,:,:,10);
else
    % If image
    I = imread(filename);
end

% If image is rotated sideways, rotate it
if size(I,2)>size(I,1)
    I = imrotate(I,270);
end
I = rgb2gray(I);

% The code below was adapted from this example from Mathworks
% https://uk.mathworks.com/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html

% Detect MSER candidate regions of text
[mserRegions, mserConnComp] = detectMSERFeatures(I, ... 
'RegionAreaRange',[200 8000],'ThresholdDelta',4);

% Use regionprops to measure MSER geometric properties
mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
'Solidity', 'Extent', 'Euler', 'Image');

% Compute the aspect ratio using bounding box data
bbox = vertcat(mserStats.BoundingBox);
w = bbox(:,3);
h = bbox(:,4);
aspectRatio = w./h;

% Threshold the data to determine which regions to remove
filterIdx = aspectRatio' > 3; 
filterIdx = filterIdx | [mserStats.Eccentricity] > .995 ;
filterIdx = filterIdx | [mserStats.Solidity] < .3;
filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
filterIdx = filterIdx | [mserStats.EulerNumber] < -4;

% Remove regions
mserStats(filterIdx) = [];
mserRegions(filterIdx) = [];

% Next, use stroke width variation to find regions to remove
for j = 1:numel(mserStats)
        
    regionImage = mserStats(j).Image;
    regionImage = padarray(regionImage, [1 1],0);
        
    distanceImage = bwdist(~regionImage);
    skeletonImage = bwmorph(regionImage, 'thin', inf);
        
    strokeWidthValues = distanceImage(skeletonImage);
    strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
    strokeWidthThreshold = 0.55; 
    strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;
        
end
    
% Remove regions based on the stroke width variation
mserRegions(strokeWidthFilterIdx) = [];
mserStats(strokeWidthFilterIdx) = [];
strokeWidthFilterIdx(:,:)=[];

% Get bounding boxes for all the regions
bboxes = vertcat(mserStats.BoundingBox);

% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax] format for convenience
xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

% Expand the bounding boxes by a small amount
expansionAmount = 0.01;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;

% Clip the bounding boxes to be within the image bounds
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));

% Compute the expanded bounding boxes
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

% Compute the overlap ratio
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

% Set the overlap ratio between a bounding box and itself to zero to
% simplify the graph representation.
n = size(overlapRatio,1); 
overlapRatio(1:n+1:n^2) = 0;

% Create the graph
g = graph(overlapRatio);

% Find the connected text regions within the graph
componentIndices = conncomp(g);

% Merge the boxes based on the minimum and maximum dimensions.
xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

% Compose the merged bounding boxes using the [x y width height] format.
textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

% Recognise Text using OCR and return text with confidence > 75%
ocrtxt = ocr(I, textBBoxes,'CharacterSet','0123456789','TextLayout', 'Line');
for i=1:size(textBBoxes,1)
    confidenceHigh = 0;
    for j=1:size(ocrtxt(i,1).CharacterConfidences,1)
        if ocrtxt(i,1).CharacterConfidences(j,1) > 0.75
            confidenceHigh = 1;
        end
    end
    if confidenceHigh == 1
        ocrtxt(i,1).Text
    end
end
end

