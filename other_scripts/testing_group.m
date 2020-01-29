% Short script to help with testing group images
test = zeros(61,3); % create matrix, change number rows to max number faces in image 
I = 'IMG_8237.jpg'; % change image to test image
test = RecogniseFace(I,'SURF','SVM');

% Plot the predictions on the faces
figure;
imshow(I); 
axis on;
hold on;
for i = 1:61 % change to max number faces in image
    xBeg = test(i,2)/4032 - 0.02; % adjust to unit scaling
    yBeg = 1 - test(i,3)/3024 + 0.04; % adjust to unit scaling
    marker = string(test(i,1)); % mark with person's label
    annotation('textbox','Units','normalized','Position',[xBeg yBeg 0.005 0.005],...
        'String',marker,'Color','green','EdgeColor','green',...
        'FitBoxToText','on');
end