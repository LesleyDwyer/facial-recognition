% Short script to show a picture of each face and matching number
I = imageSet('trainingDatabase','recursive');
for i=1:69
    Label(1,i) = string(I(i).Description);
end
for j = 1:69
    queryImage = read(I(j),1);
    subplot(7,10,j);imshow(queryImage);
    heading = Label(1,j);
    title(heading);
end