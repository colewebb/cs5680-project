% Automatic seeded region growing for color image segmentation
% Frank Y. Shih, Shouxian Cheng
% doi:10.1016/j.mavis.2005.05.015
% Implemented in Python by Cole Webb
% ---- Begin -----
im = imread("./pictures/img.png");
yIm = rgb2ycbcr(im);
% ----- Component 1, complete -----
subplot(1,2,1);
imshow(im);
title("Original image");
subplot(1,2,2);
imshow(yIm);
title("YCbCr image, false color");
disp("Press any key to continue...")
pause;
close all;
% ----- Displaying Component 1, complete -----
columns = size(yIm, 1);
rows = size(yIm, 2);
stdDevs = stdfilt(yIm(:, :, 1)) + stdfilt(yIm(:, :, 2)) + stdfilt(yIm(:, :, 3));
stdDevs = stdDevs./max(max(stdDevs));
similarities = 1.-stdDevs;
distances = zeros(columns, rows);
for column = 2:columns - 1
    for row = 2:rows - 1
        distances(column, row) = relativeEuclideanDistance(yIm, column, row);
    end
end
similarities = imbinarize(similarities, 'global');
distances = imbinarize(distances, 0.05);
seededPixels = similarities & distances;
% ----- Component 2, complete -----
subplot(1,3,1);
imshow(similarities);
title("Thresholded similarities");
subplot(1,3,2);
imshow(distances);
title("Thresholded distances");
subplot(1,3,3);
imshow(seededPixels);
title("Seeded pixels (similarities & distances)");
disp("Press any key to continue...")
pause;
close all;
% ----- Displaying Component 2, complete -----


