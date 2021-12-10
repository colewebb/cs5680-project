% Automatic seeded region growing for color image segmentation
% Frank Y. Shih, Shouxian Cheng
% doi:10.1016/j.mavis.2005.05.015
% Implemented in MATLAB by Cole Webb
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
thresholdedSimilarities = imbinarize(similarities, 'global');
thresholdedDistances = imbinarize(distances, 0.05);
seededPixels = thresholdedSimilarities & thresholdedDistances;
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
se = strel('diamond', 1);
regionNeighbors = imdilate(seededPixels, se) - seededPixels;
[regions, regionCount] = bwlabel(seededPixels, 8);
regionStats = computeRegionStats(regions, yIm);
% imshow(label2rgb(regions));
% pause;
rows = size(im, 1);
columns = size(im, 2);
neighborTable = [];
for row = 2:rows - 1
    for column = 2:columns - 1
        neighbors = findNeighbors(regions, column, row);
        if min(neighbors) > 0
            neighbors = neighbors(neighbors > 0);
            neighborDistances = [];
            for neighbor = 1:size(regionStats, 1)
                neighborStats = regionStats(neighbor, :);
                pixelValues = double(yIm(row, column, :));
                numerator = sqrt((pixelValues(1) - neighborStats.yMean)^2 + (pixelValues(2) - neighborStats.cbMean)^2 + (pixelValues(3) - neighborStats.crMean)^2);
                denominator = sqrt(pixelValues(1) ^ 2 + pixelValues(2)^2 + pixelValues(3)^2);
                neighborDistances(neighbor, :) = [neighbor, numerator/denominator];
            end
            neighborTable(end + 1) = [row, column, min(neighborDistances)];
        end
    end
end
neighborTable = sortrows(neighborTable, 3);
pause;
while size(neighborTable) > 0
    p = neighborTable(1, :);
    neighborTable = neighborTable(2, :);
    row = p(1);
    column = p(2);
    neighbors = [seededPixels(column, row - 1),
        seededPixels(column - 1, row),
        seededPixels(column + 1, row),
        seededPixels(column, row + 1)];
    if (neighbors(1) == neighbors(2) == neighbors(3) == neighbors(4)) && neighbors(1) ~= 0
        seededPixels(column, row) = neighbors(1);
    end
end