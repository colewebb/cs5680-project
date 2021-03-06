% Automatic seeded region growing for color image segmentation
% Frank Y. Shih, Shouxian Cheng
% doi:10.1016/j.mavis.2005.05.015
% Implemented (mostly) in MATLAB by Cole Webb
%
% This paper, as I mentioned in my presentation, is a data structures
% nightmare. There are sorted lists pretending to be priority queues, graph
% theory connecting image regions, the whole thing is kind of a mess.
%
% As such, and reflecting my own difficulties with MATLAB, OpenCV, and
% Python, I have not entirely succeeded in implementing this algorithm. I
% believe that this has more to do with my misunderstandings of the
% languages  than my misunderstandings of the paper. My attached documents 
% will serve as evidence of this.
%
% Finally, you will note that my first two milestones in this project were
% written in Python and OpenCV. I have attached my Python code to this as
% well, which relies on several libraries noted therein. On the Monday
% after submitting my second milestone, I decided to give MATLAB a try for
% this project and immediately made the same progress in 45 minutes as I
% had made in a week in Python. I therefore decided to move from Python to
% MATLAB. This is entirely my own work, and that can be noted from my
% GitHub commit history, which can be seen at:
%
% https://github.com/colewebb/cs5680-project/commits/master
%
% I hope that you see fit to give this project a passing grade, despite its
% deficiencies.
%
% ~ Cole
%
% ---- Begin -----
% Read the image, convert it to YCbCr
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
% Generate information about individual pixels in the image
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
% Threshold that information
thresholdedSimilarities = imbinarize(similarities, 'global');
thresholdedDistances = imbinarize(distances, 0.05);
% And elementwise AND the information matrices to get our seeded pixels
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
% use dilation to find neighbors of regions
se = strel('diamond', 1);
regionNeighbors = imdilate(seededPixels, se) - seededPixels;
% label connected components in the seeded regions
[regions, regionCount] = bwlabel(seededPixels, 8);
regionStats = computeRegionStats(regions, yIm);
% us a loop to compute neighbors and generate a table of data (T in the
% paper)
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
% pull the smallest distance neighbor off of the stack, merge with the
% closest neighbor, and then add neighbors
while size(neighborTable) > 0
    p = neighborTable(1, :);
    neighborTable = neighborTable(2:end, :);
    row = p(1);
    column = p(2);
    neighbors = findNeighbors(regions, column, row);
    if size(unique(neighbors), 2) == 1 && neighbors(1) ~= 0
        seededPixels(column, row) = neighbors(1);
    elseif size(neighbors, 2) == 1
        seededPixels(column, row) = neighbors(1);
    else
        localNeighborDistances = neighborDistances(neighbors);
        seededPixels(column, row) = neighbors(find(min(localNeighborDistances)));
    end
    neighbors = [[column, row + 1], [column - 1, row], [column + 1, row], [column, row + 1]]; % only grab pixels that exist
    for i = 1:4
        neighborTable(end + 1) = [row, column, distances(row, column)]; % need to check that the pixels aren't in the list already
    end
end
% ----- Component 3, complete -----
% find out how many regions we are working with
regionCount = max(max(seededPixels));
% generate regionCount x regionCount matrix to hold neighbor connections
% using computeNeighbors function (which I wrote, because MATLAB's concept
% of regions is a little bit lost
neighbors = computeNeighbors(seededPixels);
% compute some data about the regions
regionStats = computeRegionStats(seededPixels, yIm);
% set thresholds
sizeThreshold = (rows * columns)/150;
distanceThreshold = 0.1;
% check neighboring regions against the thresholds above, and merge if
% under either threshold to nearest neighbor
for i = 1:regionCount
    regionStats = regionStats(i, :);
    regionNeighbors = find(regions == 1);
    regionNeighbors = regionNeighbors(regionNeighbors(i, :) == 1 || regionNeigbors(:, i) == 1);
    regionNeighbors = unique(regionNeighbors);
    regionDistances = zeros(1, size(regionNeighbors, 2));
    distanceMergeRequired = false;
    for j = 1:size(regionNumbers, 2)
        otherRegionStats = regionStats(j, :);
        numerator = sqrt((regionStats(3) - otherRegionStats(3))^2 + (regionStats(4) - otherRegionStats(4))^2 + (regionStats(5) - otherRegionStats(5))^2);
        denominator = min([sqrt(regionStats(3)^2 + regionStats(4)^2 + regionStats(5)^2), sqrt(otherRegionStats(3)^2 + otherRegionStats(4)^2 + otherRegionStats(5)^2)]);
        if numerator/denominator < distanceThreshold
            distanceMergeRequired = true;
        end
    end
    if distanceMergeRequired
        region1 = i;
        region2 = regionNeighbors(find(min(regionDistances)));
        mergeRegions(seededPixels, region1, region2);
    end
    if regionStats(2) < sizeThreshold
        region1 = i;
        region2 = regionNeighbors(find(min(regionDistances)));
        mergeRegions(seededPixels, region1, region2);
    end
    neighbors = computeNeighbors(seededPixels);
end
% ----- Component 4, complete -----