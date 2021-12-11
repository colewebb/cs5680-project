function neighbors = computeNeighbors(regions)
    regionCount = max(max(regions));
    neighbors = zeros(regionCount);
    rows = size(regions, 1);
    columns = size(regions, 2);
    for row = 2:rows - 1
        for column = 2:columns - 1
            center = seededPixels(column, row);
            localNeighbors = [seededPixels(column, row + 1), seededPixels(column - 1, row), seededPixels(column + 1, row), seededPixels(column, row + 1)];
            localNeighbors = unique(localNeighbors);
            for i = 1:size(localNeighbors, 2)
                if neighbors(localNeighbors(i), center) == 0
                    neighbors(center, localNeighbors(i)) = 1;
                end
            end
        end
    end
end