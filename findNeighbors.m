function neighbors = findNeighbors(regions, column, row)
    % find values of neighbors of a given pixel in a 1-channel image
    % this really should only return neighbors that actually exist, but i
    % haven't gotten that far yet and might not get there at all
    neighbors = [regions(column, row + 1), regions(column - 1, row), regions(column + 1, row), regions(column, row + 1)];
    neighbors = unique(neighbors);
end