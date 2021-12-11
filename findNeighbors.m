function neighbors = findNeighbors(regions, column, row)
    % find values of the 4-neighbors of the pixel at [column, row] in
    % regions
    % this really should only return neighbors that actually exist, but i
    % haven't gotten that far yet and might not get there at all
    neighbors = [regions(column, row + 1), regions(column - 1, row), regions(column + 1, row), regions(column, row + 1)];
    neighbors = unique(neighbors);
end