function neighbors = findNeighbors(regions, column, row)
    neighbors = [regions(column, row + 1), regions(column - 1, row), regions(column + 1, row), regions(column, row + 1)];
    neighbors = unique(neighbors);
end