function distance = relativeEuclideanDistance(image, column, row)
    % returns the maximum relative distance between a given pixel in a an
    % image and its 8-neighbors, assuming a 3-channel image
    center = image(column, row);
    neighbors = [image(column - 1:column + 1, row - 1),
        image(column - 1, row),
        image(column + 1, row),
        image(column - 1:column + 1, row + 1)];
    distances = zeros(1, 8);
    for i = 1:8
        numerator = sqrt(sum((center - neighbors(i))^2));
        denominator = sqrt(sum(center^2));
        distances(i) = numerator/denominator;
    end
    distance = max(distances);
end