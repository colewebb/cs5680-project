function stats = computeRegionStats(regions, im)
    yChannel = im(:, :, 1);
    cbChannel = im(:, :, 2);
    crChannel = im(:, :, 3);
    regionCount = max(max(regions));
    region = zeros(regionCount, 1);
    size = zeros(regionCount, 1);
    yMean = zeros(regionCount, 1);
    cbMean = zeros(regionCount, 1);
    crMean = zeros(regionCount, 1);
    for i = 1:regionCount
        region(i) = i;
        size(i) = sum(regions == i, 'all');
        yMean(i) = mean(yChannel(regions == i));
        cbMean(i) = mean(cbChannel(regions == i));
        crMean(i) = mean(crChannel(regions == i));
    end
    stats = table(region, size, yMean, cbMean, crMean);
end