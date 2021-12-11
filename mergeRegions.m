function regions = mergeRegions(regions, region1, region2)
    if region1 > region2
        regions(regions == region1) = region2;
        regions(regions > region1) = regions(regions > region1) - 1;
    else
        regions(regions == region2) = region1;
        regions(regions > region2) = regions(regions > regions2) - 1;
    end
end