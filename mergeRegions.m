function regions = mergeRegions(regions, region1, region2)
    % takes a labeled regions matrix and two region numbers
    % returns a new labeled regions matrix, but with fewer regions in it
    % if region2 is the lower-numbered region
    if region1 > region2
        % set region1 to region2
        regions(regions == region1) = region2;
        % and decrement any regions numbered higher than region1
        regions(regions > region1) = regions(regions > region1) - 1;
    % otherwise
    else
        % do the exact same thing, just flip the regions
        regions(regions == region2) = region1;
        regions(regions > region2) = regions(regions > regions2) - 1;
    end
end