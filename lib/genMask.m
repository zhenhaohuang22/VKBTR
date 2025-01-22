function mask = genMask(sz, mis)
    % sz: mask size
    % mis: missing ratio
    mask = rand(sz);
    mask(mask >= mis) = 1;
    mask(mask < mis) = 0;
end