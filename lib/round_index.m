function [dp, dn] = round_index(d, D)
    if d == 1
        dp = D;
    else
        dp = d - 1;
    end

    if d == D
        dn = 1;
    else
        dn = d + 1;
    end
end