function [estimate_rank,G,EvG2,A,ASigma,U,indices] = pruneFactorsVI(G,EvG2,A,ASigma,U,indices,sz,rank,trun,method)
    % prune YEst and U and EvG2 (for VI)

    nDim = length(sz);

    pruneIndex = findPruneIndex(A, sz, rank, trun, method);
    estimate_rank = zeros(1,nDim);
    % prune the factors
    for d = 1:nDim
        if sum(pruneIndex{d}) >= 2
            [dp, ~] = round_index(d, nDim);
            estimate_rank(d) = sum(pruneIndex{d});
            G{d} = G{d}(pruneIndex{d}, :, :);
            G{dp} = G{dp}(:, :, pruneIndex{d});
            A{d} = A{d}(pruneIndex{d}, :, :);
            A{dp} = A{dp}(:, :, pruneIndex{d});
            U{d} = U{d}(pruneIndex{d});
            EvG2{d} = EvG2{d}(:, pruneIndex{d}, :, pruneIndex{d}, :);
            EvG2{dp} = EvG2{dp}(:, :, pruneIndex{d}, :, pruneIndex{d});
            ASigma{d} = ASigma{d}(:, :, pruneIndex{d}, :);
            ASigma{dp} = ASigma{dp}(:, :, :, pruneIndex{d});
            indices{d} = indices{d}(:, pruneIndex{d}, :, pruneIndex{d}, :);
            indices{dp} = indices{dp}(:, :, pruneIndex{d}, :, pruneIndex{d});
        else
            [dp, ~] = round_index(d, nDim);
            estimate_rank(d) = rank(d);
            G{d} = G{d};
            G{dp} = G{dp};
            A{d} = A{d};
            A{dp} = A{dp};
            U{d} = U{d};
            EvG2{d} = EvG2{d};
            EvG2{dp} = EvG2{dp};
            ASigma{d} = ASigma{d};
            ASigma{dp} = ASigma{dp};
            indices{d} = indices{d};
            indices{dp} = indices{dp};

        end
    end
end