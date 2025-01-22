function cores = randnInit(sz, rank, scale)
    N = length(sz);
    cores = cell(1,N);
    for d = 1:N
        [~, dn] = round_index(d, N);
        Gi = randn(rank(d), sz(d), rank(dn));
        cores{d} = Gi * scale;
    end
end