function cores = randInit(sz, rank, scale)
    N = length(sz);
    cores = cell(1,N);
    for d = 1:N
        [~, dn] = round_index(d, N);
        Gi = rand(rank(d), sz(d), rank(dn));
        cores{d} = Gi * scale;
    end
end