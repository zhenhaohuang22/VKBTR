function out = expectationTR2(EvG2, mask)
    N = ndims(mask);
    temp = 0;
    for n = 1:N-1
        if n == 1
            temp = einsum(EvG2{1},EvG2{2},[3,5],[2,4]);
        elseif n == N-1
            out = einsum(temp,EvG2{N},[2,3,3+n,3+n+1],[3,5,2,4]);
        else
            temp = einsum(temp,EvG2{n+1},[n+3,n+4],[2,4]);
        end
    end
    mask_out = mask .* out;
    out = sum(mask_out(:));
end
