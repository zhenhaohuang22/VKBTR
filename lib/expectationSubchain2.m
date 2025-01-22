function out = expectationSubchain2(EvG2, d, mask)
    
    

    N = ndims(mask);
    EvG2Neq = EvG2([d+1:N,1:d-1]);
    perIdx = circshift(1:N,-(d-1));
    mask = permute(mask,perIdx);
    temp = 0;
    for n = 1:N-1
        if n == 1
            temp = einsum(EvG2Neq{1},EvG2Neq{2},[3,5],[2,4]);
        elseif n == N-1
            idx = [4:4+N-3];
            out = einsum(mask,temp,[2:N],[1,idx]);
        else
            temp = einsum(temp,EvG2Neq{n+1},[n+3,n+4],[2,4]);
        end
    end
    out = permute(out,[1,4,2,5,3]);
    szNew = size(EvG2{d}, 2) * size(EvG2{d}, 3);
    out = reshape(out, [], szNew, szNew);
end
