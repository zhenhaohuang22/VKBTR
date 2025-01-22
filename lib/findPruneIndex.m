function pruneIndex = findPruneIndex(G, sz, rank, trun, method)
    % get the truncation index
    
    nDim = length(sz);
    pruneIndex = cell(nDim, 1);
    
    for d = 1:nDim
        [dp, dn] = round_index(d, nDim);
        Gd = Gunfold(G{d}, 1);
        Gp = Gunfold(G{dp}, 3);
        Gtemp = [Gd, Gp];
        comPower = sqrt(diag(Gtemp * Gtemp'));
        
        if strcmp(method, 'absolute')
            % trun method 1
            comPower = comPower / (rank(dp) * sz(dp) + rank(dn) * sz(d));
            pruneIndex{d} = comPower > trun;
        elseif strcmp(method, 'relative')
            % trun method 2
            trun_ = trun * max(comPower);
            pruneIndex{d} = comPower > trun_;
        end
        
        % test visualize
        % disp(['Mode ', num2str(d), ', ', num2str(comPower)]);
        % if d == 1
        %     bar(comPower);
        % end
    end
    
end
