function model = VKBTR(Y,mask,rank,K,opt)

Y = Y .* mask;
nObs = sum(mask(:));
nDim = ndims(Y);
sz = size(Y);

a_u = 1e-6;
b_u = 1e-6;
a_tau = 1e-6;
b_tau = 1e-6; 
tau = 1;
% init
init = opt.init;
switch init
    case 'randn'
        G = randnInit(sz, rank, opt.initScale);
        A = randnInit(sz, rank, opt.initScale);
    case 'rand'
        G = randInit(sz, rank, opt.initScale);
        A = randInit(sz, rank, opt.initScale);
end
U = cell(1, nDim);
EvG2 = cell(1, nDim);
ASigma = cell(1, nDim);
indices = cell(1, nDim);
sigma_g = cell(1, nDim);
sigma_g_alpha = cell(1, nDim);
sigma_g_beta = cell(1, nDim);
a_u_post = cell(1, nDim);
b_u_post = cell(1, nDim);
for d = 1:nDim
    [~, dn] = round_index(d, nDim);
    U{d} = ones(rank(d), 1);
    EvG2_d = zeros(sz(d), rank(d), rank(dn), rank(d), rank(dn));
    for i = 1:sz(d)
        mu = squeeze(G{d}(:, i, :));
        EvG2Mean = ncon({mu,mu},{[-1,-2],[-3,-4]});
        cov_d = eye(rank(d) * rank(dn),rank(d) * rank(dn));
        EvG2_d(i,:,:,:,:) = EvG2Mean + reshape(cov_d, rank(d), rank(dn), rank(d), rank(dn));
    end
    EvG2{d} = EvG2_d;
    ASigma_d = repmat(eye(sz(d), sz(d)), [1, 1, rank(d)*rank(dn)]);  
    ASigma{d} = reshape(ASigma_d, sz(d), sz(d), rank(d), rank(dn));
    indices_d = repmat(logical(eye(rank(d) * rank(dn))), [sz(d),1,1]);  
    indices{d} = reshape(indices_d, sz(d), rank(d), rank(dn), rank(d), rank(dn));
end
XOld = coreten2tr(G);
LB = zeros(opt.iter1);
for epoch = 1:opt.iter1
    
    for d = 1:nDim
        [~, dn] = round_index(d, nDim);
        sigma_g_alpha{d} = sz(d)*rank(d)*rank(dn)/2;
        Gd = Gunfold(G{d},2);
        Ad = Gunfold(A{d},2);
        sigma_g_beta{d} = trace((Gd-K{d}*Ad)*(Gd-K{d}*Ad)')+...
            sum(EvG2{d}(indices{d}));
        for r1 = 1:rank(d)
            for r2 = 1:rank(dn)
                sigma_g_beta{d} = sigma_g_beta{d}+trace(K{d}*ASigma{d}(:,:,r1,r2)*K{d}');
            end
        end
        sigma_g_beta{d} = sigma_g_beta{d}/2;    
        sigma_g{d} = sigma_g_alpha{d}/sigma_g_beta{d};
        
    end
    
    for d = 1:nDim
        [dp, dn] = round_index(d, nDim);
        a_u_post{d} = (a_u + 0.5 * (sz(dp) * rank(dp) + sz(d) * rank(dn))) * ones(rank(d), 1);
        
        b_u_post{d} = zeros(rank(d), 1);
        foo = zeros(rank(dp), 1);  % first part
        bar = zeros(rank(dn), 1);  % second part
        for rd = 1:rank(d)          
            for r1 = 1:rank(dp)
                foo(r1) = squeeze(A{dp}(r1,:,rd)) * squeeze(A{dp}(r1,:,rd))' + sum(diag(ASigma{dp}(:,:,r1,rd)));
            end        
            for r2 = 1:rank(dn)
                bar(r2) = squeeze(A{d}(rd,:,r2)) * squeeze(A{d}(rd,:,r2))' + sum(diag(ASigma{d}(:,:,rd,r2)));
            end
            b_u_post{d}(rd) = b_u + 0.5 * (U{dp}' * foo + U{dn}' * bar);
        end
        U{d} = a_u_post{d} ./ b_u_post{d};
    end
    
    for d = 1:nDim
        [~, dn] = round_index(d, nDim);
        for r1 = 1:rank(d)
            for r2 = 1:rank(dn)
                ASigma{d}(:,:,r1,r2) = (K{d}'*K{d}*sigma_g{d} + U{d}(r1)*U{dn}(r2)*eye(sz(d))) \ eye(sz(d), sz(d));
                A{d}(r1,:,r2) = ASigma{d}(:, :, r1,r2) * K{d}' * G{d}(r1,:,r2)' * sigma_g{d};
            end
        end
    end
    
    for d = 1:nDim
        [~, dn] = round_index(d, nDim);
        
        Yd = tenmat_sb(Y, d);
        Ad = Gunfold(A{d}, 2);
        Gd = zeros(rank(d), sz(d), rank(dn));  % container for core d
        gNeq = Z_neq(G, d);
        gNeqVec = reshape(permute(gNeq, [2, 3, 1]), size(gNeq, 2), []);
        skipChainExp = expectationSubchain2(EvG2, d, mask);
        for i = 1:sz(d)
            Gamma = tau * Yd(i, :) * gNeqVec + sigma_g{d} * K{d}(i,:) * Ad;
            V = (tau * squeeze(skipChainExp(i,:,:))  + eye(rank(d)*rank(dn)) * sigma_g{d}) \ eye(rank(d)*rank(dn));
            mu = V * Gamma';
            Gdi = reshape(mu, rank(d), rank(dn));
            Gd(:, i, :) = Gdi;
            EvG2Mean = ncon({Gdi, Gdi}, {[-1, -2], [-3, -4]});
            EvG2{d}(i,:,:,:,:) = EvG2Mean + reshape(V, rank(d), rank(dn), rank(d), rank(dn));
        end
        G{d} = Gd;        
    end
    
    % update estimates
    XNew = coreten2tr(G);
    
    % update noise level
    a_tau_post = a_tau + 0.5 * nObs;
    idx = find(mask==1);
    eTR = Y(idx)'*Y(idx) - 2 * Y(idx)'* XNew(idx);
    eTR = eTR + expectationTR2(EvG2, mask);
    b_tau_post = b_tau + 0.5 * eTR;
    tau = a_tau_post / b_tau_post;
    
    %% Lower bound
    if opt.isELBO
        temp1 = -0.5*nObs*safelog(2*pi) + 0.5*nObs*(psi(a_tau_post)-safelog(b_tau_post)) - 0.5*(a_tau_post/b_tau_post)*eTR;
        temp2 = 0;
        for d=1:nDim
            [~, dn] = round_index(d, nDim);
            temp2= temp2 + (-0.5*rank(d)*rank(dn)*sz(d)*safelog(2*pi) + 0.5*rank(d)*rank(dn)*sz(d)*(psi(sigma_g_alpha{d})-safelog(sigma_g_beta{d})-sigma_g_alpha{d}));
        end
        temp3 = 0;
        for d=1:nDim
            [~, dn] = round_index(d, nDim);
            temp3 = temp3 + (-0.5*rank(d)*rank(dn)*sz(d)*safelog(2*pi) + 0.5*sz(d)*(sum(psi(a_u_post{d})-safelog(b_u_post{d}))+ sum(psi(a_u_post{dn})-safelog(b_u_post{dn}))));    
            for r1=1:rank(d)
                for r2 = 1:rank(dn)
                    temp3 = temp3 + (-0.5* ((a_u_post{d}(r1)*a_u_post{dn}(r2))/(b_u_post{d}(r1)*b_u_post{dn}(r2)))*(squeeze(A{d}(r1,:,r2))*squeeze(A{d}(r1,:,r2))' + trace(ASigma{d}(:,:,r1,r2))));
                end
            end
        end
        temp4 = 0;
        for d=1:nDim
            temp4 = temp4 + sum(-safelog(gamma(a_u)) + a_u*safelog(b_u) -  b_u.*(a_u_post{d}./b_u_post{d}) + (a_u-1).*(psi(a_u_post{d})-safelog(b_u_post{d})));
        end
        temp5 = 0;
        for d=1:nDim
            temp5 = temp5 + (-psi(sigma_g_alpha{d})+safelog(sigma_g_beta{d}));
        end
        temp6 = -safelog(gamma(a_tau)) + a_tau*safelog(b_tau) + (a_tau-1)*(psi(a_tau_post)-safelog(b_tau_post)) - b_tau*(a_tau_post/b_tau_post);
        temp7=0;
        for d=1:nDim
            [~, dn] = round_index(d, nDim);
            temp7 = temp7 +  + 0.5*rank(d)*rank(dn)*sz(d)*(1+safelog(2*pi));
            for i=1:sz(d)
                temp7 = temp7 + 0.5*safelog(det(reshape(EvG2{d}(i,:,:,:,:),rank(d)*rank(dn),rank(d)*rank(dn))));
            end
        end
        temp8=0;
        for d=1:nDim
            [~, dn] = round_index(d, nDim);
            temp8 = temp8 + 0.5*rank(d)*rank(dn)*sz(d)*(1+safelog(2*pi));
            for r1=1:rank(d)
                for r2=1:rank(dn)
                    temp8 = temp8 + 0.5*safelog(det(ASigma{d}(:,:,r1,r2)));
                end
            end
        end
        temp9 = 0;
        for d=1:nDim
            temp9 = temp9 + sum(safelog(gamma(a_u_post{d})) - (a_u_post{d}-1).*psi(a_u_post{d}) -safelog(b_u_post{d}) + a_u_post{d});
        end
        temp10 = 0;
        for d=1:nDim
            temp10 = temp10 + sum(safelog(gamma(sigma_g_alpha{d})) - (sigma_g_alpha{d}-1).*psi(sigma_g_alpha{d}) -safelog(sigma_g_beta{d}) + sigma_g_alpha{d});
        end
        temp11 = safelog(gamma(a_tau_post)) - (a_tau_post-1)*psi(a_tau_post) -safelog(b_tau_post) + a_tau_post;
        LB(epoch) = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8 + temp9 + temp10 + temp11;
        % truncate factors
    end
    if opt.isPrune
        [estimate_rank,G,EvG2,A,ASigma,U,indices] = pruneFactorsVI(G, EvG2, A, ASigma, U, indices, sz, rank, opt.trun, opt.pruneMethod);
        rank = estimate_rank;
    end
    
    diff = norm(XNew(:)-XOld(:),'fro')/norm(XOld(:),'fro');
    fprintf('iter=%d,diff=%f,LB=%f,rank=[%s].\n',epoch,diff,LB(epoch),num2str(rank));
    XOld = XNew;
    % Convergence check
    if diff < opt.tol
        break;
    end
end


X = coreten2tr(G);

model.X = X;
model.G = G;
model.tau = tau;
model.rank = rank;
model.LB = LB;


function y = safelog(x)
x(x<1e-300)=1e-200;
x(x>1e300)=1e300;
y=log(x);