function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
Cold = C;

for kiter = 1:intermax
    % CHANGE
    % Step 1: Assign to clusters
    y = step_assign_cluster(X, Cold);
    
    % Step 2: Assign new clusters
    C = step_compute_mean(y, X);
        
    if fcdist(C,Cold) < conv_tol
        return 
    end
    Cold = C;
    % DO NOT CHANGE
end

function y = step_assign_cluster(X, Cold)
    y = zeros(N,1);
    for i = 1:N
        d = fxdist(X(:,i), Cold);
        [~, y(i)] = min(d);
    end
end

function C = step_compute_mean(y, X)
    C = zeros(D, K);
    for k = 1:K
        N_k = 0;
        sum = 0;
        for i = 1:N
            if y(i) == k
                N_k = N_k + 1;
                sum = sum + X(:,i);
            end
        end
        C(:,k) = N_k^(-1)*sum;
    end
end

end


function d = fxdist(x,C)
    % CHANGE
    d = vecnorm(C-x);
    % DO NOT CHANGE
end

function d = fcdist(C1,C2)
    % CHANGE
    d = norm(C1-C2);
    % DO NOT CHANGE
end