function [y_pred, iter, alphas, D, objective] = MJP(Ks, k, gamma, beta, err, max_iter)
% Input Parameters:
%   Ks: kernels, m * 1 or 1 * m cell_matrix, each element should be n * n
%   k: number of clusters
%   gamma: hyper-parameter of SoftMax
%   beta: trade-off parameter 
%   err: threshold of residuals of objective functions to stop, 10^-7 as default
%   max_iter: maximum_iteration, 10 as default

% Output Parameters:
%   y_pred: predicted labels
%   iter: iterations to converge
%   alphas: weights of m views
%   D: learning parameters / covariance matrix
%   objective: values of the loss 


if nargin == 4
    err = 10^-7;
    max_iter = 10;
elseif nargin == 5
    max_iter = 10;
end

shape = size(Ks{1});
m = length(Ks);
n = shape(1);
P = rand(n, k);
alphas = ones(m, 1) / m;
d = zeros(m, 1);
D = zeros(n, n);
objective = zeros(max_iter, 1);
epsilon = 10^-3;
gamma_base = -1;

% calcuate total_K
K = cal_K(Ks, alphas);
for i = 1:max_iter
    iter = i;

    % update D
    K = max(K, K');
    K = (K + K') / 2;
    [U, S] = eig(K + epsilon * eye(n));
    D = sum(diag(S).^0.5) * (U .* ((diag(S)' + epsilon).^-0.5) * U');

    % calculate eigen-vector & update P
    % [P, ~] = eigs(K, k);
    P = U(:, n-k+1:n);


    % update alpha
    for j = 1:m 
        t = trace(Ks{j}) - trace(P' * Ks{j} * P);
        l = trace(D * Ks{j});
        d(j) = t + beta * l;
    end
    if gamma_base == -1
        gamma_base = mean(d);
    end
    d = d / (gamma * gamma_base);
    % Equivalent operation, avoid numeraical overflow
    d = d - min(d);
    alphas = exp(-d);
    alphas = alphas / sum(alphas);
    % calcuate total_K
    K = cal_K(Ks, alphas);
    objective(i, 1) = trace(K * (eye(n) - P * P')) + gamma * sum(alphas .* log(alphas + 10^-20)) + beta * trace(D * K);
    if i == 1
        continue
    end
    if abs(objective(i,1) - objective(i-1, 1)) <= err
        objective = objective(1:i);
        break
    end
end
P = P ./ repmat(sqrt(sum(P.^2, 2)), 1, k);
y_pred = kmeans(P, k, 'MaxIter', 1000, 'Replicates', 10);
end

function K = cal_K(Ks, alphas)
    K = 0;
    m = length(Ks);
    for i = 1: m
        K = K + alphas(i) * Ks{i};
    end
end
