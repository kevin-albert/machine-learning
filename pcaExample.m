%% Initialization
m = 500;        % training data count
n = 100;        % number of input features
X = rand(m,n);  % bogus data
target = 0.95;  % target retained variance

%% Zero-mean normalization and feature scaling
mu = mean(X);
sigma = std(X);
X = (X - mu) ./ sigma;

%% Algorithm

% Covariance Matrix
Sigma = 1/m * (X' * X);

% Compute eigenvectors
[U,S,V] = svd(Sigma);


%% Find number of components for desired variance
S_diag = diag(S);
S_trace = trace(S);

% find the smallest number of principal components that meets our target
% retained variance
k = 0;
variance = 0;
while variance < target
    k = k + 1;
    variance = sum(S_diag(1:k)) / S_trace;
end

% Change of basis matrix
P = U(:, 1:k);

%% Compress a high dimensional input datum
x = rand(n,1);
x = (x - mu') ./ sigma';
z = P' * x;

%% Decompress a low dimensional output datum
x_ = P * z;
