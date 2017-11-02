%% Setup, load data

% training sequence params
epochs  = 100;
batch   = 200;

% optimization params
alpha   = 0.001;
beta1   = 0.9;
beta2   = 0.999;
epsilon = 1e-8;

% regularization params
lambda  = 0.001;

% topology
size_input = 28*28;
size_hidden = 200;
size_latent = 2;

W_encoder = randn(size_input, size_hidden) / sqrt(size_input);
b_encoder = rand(size_hidden, 1);
W_mu = randn(size_hidden, size_latent) / sqrt(size_hidden);
b_mu = rand(size_latent, 1);
W_logvar = randn(size_hidden, size_latent) / sqrt(size_hidden);
b_logvar = rand(size_latent, 1);
W_decoder = randn(size_latent, size_hidden) / sqrt(size_latent);
b_decoder = rand(size_hidden, 1);
W_y = randn(size_hidden, size_input) / sqrt(size_hidden);
b_y = rand(size_input, 1);

Theta = packParams(W_encoder, b_encoder, W_mu, b_mu, W_logvar, b_logvar, ...
                   W_decoder, b_decoder, W_y, b_y);

% ADAM state
Theta_m = zeros(size(Theta));
Theta_v = zeros(size(Theta));
t = 1;


addpath('..');
[X, Y] = loadMNIST('fashion-mnist/t10k-images-dx3-ubyte', ...
                   'fashion-mnist/t10k-labels-idx1-ubyte');
N = size(X, 3);
X = reshape(X, [], N);

%% Training
batches = N / batch;
E = zeros(epochs * batches, 1);
for epoch = 1:epochs
    e_start = (epoch-1)*batches;
    for i = 1:batch:N
        % Mini batch SGD
        X_batch = X(:, i:i+batch-1);
        [~, J, grad] = vaeRun(X_batch, Theta, size_input, size_hidden, ... 
                              size_latent, lambda);
        [Theta, Theta_m, Theta_v] = adam(alpha, beta1, beta2, epsilon, t, ...
                                         Theta, grad, Theta_m, Theta_v);
        E(e_start+i) = J;
    end
    
    e_end = e_start + batch;
    fprintf('[%3d] J = %f\n', epoch, mean(E(e_start+1:e_end)));
end

%% Try it out
grid_size = 20;
v = linspace(-3, 3, grid_size);
[p,q] = meshgrid(v, v);
Z = permute([p(:) q(:)], [2, 1]);
X_d = vaeDecode(Z, Theta, size_input, size_hidden, size_latent);

img = zeros(28 * grid_size);
X_img = reshape(X_d, 28, 28, grid_size, grid_size);
for i = 1:grid_size
    yi = (i-1)*28+1;
    for j = 1:grid_size
        xi = (j-1)*28+1;
        img(yi:yi+27, xi:xi+27) = X_img(:,:,i,j);
    end
end
imagesc(v,v,img);
colormap bone