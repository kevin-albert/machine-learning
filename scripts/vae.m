%% Setup, load data
addpath('..');

% training sequence params
epochs  = 50;
batch   = 200;

% optimization params
alpha   = 0.003;
beta1   = 0.9;
beta2   = 0.999;
epsilon = 1e-8;

% regularization params
lambda  = 0.001;

% topology
size_input = 28*28;
size_hidden = 200;
size_latent = 3;

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
[X, Y] = loadMNIST('fashion-mnist/train-images-idx3-ubyte', ...
                   'fashion-mnist/train-labels-idx1-ubyte');
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

%% Try out decoder

k = 0;
grid_size = 10;
images = zeros(28*grid_size, 28*grid_size, grid_size);
v = linspace(-1, 1, grid_size);
[p,q] = meshgrid(v, v);
    
for z = linspace(-1, 1, grid_size)
    
    k = k + 1;
    Z = [ permute([p(:) q(:)], [2, 1]);
          ones(1, grid_size.^2)*z ];
    X_d = vaeDecode(Z, Theta, size_input, size_hidden, size_latent);

    X_img = reshape(X_d, 28, 28, grid_size, grid_size);
    for i = 1:grid_size
        yi = (i-1)*28+1;
        for j = 1:grid_size
            xi = (j-1)*28+1;
            images(yi:yi+27, xi:xi+27, k) = X_img(:,:,i,j);
        end
    end
    
    figure(k);
    imagesc(v,v,images(:,:,k));
    colormap bone
end

%% Try out encoder
[X_cv, Y_cv] = loadMNIST('fashion-mnist/t10k-images-dx3-ubyte', ...
                         'fashion-mnist/t10k-labels-idx1-ubyte'); 
N = size(X_cv, 3);
X_cv = reshape(X_cv, [], N);
coords = vaeEncode(X_cv, Theta, size_input, size_hidden, size_latent);
scatter3(coords(1,:), coords(2,:), coords(3,:), 1, Y_cv / 9);