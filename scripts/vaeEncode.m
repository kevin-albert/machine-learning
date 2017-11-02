function [ z ] = vaeEncode( x, Theta )
addpath('..');
m = size(x,2);

% Topology
[ W_encoder, b_encoder, W_mu, b_mu, W_logvar, b_logvar] = unpackParams(Theta, ...
  [size_input,  size_hidden], [size_hidden, 1], ... Encoder
  [size_hidden, size_latent], [size_latent, 1], ... Distribution
  [size_hidden, size_latent], [size_latent, 1]);

%% Encode

% Hidden layer
h_encoder = eluForward(W_encoder' * x + b_encoder);

% Latent distribution
mu = W_mu' * h_encoder + b_mu;
logvar = W_logvar' * h_encoder + b_logvar;

% Sample
epsilon = randn(size_latent, m);
encoderStd = exp(logvar);
z = mu + encoderStd .* epsilon;


end

