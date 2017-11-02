function [ x ] = vaeDecode(z, Theta, size_input, size_hidden, size_latent)
addpath('..');


% Topology
[ ~, ~, ~, ~, ~, ~, W_decoder, b_decoder, W_y, b_y ] = unpackParams(Theta, ...
  [size_input,  size_hidden], [size_hidden, 1],  ... Encoder
  [size_hidden, size_latent], [size_latent, 1], ... Distribution
  [size_hidden, size_latent], [size_latent, 1], ...
  [size_latent, size_hidden], [size_hidden, 1], ... Decoder
  [size_hidden, size_input],  [size_input, 1]);

%% Decode

% Hidden layer
h_decoder = eluForward(W_decoder' * z + b_decoder);

% Output
x = W_y' * h_decoder + b_y;
