function [ z, J, grad ] = vaeRun(x, Theta, size_input, size_hidden, size_latent, lambda)

addpath('..');
m = size(x,2);

% Topology
[ W_encoder, b_encoder, W_mu, b_mu, W_logvar, b_logvar, ...
  W_decoder, b_decoder, W_y, b_y ] = unpackParams(Theta, ...
  [size_input,  size_hidden], [size_hidden, 1], ... Encoder
  [size_hidden, size_latent], [size_latent, 1], ... Distribution
  [size_hidden, size_latent], [size_latent, 1], ...
  [size_latent, size_hidden], [size_hidden, 1], ... Decoder
  [size_hidden, size_input],  [size_input, 1]);

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

%% Decode
if nargout < 2
    return
end

% Hidden layer
h_decoder = eluForward(W_decoder' * z + b_decoder);

% Output
y = W_y' * h_decoder + b_y;

% Kullback-Leibler divergence 
KLD = -1/(2*m) * sum(1 + 2*logvar(:) - mu(:).^2  - exp(2*logvar(:)));

% Plain old mean squared error
MSE = 1/(2*m) * sum((x(:) - y(:)) .^ 2);

% Regularization
L2 = lambda/2 * (sum(W_encoder(:).^2) + sum(W_mu(:).^2) + ...
                 sum(W_logvar(:).^2) + sum(W_decoder(:).^2) + ...
                 sum(W_y(:).^2));

J = KLD+MSE+L2;

%% Backward pass

if nargout < 2
    return
end

dy = 1/m * (y-x);
dW_y = h_decoder * dy' + lambda * W_y;
db_y = sum(dy, 2);

d_decoder = eluBackward(h_decoder, W_y * dy);
dW_decoder = z * d_decoder' + lambda * W_decoder;
db_decoder = sum(d_decoder, 2);
dz = W_decoder * d_decoder;

d_logvar = dz .* (encoderStd) .* epsilon;
d_logvar = d_logvar + 1/m * (exp(2*logvar) - 1); % KLD
dW_logvar = h_encoder * d_logvar' + lambda * W_logvar;
db_logvar = sum(d_logvar, 2);

d_mu = dz;
d_mu = d_mu + 1/m * mu; % KLD
dW_mu = h_encoder * d_mu' + lambda * W_mu;
db_mu = sum(d_mu, 2);

d_encoder = eluBackward(h_encoder, W_logvar * d_logvar + W_mu * d_mu);
dW_encoder = x * d_encoder' + lambda * W_encoder;
db_encoder = sum(d_encoder, 2);


grad = packParams(dW_encoder, db_encoder, dW_mu, db_mu, dW_logvar, db_logvar, ...
                  dW_decoder, db_decoder, dW_y, db_y);
end

