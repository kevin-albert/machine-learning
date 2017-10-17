function [ Theta, states ] = lstmInit( inputs, cells, outputs, layers, batch )
%LSTMINIT Create the parameter / state matrices for an LSTM according to the 
%   given topology
% Input Arguments:
%   'inputs'    - Number of inputs
%   'cells'     - Number of cells per layer
%   'outputs'   - Number of outputs
%   'layers'    - Number of layers
%   'batch'     - batch size (does not affect Theta)
% Output Values:
%   'Theta'     - Flattened parameter vector, acceptable by lstmBatch
%   'states'    - L x 6*N x M state matrix (L = layers, N = cells, M = batches)

% Start out with guassian random weights, then scale them according to
% 1/stddev
Theta = randn(1,          1 * 4*cells * (1+inputs+cells) + ... 
                 (layers-1) * 4*cells * (1+inputs+cells+cells) + ...
                    outputs * (1 + layers*cells));

% Input weights
n = 4*cells*(1+inputs+cells);
Theta(1:n-1) = Theta(1:n-1) / sqrt(inputs+cells);

% Hidden weights
m = n + (layers-1) * 4*cells * (1+inputs+cells+cells);
Theta(n:m-1) = Theta(n:m-1) / sqrt(inputs+cells+cells);

% Output weights
Theta(m:end) = Theta(m:end) / sqrt(layers*cells);

% Biases
W = reshape(Theta(1:1 * 4*cells * (1+inputs+cells)), 4*cells, []);
W(1+2*cells:3*cells, 1) = 5;
Theta(1:1 * 4*cells * (1+inputs+cells)) = reshape(W, 1, []);
offset = 1;
for layer = 2:layers
    offset = offset + numel(W);
    W = reshape(Theta(offset:offset+4*cells*(1+inputs+cells+cells)-1), 4*cells, []);
    % activation / input gate biases
    W(1:2*cells, 1) = randn(2*cells, 1);
    
    % forget gate biases
    W(1+2*cells:3*cells, 1) = 5;
    
    % output gate biases
    W(1+3*cells:4*cells, 1) = randn(cells, 1);
    Theta(offset:offset+4*cells*(1+inputs+cells+cells)-1) = reshape(W, 1, []);
end

states = zeros(layers, 6*cells, batch);
