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

Theta = randn(1,          1 * 4*cells * (1+inputs+cells) + ... 
                 (layers-1) * 4*cells * (1+inputs+cells+cells) + ...
                    outputs * (1+cells));
states = zeros(layers, 6*cells, batch);
