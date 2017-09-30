function [ dx, dW, db ] = conv2dBackward( x, W, dh, padded )
%CONV2DBACKWARD Compute 2d convolutional backward pass
% Input Arguments:
%   'x'         - Input activations
%   'W'         - 4d weight matrix
%   'dh'        - This layer's output errors
%   'padded'    - (optional, default=false) Boolean - whether to zero-pad 
%                 input so that output x / y dimensions match x.
% Return Values:
%   'dx'        - Input errors
%   'dW'        - Weight gradients
%   'db'        - Bias gradients

    n = size(W,3);  % number of incoming channels
    m = size(W,4);  % number of outgoing channels

    if (nargin < 4)
        % Defaults to shape = 'valid' for convolution
        padded = false;
    end
    
    % Bias gradients
    db = squeeze(sum(sum(dh)));
    
    % Determine output shape    
    if ~padded
        % Simple case - no padding, backprop unaffected
        shape = 'full';
        x_ = x;
    else
        % Hmmm... this is a bit of a hack
        % We manually zero-pad x so we can get the right shape back from
        % filter2 down below. It's probably quite inefficient
        % But it seems to work ;)
        shape = 'same';
        pad = [size(W,1), size(W,2)] - 1;
        pre = floor(pad/2);
        post = ceil(pad/2);

        x_ = padarray(x, pre, 0, 'pre');
        x_ = padarray(x_, post, 0, 'post');
    end

    % Weight gradients
    dW = zeros(size(W));

    % Backprop
    dx = zeros(size(x));

    for i = 1:n
        for j = 1:m
            dx(:,:,i) = dx(:,:,i) + conv2(dh(:,:,j), W(:,:,i,j), shape);
            dW(:,:,i,j) = conv2(x_(:,:,i), rot90(dh(:,:,j), 2), 'valid');
        end
    end
end
