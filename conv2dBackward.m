function [ dx, dW, db ] = conv2dBackward( x, W, dh )
%CONV2DBACKWARD Compute 2d convolutional backward pass
% Input Arguments:
%   'x'         - Input activations (2d, 3d, or 4d matrix) where
%                   dimension 1 is data along the y axis
%                   dimension 2 is data along the x axis
%                   dimension 3 is input channels
%                   dimension 4 is is batches
%   'W'         - 4d weight matrix
%   'dh'        - This layer's output errors
% Return Values:
%   'dx'        - Input errors
%   'dW'        - Weight gradients
%   'db'        - Bias gradients

    n = size(W,3);  % number of incoming channels
    m = size(W,4);  % number of outgoing channels
    batches = size(x,4);
    assert(size(dh, 4) == batches);

    % Bias gradients
    db = squeeze(sum(sum(sum(dh,4))));
    
    % Determine output shape    
    if (size(x,1) ~= size(dh,1) || size(x,2) ~= size(dh,2))
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

    for k = 1:batches
        for i = 1:n
            for j = 1:m
                dx(:,:,i,k) = dx(:,:,i,k) + conv2(dh(:,:,j,k), W(:,:,i,j), shape);
                dW(:,:,i,j) = dW(:,:,i,j) + ...
                              conv2(x_(:,:,i,k), rot90(dh(:,:,j,k), 2), 'valid');
            end
        end
    end
end
