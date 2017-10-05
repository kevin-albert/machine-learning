function [ W, b, sz ] = conv2dInit(input_shape, kernel, filters, padded)
%CONV2dINIT Initialize weights and biases for 2d convolutional layer
% Input Arguments:
%   'input_shape'   - List of 3 values for x, y, channels
%   'kernel'        - Number or list of 2 numbers, size of receptive field
%   'filters'       - Number of output channels
%   'padded'        - (optional, default=false) Boolean - whether to zero-
%                     pad input so that output x / y dimensions match x.
% Return Values:
%   'W'             - 4d Weight matrix
%   'b'             - bias vector
%   'sz'            - Output feature map dimensions

    % Sanity check
    assert(numel(kernel) == 1 || numel(kernel) == 2);
    assert(numel(input_shape) >= 2 && numel(input_shape) <= 4);
    
    if nargin < 4
        % Defaults to shape = 'valid' for convolution
        padded = false;
    end

    if numel(kernel) == 1
        % Square kernel
        kernel = [kernel kernel];
    end
    
    if numel(input_shape) == 2
        % batch size = 1, input channels = 1
        input_shape = [input_shape 1 1];
    elseif numel(input_shape) == 3
        % 1 input channel
        input_shape = [input_shape 1];
    end
    
    if padded
        % Zero-padded so that input x / y matches output x / y
        sz = [input_shape(1), input_shape(2), filters, input_shape(4)];
    else
        % Retain only the valid portion of the convolution
        sz = [input_shape(1:2) - kernel + 1,  filters, input_shape(4)];
        sz = max(sz, 1);
    end
    
    % Scale weights according to size of input
    in_features = input_shape(4) * sum(kernel);
    W = randn(kernel(1), kernel(2), input_shape(3), filters) / ...
        sqrt(in_features);
    
    % Start with positive bias
    b = 0.1 * ones(filters, 1);
end

