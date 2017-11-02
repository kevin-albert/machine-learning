function [ W, b ] = denseInit( input_size, output_size )
%DENSEINIT Initialize weights and biases for fully-connected layer

W = randn(input_size, output_size) / sqrt(input_size);
b = rand(output_size, 1);

end

