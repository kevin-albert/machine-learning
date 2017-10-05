function [ J, grad ] = conv2dTestComputeBasic( x, y, theta, padded )

W = reshape(theta(1:150), [5 5 2 3]);
b = reshape(theta(151:153), [3 1]);
h = conv2dForward(x, W, b, padded);
J = 1/2 * sum((y(:) - h(:)) .^ 2) / size(x,4);
dh = (h-y)/ size(x,4);
[~, dW, db] = conv2dBackward(x, W, dh);
grad = [reshape(dW, [], 1); reshape(db, [], 1)];