function [ J, grad ] = conv2dTestCompute( x, y, theta )

% Get params from theta
W1 = reshape(theta(1:36),    [3 4 1 3]);
b1 = reshape(theta(37:39),   [3 1]);
W2 = reshape(theta(40:135),  [4 4 3 2]);
b2 = reshape(theta(136:137), [2 1]);

% Forward pass
z2 = conv2dForward(x, W1, b1, true);
a2 = eluForward(z2);
z3 = conv2dForward(a2, W2, b2, false);
a3 = eluForward(z3);

% Compute output error
J = sum((y(:) - a3(:)) .^ 2) / 2;

% Backward pass
d3 = a3 - y;
dz3 = eluBackward(a3, d3);
[d2, dW2, db2] = conv2dBackward(a2, W2, dz3, false);
dz2 = eluBackward(a2, d2);
[~, dW1, db1] = conv2dBackward(x, W1, dz2, true);

grad = [ reshape(dW1, [], 1);
         reshape(db1, [], 1);
         reshape(dW2, [], 1);
         reshape(db2, [], 1) ];