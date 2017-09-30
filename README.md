# machine-learning
A collection of machine learning operations

### Contrived Example
A 5-layer convolutional network with ELU activations. See [examples/conv2dExample.m]

```matlab
%% Trivial 2d CNN example
% Setup network
% Input layer - 20x20 image with 3 channels
x = rand(20, 20, 3);

% CNN layer 1:
%   5x5 receptive field
%   6 output channels
%   zero-padded
[ W1, b1, dim1 ] = convLayer2d(size(x), 5, 6, true);

% CNN layer 2:
%   5x5 receptive field
%   3 output channels
%   zero-padded
[ W2, b2, dim2 ] = convLayer2d(dim1, 5, 3, true);

% CNN layer 3:
%   4x4 receptive field
%   2 output channels
[ W3, b3, dim3 ] = convLayer2d(dim2, 4, 2);

% CNN layer 4:
%   6x6 receptive field
%   1 output channel
[ W4, b4, dim4 ] = convLayer2d(dim3, 6, 1);

% CNN layer 5:
%   4x12 receptive field (flattens output feature map to 10x1x1)
[ W5, b5, dim5 ] = convLayer2d(dim4, [3 12], 1);

% Expected output
y = rand(dim5);
m = numel(y);

%% Training - basic SGD
epochs = 50;
alpha = 0.03;
MSE = zeros(epochs,1);
for i = 1:epochs
    %% Forward pass
    h1 = convff2d( x, W1, b1, true);
    e1 = elu(h1);
    
    h2 = convff2d(e1, W2, b2, true);
    e2 = elu(h2);
    
    h3 = convff2d(e2, W3, b3);
    e3 = elu(h3);
    
    h4 = convff2d(e3, W4, b4);
    e4 = elu(h4);
    
    h5 = convff2d(e4, W5, b5);
    e5 = elu(h5);
    
    % Output error
    MSE(i) = 1/2 * sum((e5(:)-y(:)).^2) / m;
    
    %% Backward pass
    de5 = (e5-y)/m;
    dh5 = elubp(e5, de5);
    [ de4, dW5, db5 ] = convbp2d(e4, W5, dh5);
    
    dh4 = elubp(e4, de4);
    [ de3, dW4, db4 ] = convbp2d(e3, W4, dh4);
    
    dh3 = elubp(e3, de3);
    [ de2, dW3, db3 ] = convbp2d(e2, W3, dh3);
    
    dh2 = elubp(e2, de2);
    [ de1, dW2, db2 ] = convbp2d(e1, W2, dh2, true);
    
    dh1 = elubp(e1, de1);
    [   ~, dW1, db1 ] = convbp2d(x,  W1, dh1, true); 
    
    %% SGD
    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;
end

%% Plot the results
plot(MSE); 
ylim([0, min(5, max(MSE))]);
legend('1/2 Mean squared error');
```
