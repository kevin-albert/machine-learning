%% Trivial 2d CNN example
addpath('../');
% Setup network
% Input layer - 20x20 image with 3 channels
x = rand(20, 20, 3);

% CNN layer 1:
%   5x5 receptive field
%   6 output channels
%   zero-padded
[ W1, b1, dim1 ] = conv2dInit(size(x), 5, 6, true);

% CNN layer 2:
%   5x5 receptive field
%   3 output channels
%   zero-padded
[ W2, b2, dim2 ] = conv2dInit(dim1, 5, 3, true);

% CNN layer 3:
%   4x4 receptive field
%   2 output channels
[ W3, b3, dim3 ] = conv2dInit(dim2, 4, 2);

% CNN layer 4:
%   6x6 receptive field
%   1 output channel
[ W4, b4, dim4 ] = conv2dInit(dim3, 6, 1);

% CNN layer 5:
%   4x12 receptive field (flattens output feature map to 10x1x1)
[ W5, b5, dim5 ] = conv2dInit(dim4, [3 12], 1);

% Expected output
y = rand(dim5);
m = numel(y);

%% Training - basic SGD
epochs = 50;
alpha = 0.03;
MSE = zeros(epochs,1);
for i = 1:epochs
    %% Forward pass
    h1 = conv2dForward( x, W1, b1, true);
    e1 = eluForward(h1);
    
    h2 = conv2dForward(e1, W2, b2, true);
    e2 = eluForward(h2);
    
    h3 = conv2dForward(e2, W3, b3);
    e3 = eluForward(h3);
    
    h4 = conv2dForward(e3, W4, b4);
    e4 = eluForward(h4);
    
    h5 = conv2dForward(e4, W5, b5);
    e5 = eluForward(h5);
    
    % Output error
    MSE(i) = 1/2 * sum((e5(:)-y(:)).^2) / m;
    
    %% Backward pass
    de5 = (e5-y)/m;
    dh5 = eluBackward(e5, de5);
    [ de4, dW5, db5 ] = conv2dBackward(e4, W5, dh5);
    
    dh4 = eluBackward(e4, de4);
    [ de3, dW4, db4 ] = conv2dBackward(e3, W4, dh4);
    
    dh3 = eluBackward(e3, de3);
    [ de2, dW3, db3 ] = conv2dBackward(e2, W3, dh3);
    
    dh2 = eluBackward(e2, de2);
    [ de1, dW2, db2 ] = conv2dBackward(e1, W2, dh2);
    
    dh1 = eluBackward(e1, de1);
    [   ~, dW1, db1 ] = conv2dBackward(x,  W1, dh1); 
    
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
