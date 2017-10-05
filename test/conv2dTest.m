%% Conv2d - basic test no padding
clear;
addpath('../');
fprintf('conv2dTest\n');

x = rand(8,8,2);
[ W, b, sz ] = conv2dInit(size(x), 5, 3, false);
y = rand(sz);
theta = [ reshape(W, [], 1); reshape(b, [], 1) ];
f = @(t) conv2dTestComputeBasic(x, y, t, false);
diff = gradCheck(f, theta, 1e-9);
fprintf('    diff: %f\n', diff);
assert(diff < 1e-3);

%% Conv2d - basic test with padding
clear;

x = rand(8,8,2);
[ W, b, sz ] = conv2dInit(size(x), 5, 3, true);
y = rand(sz);
theta = [ reshape(W, [], 1); reshape(b, [], 1) ];
f = @(t) conv2dTestComputeBasic(x, y, t, true);
diff = gradCheck(f, theta, 1e-8);
fprintf('    diff: %f\n', diff);
assert(diff < 1e-3);

%% Conv2d - batched
clear;

% 10 examples / batch
x = rand(8,8,2,10);
[ W, b, sz ] = conv2dInit(size(x), 5, 3, false);
y = rand(sz);
theta = [ reshape(W, [], 1); reshape(b, [], 1) ];
f = @(t) conv2dTestComputeBasic(x, y, t, false);
diff = gradCheck(f, theta, 1e-7);
fprintf('    diff: %f\n', diff);
assert(diff < 1e-3);

%% Conv2d - multilayer test w/ elu
clear;

x = rand(16, 24);
[ W1, b1, sz1 ] = conv2dInit( size(x), [3,4], 3, true );
[ W2, b2, sz2 ] = conv2dInit( sz1, 4, 2, false );
y = rand(sz2);

theta = [ reshape(W1, [], 1);
          reshape(b1, [], 1);
          reshape(W2, [], 1);
          reshape(b2, [], 1) ];

f = @(t) conv2dTestCompute(x, y, t);

diff = gradCheck(f, theta, 1e-8);
fprintf('    diff: %f\n', diff);
assert(diff < 1e-3);

fprintf('\n');
