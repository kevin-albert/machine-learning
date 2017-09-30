%% Conv2d - basic test no padding
clear;
fprintf('conv2dTest - basic (no padding)\n');

x = rand(8,8,2);
[ W, b, sz ] = conv2dInit(size(x), 5, 3, false);
y = rand(sz);
theta = [ reshape(W, [], 1); reshape(b, [], 1) ];
f = @(t) conv2dTestComputeBasic(x, y, t, false);
diff = gradCheck(f, theta, 1e-9);
fprintf('diff = %f\n', diff);
assert(diff < 1e-3);

%% Conv2d - basic test with padding
clear;
fprintf('conv2dTest - basic (with padding)\n');

x = rand(8,8,2);
[ W, b, sz ] = conv2dInit(size(x), 5, 3, true);
y = rand(sz);
theta = [ reshape(W, [], 1); reshape(b, [], 1) ];
f = @(t) conv2dTestComputeBasic(x, y, t, true);
diff = gradCheck(f, theta, 1e-8);
fprintf('diff = %f\n', diff);
assert(diff < 1e-3);

%% Conv2d - multilayer test w/ elu
clear;
fprintf('conv2dTest - multilayer network w/ elu\n');

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
fprintf('diff = %f\n', diff);
assert(diff < 1e-3);

fprintf('\n');
