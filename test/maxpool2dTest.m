clear;
addpath('../');
fprintf('maxpool2dTest\n');

h = maxpool2dForward(rand(10,12,2), [2,3]);
assert(size(h,1) == 5);
assert(size(h,2) == 4);
assert(size(h,3) == 2);

y = rand(size(h));
J = @(x) 1/2*sum((y(:)-reshape(maxpool2dForward(x, [2,3]), [], 1)).^2);
grad = @(x) maxpool2dBackward(x, maxpool2dForward(x, [2,3])-y);
f = @(x) deal(J(x), grad(x));

x_init = reshape(randperm(10*12*2), 10, 12, 2)/240;
diff = gradCheck(f, x_init, 1e-4);
fprintf('    diff: %f\n', diff);
assert(diff < 1e-9);

fprintf('\n');