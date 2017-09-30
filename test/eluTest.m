fprintf('eluTest\n');
addpath('../');

theta = [-0.5; 0.1; 2; 10];
y = 1/3 * ones(size(theta));

elf = @(x) eluForward(x);
f = @(t) deal(1/2*sum((y-elf(t)).^2), ...
                  eluBackward(elf(t), elf(t)-y) );

diff = gradCheck(f, theta, 1e-4);
fprintf('diff = %f\n', diff);
assert(diff < 1e-7);


fprintf('\n');