% Parameter initialization
addpath('../');
addpath('../../');
size_input = 10;
size_hidden = 5;
size_latent = 2;
lambda = 0.01;
m = 4; % batch size
x = rand(size_input, m);

W_encoder = randn(size_input, size_hidden) * 0.001;
b_encoder = rand(size_hidden, 1);
W_mu = randn(size_hidden, size_latent) * 0.001;
b_mu = rand(size_latent, 1);
W_logvar = randn(size_hidden, size_latent);
b_logvar = rand(size_latent, 1);
W_decoder = randn(size_latent, size_hidden) * 0.001;
b_decoder = rand(size_hidden, 1);
W_y = randn(size_hidden, size_input) * 0.001;
b_y = rand(size_input, 1);

Theta = [ W_encoder(:); b_encoder(:); W_mu(:); b_mu(:); 
          W_logvar(:); b_logvar(:);
          W_decoder(:); b_decoder(:); W_y(:); b_y(:); ];
      
rng(1);
[ ~, J, grad_calc ] = vaeRun(x, Theta, size_input, size_hidden, size_latent, lambda);


epsilon = 1e-7;
grad_comp = zeros(size(grad_calc));
% start = 1+numel(Theta)-numel(b_y)-numel(W_y) ...
%          -numel(b_decoder)-numel(W_decoder) ...
%          -numel(b_logvar)-numel(W_logvar) ...
%          -numel(b_mu)-numel(W_mu);
start = 1;

for i = start:numel(Theta)
    theta = Theta(i);
    Theta(i) = theta - epsilon;
    rng(1);
    [~,J1] = vaeRun(x, Theta, size_input, size_hidden, size_latent, lambda);
    Theta(i) = theta + epsilon;
    rng(1);
    [~,J2] = vaeRun(x, Theta, size_input, size_hidden, size_latent, lambda);
    Theta(i) = theta;
    grad_comp(i) = (J2-J1)/(2*epsilon);
end

grad_diff = grad_calc-grad_comp;
norm(grad_diff(start:end))
plot(grad_diff(start:end));
