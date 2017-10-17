function [w, m, v] = adam(alpha, beta1, beta2, epsilon, t, w, g, m, v)
%ADAM Adaptive moment estimation optimzer
% update 1st and second moment averages
m = beta1 * m + (1 - beta1) * g;
v = beta2 * v + (1 - beta2) * g.^2;

% update param ws
alpha_t = alpha * sqrt(1 - beta2^t)/(1 - beta1^t);
w = w - alpha_t * m ./ (sqrt(v + epsilon));
end