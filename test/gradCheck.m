function [ diff, f_grads, nc_grads ] = gradCheck( f, theta, epsilon )
%GRADCHECK Compute numerical gradients for f(theta) w.r.t theta to validate
%   backprop algorithms

[~, f_grads] = f(theta);
nc_grads = zeros(size(theta));
for i = 1:numel(nc_grads)
    theta_orig = theta(i);
    theta(i) = theta_orig-epsilon;
    [J1, ~] = f(theta);
    theta(i) = theta_orig+epsilon;
    [J2, ~] = f(theta);
    nc_grads(i) = (J2-J1)/(2*epsilon);
end

diff = norm(f_grads-nc_grads);

