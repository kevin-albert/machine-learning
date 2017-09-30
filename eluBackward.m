function [ dz ] = eluBackward( a, da )
%ELUBACKWARD Exponential linear unit derivative
% Input Parameters:
%   'a'     - ouput activation (e.g. "a = elu(z)" for some z)
%   'da'    - partial derivative of error WRT a
% Returns the partial derivative of error WRT z
% Inputs and outputs are all the same shape
dz = a+1;
dz(dz > 1) = 1;
dz = dz .* da;
end

