function a = eluForward( z )
%ELU Exponential linear unit activation
% Input: value of any shape
% Output: ELU activation
a = z;
neg = a < 0;
a(neg) = exp(a(neg))-1;
end

