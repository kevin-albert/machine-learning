function [ h ] = maxpool2dForward( x, stride )
%MAXPOOL2DFORWARD Summary of this function goes here
%   Detailed explanation goes here

if numel(stride) == 1
    stride = [stride stride];
else
    assert(numel(stride) == 2);
end
stride = reshape(stride, [], 1);

h = zeros(ceil(size(x,1)/stride(1)), ceil(size(x,2)/stride(2)), size(x,3));
sz2d = [size(x,1), size(x,2)];

for i = 1:size(h,1)
    for j = 1:size(h,2)
        idx = min([ i-1 i; j-1 j] .* stride + [1 0; 1 0], sz2d);
        pool = x(idx(1,1):idx(1,2), idx(2,1):idx(2,2), :);
        h(i,j,:) = max(max(pool, [], 1), [], 2);
    end
end