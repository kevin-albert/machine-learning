function [ dx ] = maxpool2dBackward( x, dh )
%MAXPOOL2DBACKWARD Max pooling backward pass


szx = size(x);
szh = size(dh);
assert(szx(3) == szh(3));

dx = zeros(szx);
stride = ceil(szx(1:2) ./ szh(1:2));
stride = reshape(stride, 2, 1);
sz2d = szx(1:2);

for i = 1:size(dh,1)
    for j = 1:size(dh,2)
        idx = min([ i-1 i; j-1 j] .* stride + [1 0; 1 0], sz2d);
        pool = x(idx(1,1):idx(1,2), idx(2,1):idx(2,2), :);
        h = max(max(pool, [], 1), [], 2);
        ind = x(idx(1,1):idx(1,2), idx(2,1):idx(2,2), :) == h;
        dx(idx(1,1):idx(1,2), idx(2,1):idx(2,2), :) = dh(i,j,:) .* ind;
    end
end
