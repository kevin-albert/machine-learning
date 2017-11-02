function [ varargout ] = unpackParams( Theta, varargin )
    st = 1;
    for i = 1:nargout
        sz = varargin{i};
        n = prod(sz);
        varargout{i} = reshape(Theta(st:st+n-1), sz);
        st = st + n;
    end
end

