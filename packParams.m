function [ Theta ] = packParams( varargin )
    Theta = [];
    for i = 1:nargin
        w = varargin{i};
        Theta = [ Theta; w(:) ];
    end
end

