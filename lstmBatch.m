function [ H, states, J, Theta_grad ] = lstmBatch(X, Y, states, Theta, ...
                                                  lambda, dropout)
%LSTMBATCH xxx
% X: m x 
% states: L x 6*N x M matrix

L = size(states, 1);
N = size(states, 2)/6;
M = size(states, 3);

% backprop only if Theta_grad is being returned
if nargout > 3
    Theta_grad = zeros(size(Theta));
end

% XXX dropout
% dropout mask
if nargin < 6 || (nargin == 6 && dropout == 1)
%     mask = ones(M,1);
else
    % XXX dropout
end

% use final state from last batch
previous = states(:, :, M);

%% Forward pass
for t = 1:M     % for each timestep
    
    % Offset into Theta (for retrieving W)
    offset = 1;
    for l = 1:L % for each layer
        
        if t == 1
            cp = previous(1+4*N:5*N);
            hp = previous(1+5*N:6*N);
        else
            cp = states(l, 1+4*N:5*N, t-1);
            hp = states(l, 1+5*N:6*N, t-1);
        end
        
        rows = 4*N;
        if l == 1
            cols = 1 + N + size(X, 2);
            x = X(t, :);
        else
            cols = 1 + N + N + size(X, 2);
            x = [ states(l-1, 5*N+1:end, t) X(t, :) ];
        end
        
        W = reshape(Theta(offset:offset+rows*cols-1), rows, cols);
        offset = offset + numel(W);

        % Calc all gates, input values
        z = [1 x hp] * W';

        a = tanh(z(1:N));
        i = sigmoid(z(1+1*N:2*N));
        f = sigmoid(z(1+2*N:3*N));
        o = sigmoid(z(1+3*N:4*N));
        
        % Linear cell state
        c = a .* i + cp .* f;
        
        % Output value (and input to next layer)
        h = tanh(c) .* o;

        % Save state
        states(l, :, t) = [a i f o c h];
    end
end

% output layer
rows = size(Y,2);
cols = 1+N;
Wy = reshape(Theta(offset:offset+rows*cols-1), rows, cols);

% lstm output activations
A = [ ones(M, 1) reshape(states(L, 1+5*N:6*N, :), M, []) ];

% output layer + softmax activation
H = exp(A * Wy');
H = H ./ sum(H, 2);

% cross-entropy error
% - find max of each row in Y
% - convert to linear indices
% - compute cross-entropy error based on softmax output
[~,k] = max(Y, [], 2);
k = sub2ind(size(Y), (1:size(Y,1))', k);
J = sum(-log(max(H(k), exp(-10))))/M;

if nargout < 4
    return
end

%% Backward pass
dH = (H - Y)/M;
dWy = dH' * A;
Theta_grad(offset:end) = reshape(dWy, [], 1);

% Recurrent errors
dC = zeros(L, N, M);
dA = zeros(L, N, M);
dA(L, :, :) = reshape(dH*Wy(:, 2:end), 1, N, M);

for t = M:-1:1      % for each time step
    offset = length(Theta)-numel(Wy)+1;
    for l = L:-1:1  % for each layer
        
        dc = dC(l, :, t);
        dh = dA(l, :, t);
        
        rows = 4*N;
        if l == 1
            cols = 1 + N + size(X, 2);
            x = X(t, :);
        else
            cols = 1 + N + N + size(X, 2);
            x = [ states(l-1, 5*N+1:end, t) X(t, :) ];
        end

        % Get W
        offset = offset - rows*cols;
        W = reshape(Theta(offset:offset+rows*cols-1), rows, cols);
     
        % Get state info
        a = states(l,     1:1*N, t);
        i = states(l,   1+N:2*N, t);
        f = states(l, 1+2*N:3*N, t);
        o = states(l, 1+3*N:4*N, t);
        c = states(l, 1+4*N:5*N, t);       
        if t == 1
            cp = previous(1+4*N:5*N);
            hp = previous(1+5*N:6*N);
        else
            cp = states(l, 1+4*N:5*N, t-1);
            hp = states(l, 1+5*N:6*N, t-1);
        end
        
        % Compute gradients

        % Carry over BPTT computation from t+1
        %   tanh'(x) = 1-tanh.^2(x)
        %   g'(x) = g(x)(1-g(x))
        dc = dc + dh .* o  .* (1-tanh(c).^2);

        da = dc .* i       .* (1-a.^2);
        di = dc .* a       .* i .* (1-i);
        df = dc .* cp      .* f .* (1-f);
        do = dh .* tanh(c) .* o .* (1-o);

        % Weights, dx, etc
        dz = [da di df do];
        dW = dz' * [1 x hp];

        %dI = WT * dzT
        dI = dz * W(:, 2:end);

        % Previous timestamp
        if t > 1
            dhp = dI(1+length(x):end);
            dA(l, :, t-1) = dA(l, :, t-1) + dhp;
            dC(l, :, t-1) = dc .* f;
        end

        % Previous layer
        if l > 1
            dhl = dI(1:N);
            dA(l-1, :, t) = dA(l-1, :, t) + dhl;
        end

        Theta_grad(offset:offset+rows*cols-1) = ...
            Theta_grad(offset:offset+rows*cols-1) + ... 
            reshape(dW, 1, []);
    end
end