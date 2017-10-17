%% Init
addpath('..');

%% Load training data

if ~exist('X', 'var') || ~exist('Y', 'var') || ... 
        ~exist('num2char', 'var') || ~exist('char2num', 'var')
    % Read text, map each unique character to a number
    text_data = fileread('tng.txt');
    N = length(text_data);
    hazchar = false(255,1);
    for i = 1:N
        c = text_data(i);
        hazchar(+c) = true;
    end

    % Create mapping arrays
    num_classes = sum(hazchar);
    char2num = zeros(255,1) - 1;
    num2char = zeros(num_classes, 1);
    i = 1;
    for c = 1:255
        if hazchar(c)
            char2num(c) = i;
            num2char(i) = char(c);
            i = i + 1;
        end
    end

    %% Create training data matrices
    X = zeros(N, num_classes);
    for i = 1:N
        j = char2num(+text_data(i));
        X(i, j) = 1;
    end
    
    N = length(X) - 1;
    Y = X(2:N+1, :);
    X = X(1:N, :);
end

%% Train it

% topology
layers = 3;
num_cells = 300;

% training sequence params
epochs  = 10;
batch   = 100;

% optimization params
alpha   = 0.001;
beta1   = 0.9;
beta2   = 0.999;
epsilon = 1e-8;

% regularization params
lambda  = 0.001;
dropout = 1;

% initial params + state
[ Theta, states ] = lstmInit(num_classes, num_cells, num_classes, layers, batch);
[ ~, test_state ] = lstmInit(num_classes, num_cells, num_classes, layers, 1);
Theta_m = zeros(size(Theta));
Theta_v = zeros(size(Theta));
t = 1;

for epoch = 1:epochs
    J_train = 0;
    for i = 1:batch:N
        j = min(i+batch-1, N);
        x = [X(i:j, :); zeros(batch-j, num_classes)];
        y = [Y(i:j, :); zeros(batch-j, num_classes)];
        x(j:batch, char2num(+' ')) = 1;
        y(j:batch, char2num(+' ')) = 1;
        [ ~, states, J, Theta_g ] = lstmBptt(x, states, Theta, y, lambda, dropout);
        [Theta, Theta_m, Theta_v] = adam(alpha, beta1, beta2, epsilon, t, ...
                                         Theta, Theta_g, Theta_m, Theta_v);
        t = t + 1;
        J_train = J_train + J / ceil(N/batch);
        if mod((i-1)/batch, 100) == 0
            fprintf('[%3d] %2.2f%%\n', epoch, 100*i/N);
        end
    end
    fprintf('[%3d] J = %2.4f, sample text:\n', epoch, J_train);
    fprintf('================================================================================\n');
    
    x = zeros(1, num_classes);
    test_state = zeros(size(test_state));
    x(char2num('^')) = 1;
    for i = 1:500
        [ y, test_state ] = lstmBatch(x, test_state, Theta);
        p = randsample(num_classes, 1, true, y);
        x = zeros(1, num_classes);
        x(p) = 1;
        fprintf('%s', char(num2char(p)));
    end
    fprintf('\n');
    fprintf('================================================================================\n');
end