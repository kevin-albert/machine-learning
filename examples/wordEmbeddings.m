%% Parameters
skipLength = 5;

%% Load data

fprintf('Loading data\n');
fp = fopen('tng_processed.txt');
raw = textscan(fp, '%s'); raw = raw{1};
n = length(raw);
k = 256; 

dictionary = unique(raw);
m = length(dictionary);
encoder = containers.Map(dictionary, 1:m);

X = zeros(n,1);
for i = 1:n
    X(i) = encoder(char(raw{i}));
end


%% Calculate unigram probabilities
fprintf('Counting unigrams\n');

Punig = zeros(m,1);
for i = 1:length(X)
    Punig(X(i)) = Punig(X(i)) + 1;
end
Punig = Punig / sum(Punig);

%% Calculate skipgram probabilities
fprintf('Counting weighted skipgrams\n');

Pskip = sparse(1:m, 1:m, zeros(m,1));
for i = 1:n
    from = max(i-skipLength, 1);
    to = min(i+skipLength, length(raw));
    for j = from:to
        if X(i) ~= X(j)
            Pskip(X(i), X(j)) = Pskip(X(i), X(j)) + (1+skipLength - abs(i-j))/skipLength;
        end
    end
end
Pskip = Pskip / sum(Pskip(:));

%% Pointwise mutual information
PMI = Pskip ./ Punig ./ Punig';
idx = PMI ~= 0;
PMI(idx) = max(0,log(PMI(idx)));

% this one can't be right - all zeros become -Inf, borking my sparse matrix 
% PMI = log(Pskip ./ Punig ./ Punig');

%% Reduce dimensionality

fprintf('Computing SVD\n');
[U,S,V] = svds(PMI, k);

fprintf('Done\n');

save wordVecData

%% Try it out

