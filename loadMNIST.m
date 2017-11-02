function [ X, Y ] = loadMNIST( imageFile, labelFile )
%loadMNIST load MNIST database
% Input Parameters
%   'imageFile' - where to load images from
%   'labelFile' - where to load labels from
% Output Variables
%   'X'         - 28x28xN matrix of images
%   'Y'         - Nx1 vector of labels

fp = fopen(imageFile, 'rb');
[~] = fread(fp, 1, 'int32', 0, 'ieee-be');

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
X = fread(fp, inf, 'unsigned char');

fclose(fp);

X = reshape(X, numCols, numRows, numImages) / 255;
X = permute(X, [2,1,3]);

fp = fopen(labelFile, 'rb');
[~] = fread(fp, 1, 'int32', 0, 'ieee-be');
[~] = fread(fp, 1, 'int32', 0, 'ieee-be');
Y = fread(fp, inf, 'unsigned char');
fclose(fp);

end


