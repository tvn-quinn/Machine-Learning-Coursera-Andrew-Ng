function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.


% Initialize a set of C's and sigma's to try
% We try the same values for C's and sigma's so we only initialize C's
Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
numCs = length(Cs);

% Initialize a very high error value
final_error = inf

for i=1:numCs % each i represents a location from C
    for j=1:numCs % each j represents a location from sigmas
        model = svmTrain(X, y, Cs(i), @(x1,x2) gaussianKernel(x1,x2,Cs(j)));
        predictions = svmPredict(model,Xval);
        error = mean(double(predictions ~= yval));
        
        if error < final_error
            final_error = error;
            C = Cs(i);
            sigma = Cs(j);
        end
    end
end
% =========================================================================

end
