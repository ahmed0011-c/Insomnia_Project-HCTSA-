%-------------------------------------------------------------------------------
% ---Day/night analysis
% Runs hctsa to understand differences in fly movement between
% day and night.
%-------------------------------------------------------------------------------
%% Label, normalize, and load data:
% Set how to normalize the data:
whatNormalization = 'mixedSigmoid'; % 'zscore', 'scaledRobustSigmoid'
% Label all time series by either 'insomnia' or 'good  sleepers':
TS_LabelGroups('raw',{'insomnia','gs'});
% Normalize the data, filtering out features with any special values:
% Load data in as a structure:
unnormalizedData = load('HCTSA.mat');TS_normalize(whatNormalization,[0,0.9],[],true);

% Load normalized data in a structure:
normalizedData = load('HCTSA_N.mat');
% Set classification parameters:
cfnParams = GiveMeDefaultClassificationParams(unnormalizedData,2);
cfnParams.whatClassifier = 'svm_linear';

%-------------------------------------------------------------------------------
% Optionally cluster and plot a colored data matrix:
doCluster = true
if doCluster
    TS_cluster();
    % reload with cluster info:
    normalizedData = load('HCTSA_N.mat');
end
TS_PlotDataMatrix(normalizedData,'colorGroups',true)

%-------------------------------------------------------------------------------
%% How accurately can inosmnia versus good sleepers be classified using all features:
numNulls = 1000; % (don't do any comparison to shuffled-label nulls)
TS_classify(normalizedData,cfnParams,numNulls);

%-------------------------------------------------------------------------------


%-------------------------------------------------------------------------------
%% What individual features best discriminate insomnia from good sleepers?
% Uses 'ustat' between day/night as a statistic to score individual features
% Produces 1) a pairwise correlation plot between the top features
%          2) class distributions of the top features, with their stats
%          3) a histogram of the accuracy of all features

numTopFeatures = 100; % number of features to include in the pairwise correlation plot
numFeaturesDistr = 100; % number of features to show class distributions for
whatStatistic = 'classification'; % rank-sum test p-value

TS_TopFeatures(normalizedData,whatStatistic,struct(),...
            'numTopFeatures',numTopFeatures,...
            'numFeaturesDistr',numFeaturesDistr,...
            'whatPlots',{'histogram','distributions','cluster'},'numNulls',1000);

%-------------------------------------------------------------------------------
%% Investigate particular individual features in some more detail
annotateParams = struct('maxL',4320);
featureID = 1752;
TS_FeatureSummary(featureID,normalizedData,true,annotateParams)
featureID = 1099;
TS_FeatureSummary(featureID,unnormalizedData,true,annotateParams)
