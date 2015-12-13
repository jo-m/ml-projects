% Creation : 7 November 2015
% Author   : dtedali
% Project  : ML_prj_3rd

clear all
close all

addpath('./PHOG')
addpath('./DIPUMToolboxV1.1.3')

%%
% Adjust this number as you change the number of features.
% NUM_FEATURES = 696;
NUM_FEATURES = 698;

%% Generate 'train.csv'.

% Read the labels for the samples.
train_labels = csvread('train_labels.csv');

training_data = zeros(length(train_labels), NUM_FEATURES + 2);
sum1 = 0;
sum2 = 0;
count1 = 0;
count2 = 0;
errors1 = 0;
errors2 = 0;
for i = 1:length(train_labels)
    did = train_labels(i, 1);
    label = train_labels(i, 2);
    
    features = process_image('images/', did);
    
    
%     if label == 1
%         sum1 = sum1 + features;
%         count1 = count1 + 1;
%         if features <= 1.487125886319681
%             errors1 = errors1+1;
% %             imshow(imread(strcat('images/', sprintf('%04d', did), '_msk.png')));
% %             pause;
%         end
%     end
%   
%     if label == 2
%         sum2 = sum2 + features;
%         count2 = count2 + 1;
%         if features >=  1.487125886319681
%             errors2 = errors2+1;
%         end
%         
%     end
    
    
    
    training_data(i, :) = [did, features, label];
end

% 
% sum1/count1
% sum2/count2
% sum1/count1 - sum2/count2
% 
% 
% 
% errors1 / count1
% errors2 / count2
%  
% (errors1 + errors2) / (count1 + count2)

csvwrite('train.csv', training_data);

%% Generate 'test_validate.csv'.

train_ids = train_labels(:, 1);

test_validate_data = zeros(382, NUM_FEATURES + 1);

i = 1;
for did = 1:1272
    if any(train_ids == did)
        continue;
    end
  
    features = process_image('images/', did);

    test_validate_data(i, :) = [did, features];
    i = i + 1;
end

csvwrite('test_validate.csv', test_validate_data);


