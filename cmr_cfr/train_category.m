function [curr, prev, base, trainpos] = train_category(pres_category)
%TRAIN_CATEGORY   Categories associated with the current train.
%
%  Given a set of category labels for presented lists, determines
%  the category associated with the previous train and the other
%  baseline category that is neither the previous category or the
%  current one.
%
%  [curr, prev, base, trainpos] = train_category(pres_category)
%
%  INPUTS
%  pres_category - [trials x items] numeric array
%      Numeric code for the item presented at each serial position
%      in each list.
%
%  OUTPUTS
%  curr - [trials x items] numeric array
%      Numeric code for the category of items in the current
%      train. The first train is NaN, to match the other outputs
%      that are undefined for the first train.
%
%  prev - [trials x items] numeric array
%      Numeric code for the category of items in the previous train.
%
%  base - [trials x items] numeric array
%      Numeric code for the category of items that is not the
%      current category or the previous category.
%
%  trainpos - [trials x items] numeric array
%      Position of each item within the current train.

[n_trial, n_item] = size(pres_category);

curr = NaN(n_trial, n_item);
prev = NaN(n_trial, n_item);
base = NaN(n_trial, n_item);
trainpos = NaN(n_trial, n_item);
for i = 1:n_trial
    % train labels (starting at zero)
    train = cumsum([0 diff(pres_category(i,:)) ~= 0]);
    utrain = unique(train);
    
    % check category labels for this list
    ucat = unique(pres_category(i,:));
    if length(ucat) ~= 3
        error('There must be three unique categories in each list.');
    end
    
    for j = 1:length(utrain)
        trainind = train==utrain(j);
        trainpos(i,trainind) = 1:nnz(trainind);
        if j > 1
            % get the previous and baseline category for each train after
            % the first
            currcat = pres_category(i,find(train==utrain(j),1));
            prevcat = pres_category(i,find(train==utrain(j-1),1));
            basecat = ucat(ucat~=currcat & ucat~=prevcat);
            
            curr(i,trainind) = currcat;
            prev(i,trainind) = prevcat;
            base(i,trainind) = basecat;
        end
    end
end
