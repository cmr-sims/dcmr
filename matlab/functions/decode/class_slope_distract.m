function stats = class_slope_distract(decode)
%CLASS_SLOPE_DISTRACT   Calculate slope over train position by distraction.
%
%  Takes output from decode_cfrl, loaded using load_decode_cfrl.
%
%  stats = class_slope_distract(decode)

% distraction and train position information
distract = [0 2.5 7.5];
[~, ~, ~, trainpos] = train_category(decode.subj_data{1}.pres.category);

n_subj = length(decode.subj_data);
n_ctype = 3;
n_distract = length(distract);
n_trainpos = length(unique(trainpos));

eeg_m = NaN(n_trainpos, n_ctype, n_distract, n_subj);
con_m = NaN(n_trainpos, n_ctype, n_distract, n_subj);
eeg_b = NaN(n_ctype, n_distract, n_subj);
con_b = NaN(n_ctype, n_distract, n_subj);
for i = 1:n_distract
    for j = 1:n_subj
        % vectorize the distraction matrix to match the trial vectors
        dmat = decode.subj_data{j}.pres.distractor;
        temp = dmat';
        dvec = temp(:);
        
        % get evidence by train position for this distraction condition
        vec_ind = dvec == distract(i);
        mat_ind = dmat(:,1) == distract(i);
        [eeg_evid, con_evid, n] = ...
            evidence_trainpos(decode.eeg_evidence{j}(vec_ind,:), ...
                              decode.con_evidence{j}(vec_ind,:), ...
                              decode.subj_data{j}.pres.category(mat_ind,:));
        eeg_m(:,:,i,j) = eeg_evid;
        con_m(:,:,i,j) = con_evid;
        
        % slope over train position
        x = 1:3;
        for k = 1:n_ctype
            % data to fit
            tot_n = n(x,k);
            eeg_y = eeg_evid(x,k)';
            con_y = con_evid(x,k)';
            
            % estimate slope, weighting by trial count
            b = glmfit(x, eeg_y, 'normal', 'weights', tot_n);
            eeg_b(k,i,j) = b(2);
            b = glmfit(x, con_y, 'normal', 'weights', tot_n);
            con_b(k,i,j) = b(2);
        end
    end
end

stats = struct;
stats.trainpos = unique(trainpos)';
stats.ctype = {'curr' 'prev' 'base'};
stats.distract = distract;
stats.eeg_m = eeg_m;
stats.con_m = con_m;
stats.eeg_b = eeg_b;
stats.con_b = con_b;
