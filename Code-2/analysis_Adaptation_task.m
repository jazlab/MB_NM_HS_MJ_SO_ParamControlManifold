% This code is used to generate the ephys results from:
% 
% - Figure 7C
% 
% reported in the following paper:
%
% Beiran M, Meirhaeghe N, Sohn H, Jazayeri M, Ostojic S (2021) Parametric 
% control of flexible timing through low-dimensional neural manifolds. 
% bioRxiv; https://doi.org/10.1101/2021.11.08.467806
%
% The script runs in one go and depends on functions appended at the end.
% For additional information contact nmrghe@gmail.com

% Clear workspace
close all
clear all
clc

% Specify the monkey (G versus H)
id_monkey = 'G'; id_session = 'G_200407';
% id_monkey = 'H'; id_session = 'H_200426';

% Load data
load(['../Data/' id_session '_Readym200-Set_bin20_alignedReady'])

% Specify fixed parameters for the analysis
t_min_peri_ready = 200; % in msec
wkernel = 40;
nBootstrap = 100;
offset_post_ready = 400;
t_s_short_pre_unique = unique(t_s_all(id_short_all));
t_s_long_pre_unique = unique(t_s_all(~id_short_all));
nb_trials_long_post = 100; % nb of trials in blocks after transition 

% Remove outlier trials based on behavior
[D_clean, ~, ~, id_left_clean, id_short_clean, id_pre_clean] ...
    = cleanDataTensor_2prior(D, t_p_all, t_s_all, id_left_all, id_short_all, id_pre_all, id_trial_all);

% Compute PSTH from Ready to Set
PSTH_short_pre = generateBlockedTrialAvgPSTHwithBootstrap( D_clean(t_min_peri_ready/wbin:(t_min_peri_ready+max(t_s_short_pre_unique))/wbin, :, logical(id_short_clean.*id_pre_clean)), id_left_clean(logical(id_short_clean.*id_pre_clean)), wbin, wkernel, 1, 0, true, nBootstrap);
PSTH_long_pre = generateBlockedTrialAvgPSTHwithBootstrap( D_clean(t_min_peri_ready/wbin:(t_min_peri_ready+max(t_s_long_pre_unique))/wbin, :, logical(~id_short_clean.*id_pre_clean)), id_left_clean(logical(~id_short_clean.*id_pre_clean)), wbin, wkernel, 1, 0, true, nBootstrap);
PSTH_long_post = generateBlockedTrialAvgPSTHwithBootstrap( D_clean(t_min_peri_ready/wbin:(t_min_peri_ready+max(t_s_long_post_unique))/wbin, :, logical(~id_short_clean.*~id_pre_clean)), id_left_clean(logical(~id_short_clean.*~id_pre_clean)), wbin, wkernel, 0, nb_trials_long_post, true, nBootstrap);
nb_blocks_post = length(PSTH_long_post);

% Fixed the format of PSTH to match
PSTH_short_pre = PSTH_short_pre{1};
PSTH_long_pre = PSTH_long_pre{1};

% Make sure no neuron has nan in them or 
neurons2keep = find(logical(~isnan(nanmean(nanmean(PSTH_short_pre, 3), 1)).*nanmean(nanmean(PSTH_short_pre, 3), 1)>1));
neurons2keep = intersect(neurons2keep, find(logical(~isnan(nanmean(nanmean(PSTH_long_pre, 3), 1)).*nanmean(nanmean(PSTH_long_pre, 3), 1)>1)));
% Identify unstable neurons during adaptation
for ind_block = 1:nb_blocks_post
    neurons2keep = intersect(neurons2keep, find(logical(~isnan(nanmean(nanmean(PSTH_long_post{ind_block}, 3), 1)).*nanmean(nanmean(PSTH_long_post{ind_block}, 3), 1)>1)));
end
nb_neurons = length(neurons2keep);
idNeurons = idSU(neurons2keep, 1);

% Remove all identified neurons
PSTH_short_pre = PSTH_short_pre(:, neurons2keep, :);
PSTH_long_pre = PSTH_long_pre(:, neurons2keep, :);
for ind_block = 1:nb_blocks_post
    PSTH_long_post{ind_block} = PSTH_long_post{ind_block}(:, neurons2keep, :);
end

% Define the state at the time of Ready (here I use spike counts directly)
% using a 200-ms window centered around Ready
ind_bin_start=5; % 5*wbin = 100 ms, so 100 ms before Ready
ind_bin_end=15; % 15*wbin = 300 ms, so 100 ms after Ready

% Get the spike counts to define the Ready state for short/long pre/post
spkcount_short_pre = nanmean(squeeze(nansum(D(ind_bin_start:ind_bin_end, :, logical(id_short_all.*id_pre_all)), 1)), 2)/((ind_bin_end-ind_bin_start+1)*wbin*1e-3);
spkcount_long_pre = nanmean(squeeze(nansum(D(ind_bin_start:ind_bin_end, :, logical(~id_short_all.*id_pre_all)), 1)), 2)/((ind_bin_end-ind_bin_start+1)*wbin*1e-3);
ind_trial_post = find(logical(~id_short_all.*~id_pre_all));
spkcount_long_post = nanmean(squeeze(nansum(D(ind_bin_start:ind_bin_end, :, ind_trial_post(1:nb_trials_long_post)), 1)), 2)/((ind_bin_end-ind_bin_start+1)*wbin*1e-3);

% Define the context dimension pointing from Long to Short 
context_dim = spkcount_short_pre(neurons2keep)-spkcount_long_pre(neurons2keep);
context_dim = context_dim/norm(context_dim);

% Compute the projection onto the context dimension at the time of Ready
spkcount_short_pre_alltrials = squeeze(nansum(D(ind_bin_start:ind_bin_end, :, logical(id_short_all.*id_pre_all)), 1))/((ind_bin_end-ind_bin_start+1)*wbin*1e-3);
spkcount_long_pre_alltrials = squeeze(nansum(D(ind_bin_start:ind_bin_end, :, logical(~id_short_all.*id_pre_all)), 1))/((ind_bin_end-ind_bin_start+1)*wbin*1e-3);
spkcount_long_post_alltrials = squeeze(nansum(D(ind_bin_start:ind_bin_end, :, ind_trial_post(1:nb_trials_long_post)), 1))/((ind_bin_end-ind_bin_start+1)*wbin*1e-3);

% Use bootstrap to estimate the uncertainty around the mean state
for iBootstrap = 1:nBootstrap
    spkcount_short_pre_bootstrap(:, iBootstrap) = nanmean(datasample(spkcount_short_pre_alltrials, size(spkcount_short_pre_alltrials, 2), 2), 2);
    spkcount_long_pre_bootstrap(:, iBootstrap) = nanmean(datasample(spkcount_long_pre_alltrials, size(spkcount_long_pre_alltrials, 2), 2), 2);
    spkcount_long_post_bootstrap(:, iBootstrap) = nanmean(datasample(spkcount_long_post_alltrials, size(spkcount_long_post_alltrials, 2), 2), 2);
end

% Center the data using the average pre state
x_center = (spkcount_short_pre(neurons2keep)+spkcount_long_pre(neurons2keep))/2;
proj_short_pre_bootstrap = context_dim'*(spkcount_short_pre_bootstrap(neurons2keep, :)-x_center);
proj_long_pre_bootstrap = context_dim'*(spkcount_long_pre_bootstrap(neurons2keep, :)-x_center);
proj_long_post_bootstrap = context_dim'*(spkcount_long_post_bootstrap(neurons2keep, :)-x_center);

% For each bootstrap, compute the speed of trajectories using two methods:
% - the slope of the speed mapping
% - the average instantaneous euclidean distance between nearby states
for ind_bootstrap = 1:nBootstrap
    [vec_min_pre_short(ind_bootstrap, :), slope_pre_short(ind_bootstrap)] = computeSpeedMapping_fromReady(PSTH_short_pre(:, :, 1), PSTH_short_pre(:, :, ind_bootstrap), offset_post_ready/wbin);
    [vec_min_pre_long(ind_bootstrap, :), slope_pre_long(ind_bootstrap)] = computeSpeedMapping_fromReady(PSTH_short_pre(:, :, ind_bootstrap), PSTH_long_pre(:, :, ind_bootstrap), offset_post_ready/wbin);
    [vec_min_post_long(ind_bootstrap, :), slope_post_long(ind_bootstrap)] = computeSpeedMapping_fromReady(PSTH_short_pre(:, :, ind_bootstrap), PSTH_long_post{1}(:, :, ind_bootstrap), offset_post_ready/wbin);
    speed_pre_short(ind_bootstrap) = mean(computeInstantaneousSpeed(PSTH_short_pre(:, :, ind_bootstrap))); %(1:ind_t_s_shortest, :, ind_bootstrap)));
    speed_pre_long(ind_bootstrap) = mean(computeInstantaneousSpeed(PSTH_long_pre(:, :, ind_bootstrap))); %(1:ind_t_s_shortest, :, ind_bootstrap)));
    speed_post_long(ind_bootstrap) = mean(computeInstantaneousSpeed(PSTH_long_post{1}(:, :, ind_bootstrap))); %(1:ind_t_s_shortest, :, ind_bootstrap)));
end

% Plot speed versus projection
% Here speed is estimated based the average euclidean distance between
% nearby state
% > Figure 7C
figure
alpha_proj = 1/(mean(proj_short_pre_bootstrap)-mean(proj_long_pre_bootstrap));
alpha_speed = 1/mean(mean(speed_pre_short));
plot(mean(proj_short_pre_bootstrap)*alpha_proj, mean(speed_pre_short)*alpha_speed, 'ro', 'markerfacecolor', 'r', 'markeredgecolor', 'r', 'markersize', 12)
hold on
plot(mean(proj_long_pre_bootstrap)*alpha_proj, mean(speed_pre_long)*alpha_speed, 'bo', 'markerfacecolor', 'b', 'markeredgecolor', 'b', 'markersize', 12)
hold on
plot(mean(proj_long_post_bootstrap)*alpha_proj, mean(speed_post_long)*alpha_speed, 'ko', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 12)
hold on
x = [mean(proj_short_pre_bootstrap) mean(proj_long_pre_bootstrap) mean(proj_long_post_bootstrap)]*alpha_proj;
y = [mean(speed_pre_short) mean(speed_pre_long) mean(speed_post_long)]*alpha_speed;
speed_pre_short_sorted = sort(speed_pre_short);
speed_pre_long_sorted = sort(speed_pre_long);
speed_post_long_sorted = sort(speed_post_long);
proj_pre_short_sorted = sort(proj_short_pre_bootstrap);
proj_pre_long_sorted = sort(proj_long_pre_bootstrap);
proj_post_long_sorted = sort(proj_long_post_bootstrap);
yneg = y-[(speed_pre_short_sorted(1)) (speed_pre_long_sorted(1)) (speed_post_long_sorted(1))]*alpha_speed;
ypos = [(speed_pre_short_sorted(99)) (speed_pre_long_sorted(99)) (speed_post_long_sorted(99))]*alpha_speed-y;
xneg = x-[(proj_pre_short_sorted(1)) (proj_pre_long_sorted(1)) (proj_post_long_sorted(1))]*alpha_proj;
xpos = [(proj_pre_short_sorted(99)) (proj_pre_long_sorted(99)) (proj_post_long_sorted(99))]*alpha_proj-x;
% errorbar(x,y,yneg,ypos,xneg,xpos,'o')
e = errorbar(x(1),y(1),yneg(1),ypos(1),xneg(1),xpos(1));
e.CapSize = 0;
e.LineWidth = 1;
e.Color = 'r';
e = errorbar(x(2),y(2),yneg(2),ypos(2),xneg(2),xpos(2));
e.CapSize = 0;
e.LineWidth = 1;
e.Color = 'b';
e = errorbar(x(3),y(3),yneg(3),ypos(3),xneg(3),xpos(3));
e.CapSize = 0;
e.LineWidth = 1;
e.Color = 'k';
xlabel('Projection (a.u.)')
ylabel('Speed (a.u.)')
fixTicks


% Plot speed versus projection
% Here speed is estimated using the slope of the speed mapping
figure
plot(mean(proj_short_pre_bootstrap), 1/mean(slope_pre_short), 'ro', 'markerfacecolor', 'r', 'markeredgecolor', 'r', 'markersize', 12)
hold on
plot(mean(proj_long_pre_bootstrap), 1/mean(slope_pre_long), 'bo', 'markerfacecolor', 'b', 'markeredgecolor', 'b', 'markersize', 12)
hold on
plot(mean(proj_long_post_bootstrap), 1/mean(slope_post_long), 'ko', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 12)
hold on
x = [mean(proj_short_pre_bootstrap) mean(proj_long_pre_bootstrap) mean(proj_long_post_bootstrap)];
y = [1/mean(slope_pre_short) 1/mean(slope_pre_long) 1/mean(slope_post_long)];
slope_pre_short_sorted = sort(slope_pre_short);
slope_pre_long_sorted = sort(slope_pre_long);
slope_post_long_sorted = sort(slope_post_long);
proj_pre_short_sorted = sort(proj_short_pre_bootstrap);
proj_pre_long_sorted = sort(proj_long_pre_bootstrap);
proj_post_long_sorted = sort(proj_long_post_bootstrap);
yneg = y-[1/mean(slope_pre_short_sorted(99)) 1/mean(slope_pre_long_sorted(99)) 1/mean(slope_post_long_sorted(99))];
ypos = [1/mean(slope_pre_short_sorted(1)) 1/mean(slope_pre_long_sorted(1)) 1/mean(slope_post_long_sorted(1))]-y;
xneg = x-[mean(proj_pre_short_sorted(1)) mean(proj_pre_long_sorted(1)) mean(proj_post_long_sorted(1))];
xpos = [mean(proj_pre_short_sorted(99)) mean(proj_pre_long_sorted(99)) mean(proj_post_long_sorted(99))]-x;
% errorbar(x,y,yneg,ypos,xneg,xpos,'o')
e = errorbar(x(1),y(1),yneg(1),ypos(1),xneg(1),xpos(1));
e.CapSize = 0;
e.LineWidth = 1;
e.Color = 'r';
e = errorbar(x(2),y(2),yneg(2),ypos(2),xneg(2),xpos(2));
e.CapSize = 0;
e.LineWidth = 1;
e.Color = 'b';
e = errorbar(x(3),y(3),yneg(3),ypos(3),xneg(3),xpos(3));
e.CapSize = 0;
e.LineWidth = 1;
e.Color = 'k';
xlabel('Projection (a.u.)')
ylabel('Speed (a.u.)')
fixTicks


%% FUNCTIONS

function [dataTensor_clean, t_p_clean, t_s_clean, id_left_clean, id_short_clean, id_pre_clean, id_trial_clean] = cleanDataTensor_2prior(dataTensor, t_p, t_s, id_left, id_short, id_pre, id_trial)
% [dataTensor_clean, t_p_clean, t_s_clean, id_left_clean, id_trial_clean] = cleanDataTensor(dataTensor, t_p, t_s, id_left, id_short, id_pre, id_trial)
% removes the trial entries in the original data tensor corresponding to
% outlier trials 
%
% INPUT: 
% - dataTensor: 3D matrix (time points x neurons x trials)
% - t_p: produced intervals (trials x 1) 
% - t_s: sample intervals (trials x 1) 
% - id_left: information about target location (trials x 1)
% - id_trial: identity of trial from original data (trials x 1)
%
% OUTPUT: 
% - dataTensor_clean: 3D matrix (time points x neurons x trials) where
% trial entries asssociated with outliers have been removed from the tensor
% - t_p_clean: clean produced intervals (trials x 1) 
% - t_s_clean: clean sample intervals (trials x 1) 
% - id_left_clean: clean information about target location (trials x 1)
% - id_trial_clean: clean identity of trial from original data (trials x 1)

    [t_p_clean, t_s_clean, id_nooutlier] = removeOutlierBehav(t_p, t_s);
    dataTensor_clean = dataTensor(:, :, id_nooutlier);
    id_left_clean = id_left(id_nooutlier);
    id_trial_clean = id_trial(id_nooutlier);
    id_pre_clean = id_pre(id_nooutlier);
    id_short_clean = id_short(id_nooutlier);
end

function [t_p_out, t_s_out, id_nooutlier] = removeOutlierBehav(t_p_in, t_s_in)
% [t_p_out, t_s_out, id_nooutlier] = removeOutlierBehav(t_p_in, t_s_in)
% ALWAYS MAKE SURE TRIALS WITH tp<0 HAVE BEEN REMOVED BEFORE PASSING TO FUNCTION 

% removing outliers in a t_s specific fashion
% [t_p_out, t_s_out]=removeOutlierBehav(t_p_in, t_s_in)
% input: t_p (Nx1) and t_s (Nx1)
% output: t_p_clean (Mx1) and associated t_s_clean (Mx1) (where M<N is the 
% number of clean trials), id_noutlier (Nx1) is 1 if tp was not outlier

%     % First remove t_p that may be negative or equal to 0
%     t_s_in = t_s_in(t_p_in>0); 
%     t_p_in = t_p_in(t_p_in>0);
    
    id_nooutlier = zeros(length(t_p_in), 1);
    
    t_s_unique = unique(t_s_in);
    nb_t_s_unique = length(t_s_unique);
    
    t_p_out = [];
    t_s_out = [];
    
    for ind_t_s = 1:nb_t_s_unique
        t_s = t_s_unique(ind_t_s);
        ind_t_p_per_t_s = find(t_s_in==t_s);
        [~, id_clean] = removeOutlier(t_p_in(ind_t_p_per_t_s),  0, [0 3*t_s]);
        id_nooutlier(ind_t_p_per_t_s(id_clean)) = 1;
    end
    
    id_nooutlier = logical(id_nooutlier);
    t_p_out = t_p_in(id_nooutlier);
    t_s_out = t_s_in(id_nooutlier);
end

function [dataWOout,id,pOut,varargout]=removeOutlier(data,nSD,varargin)
% removing outliers
% [dataWOout,id,pOut,varargout]=removeOutlier(data,nSD,varargin)
% input: data [n x 1], nSD for criteria of SD
% output: data without outlier, id to indicate not outlier in the original
% data, pOut for percetage of outliers

    if nSD>0 % simple SD-based outlier removeal
        idNN=(~isnan(data)); % removing NaN first
        idNO=abs(data(idNN)-mean(data(idNN)))<nSD*std(data(idNN));
        id=zeros(length(data),1); id(idNN)=idNO; id=logical(id);
        pOut=(length(data)-nnz(id))/length(data)*100;
        dataWOout=data(id);
        varargout(1)={data(~id)};

    else % use fitting mixture(w*U(min(tp),max(tp))+(1-w)*N(u,s^2))
        % 3SD removel to estimate initial u,s
        nSD=3;
        idNN=(~isnan(data)); % removing NaN first
        idNO=abs(data(idNN)-mean(data(idNN)))<nSD*std(data(idNN));
        id=zeros(length(data),1); id(idNN)=idNO; id=logical(id);
        pOut=(length(data)-nnz(id))/length(data)*100;
        dataWOout=data(id);

        if length(varargin)==1 % uniform support
            uMin=varargin{1}(1);uMax=varargin{1}(2);
        elseif isempty(varargin)
            uMin=min(data); uMax=max(data);
        end

        if length(data)>15
            u0=mean(dataWOout);
            s0=std(dataWOout);
            w0=pOut/100;
            p0=[w0 u0 s0];
            p0(p0==[0 0 0])=realmin; % to make sure
            if p0(1)==1, p0(1)=1-realmin; end;

            % 3 free parameters:
            % w as lapse rate
            % u, s for normal distribution
            % for now, use [0,max(tp)] as support of U (alternative: min and max of obsered tp, [0, 3*ts] or even fittable?)
            likeMixUG=@(t,w,u,s) (w/(uMax-uMin)+(1-w)*normpdf(t,u,s)); % or 0
            opts=statset('FunValCheck','on','MaxFunEvals',1000*length(p0),'MaxIter',1000*length(p0)); % ,...
            % 'Robust','on','WgtFun','logistic','Display','iter');
    %         if sum(p0<=[0 0 0])|sum(p0>=[1 Inf Inf])
    %             disp('');
    %         end
            [phat,pci]=mle(data,'pdf',likeMixUG,...
                'start',p0,'lowerbound',[0 0 0],'upperbound',[1 Inf Inf],...
                'options',opts);
            u=phat(2); s=phat(3);
            %     disp(phat);
            pOut=phat(1)*100;

            % determine outlier based on likelihood
            L=normpdf(data,u,s);
            id=1/(uMax-uMin)<L; % max(data)-min(data))<L; % min(data) or 0
            dataWOout=data(id);
            varargout(1)={data(~id)};
        else
            varargout={[]};
        end

    end
end

function [PSTH_blocked, edges_block, nTrials] = generateBlockedTrialAvgPSTHwithBootstrap( dataTensor, id_left, wbin, wkernel, nb_blocks, nb_trials_in_block, subsample, nbBootstraps)
% PSTH_blocked = generateBlockedTrialAvgPSTH( dataTensor, t_p, t_s, wbin, wkernel, nb_blocks, nb_trials_in_block)
% generates PSTH in blocks of trials but by randomly selecting trials (with
% replacement) using bootstrap
%
%
% INPUT:
% - dataTensor: 3D matrix (time points x neurons x trials)
% - id_left: info about target location (trials x 1)
% - wbin: size of window (in ms) used to bin the data
% - wkernel: size of window (in ms) used to smooth the data
% - nb_blocks: nb of blocks of trials you want to divide your tensor into
% if nb_blocks = 0, then nb_trials_in_block (see below) becomes relevant,
% if nb_blocks = 1, then there is no block structure created
% if nb_blocks > 1, then there is a block structure created
% - nb_trials_in_block: if nb_blocks = 0, this parameter defines an
% approximate number of trials you want to have in each of your blocks of
% trials
% - subsample: if true, trials within each block are subsampled to match the
% number of left and right target trials

% OUTPUT:
% - PSTH_blocked: cell structure, each entry containing a 3D tensor (time
% points x neurons x nb bootstraps) for a given block
% - edges_block: demarcate the edges of the blocks

    PSTH_blocked = [];
    nb_trials = size(dataTensor, 3);
    if nb_blocks==0 % it means the nb of trials in each block post-transition was specified instead
        if nb_trials_in_block>0
            nb_blocks_init = floor(nb_trials/nb_trials_in_block); % initial guess for the nb of blocks
            edges_block = [1 ceil(quantile(1:nb_trials, nb_blocks_init)) nb_trials]; % roughly equal nb of trials in each block
        else
            disp('nb_trials_in_block MUST be positive')
        end
    elseif nb_blocks==1 % it means there is NO block structure to create
        edges_block = [1 nb_trials];
    elseif nb_blocks==2 % split post in half
        edges_block = [1 ceil(nb_trials/2) nb_trials];
    else % it means there is a block structure to create
        edges_block = [1 ceil(quantile(1:nb_trials, nb_blocks-1)) nb_trials]; % roughly equal nb of trials in each block
    end
    for ind_block = 1:length(edges_block)-1
        id_trial_in_block = ismember(1:nb_trials, (edges_block(ind_block):edges_block(ind_block+1)))';
        if subsample % if want to match number of left and right target trials
            nb_trials_left_in_block = nnz(id_left(id_trial_in_block));
            nb_trials_right_in_block = nnz(~id_left(id_trial_in_block));
            diff_nb_trials_in_block = nb_trials_left_in_block-nb_trials_right_in_block;
            if diff_nb_trials_in_block<0 % if there were fewer left trials in the block
                ind_trials_right_in_block = find(logical(~id_left.*id_trial_in_block));
                nb_trials2exclude = abs(diff_nb_trials_in_block);
                id_trial_in_block(datasample(ind_trials_right_in_block, nb_trials2exclude, 'Replace', false)) = 0; % randomly exclude right trials in excess
            elseif diff_nb_trials_in_block>0 % if there were fewer right trials in the block
                ind_trials_left_in_block = find(logical(id_left.*id_trial_in_block));
                nb_trials2exclude = abs(diff_nb_trials_in_block);
                id_trial_in_block(datasample(ind_trials_left_in_block, nb_trials2exclude, 'Replace', false)) = 0; % randomly exclude left trials in excess
            end
        end
        ind_trial_in_block = find(id_trial_in_block); % get the final index of trials in block
        nTrials(ind_block) = nnz(id_trial_in_block);
        if ~isequal(nnz(id_left(ind_trial_in_block)), nnz(~id_left(ind_trial_in_block)))
            disp('Unequal nb of left/right target trials')
        else
            id_trial_left_in_block = (id_left(ind_trial_in_block));
        end
        for ind_bootstrap = 1:nbBootstraps
            ind_trial_left_in_block = (ind_trial_in_block(id_trial_left_in_block));
            ind_trial_right_in_block = (ind_trial_in_block(~id_trial_left_in_block));
            PSTH_blocked{ind_block}(:, :, ind_bootstrap) = generateTrialAvgPSTH(dataTensor(:, :, ...
                        [datasample(ind_trial_left_in_block, length(ind_trial_left_in_block)) datasample(ind_trial_right_in_block, length(ind_trial_right_in_block))]), wbin, wkernel); % compute PSTH using clean trials only
        end
    end
end

function [PSTH_smooth, PSTH_raw] = generateTrialAvgPSTH(dataTensor, wbin, wkernel)
% [PSTH_smooth, PSTH_raw] = generateTrialAvgPSTH(dataTensor, wbin, wkernel)
% computes raw and smoothed trial-averaged PSTHs starting from the data tensor
%
% INPUT:
% - dataTensor: 3D matrix (time points x neurons x trials)
% - wbin: size of window (in ms) used to bin the data
% - wkernel: size of window (in ms) used to smooth the data
%
% OUTPUT:
% - PSTH_smooth: 2D matrix (time points x neurons) of smoothed firing rates
% - PSTH_raw: 2D matrix (time points x neurons) of raw firing rates

    PSTH_raw = nanmean(dataTensor, 3)/(1e-3*wbin);
    PSTH_smooth = smoother(PSTH_raw', wkernel, wbin)';
    
end

function yOut = smoother(yIn, kernSD, stepSize, varargin)
%
% yOut = smoother(yIn, kernSD, stepSize)
%
% Gaussian kernel smoothing of data across time.
%
% INPUTS:
%
% yIn      - input data (yDim x T)
% kernSD   - standard deviation of Gaussian kernel, in msec
% stepSize - time between 2 consecutive datapoints in yIn, in msec
%
% OUTPUT:
%
% yOut     - smoothed version of yIn (yDim x T)
%
% OPTIONAL ARGUMENT:
%
% causal   - logical indicating whether temporal smoothing should
%            include only past data (true) or all data (false)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

% Aug 21, 2011: Added option for causal smoothing

      causal = false;
      assignopts(who, varargin);

      if (kernSD == 0) || (size(yIn, 2)==1)
        yOut = yIn;
        return
      end

      % Filter half length
      % Go 3 standard deviations out
      fltHL = ceil(3 * kernSD / stepSize);

      % Length of flt is 2*fltHL + 1
      flt = normpdf(-fltHL*stepSize : stepSize : fltHL*stepSize, 0, kernSD);

      if causal
        flt(1:fltHL) = 0;
      end

      [yDim, T] = size(yIn);
      yOut      = nan(yDim, T);

      % Normalize by sum of filter taps actually used
      nm = conv(flt, ones(1, T));

      for i = 1:yDim
        ys = conv(flt, yIn(i,:)) ./ nm;
        % Cut off edges so that result of convolution is same length 
        % as original data
        yOut(i,:) = ys(fltHL+1:end-fltHL);
      end

end


function [vec_min, slope, onset, offset] = computeSpeedMapping_fromReady(PSTH_ref, PSTH_test, ind_offset)
% Compute the mapping between t_ref and t_test to measure relative speed
    
    % handle case where offset post ready is zero
    if ind_offset==0
        ind_offset=1;
    end
    % In case want to compute the speed starting post ready
    PSTH_ref = PSTH_ref(ind_offset:end, :);
    PSTH_test = PSTH_test(ind_offset:end, :);

    ind_bin_max_ref = size(PSTH_ref, 1);
    ind_bin_max_test = size(PSTH_test, 1);
    edges = (1:ind_bin_max_test)';
    % Find mapping between short and long
    map = [];
    for ind_ref = 1:ind_bin_max_ref % for every time point on the ref
        for ind_test = 1:ind_bin_max_test % iterate over time on the test

            % compute the distance traveled along short and long for t_short
            % and t_long (starting from the origin, i.e., ready)
            normspeed_ref = sum(sqrt(sum(diff(PSTH_ref(1:ind_ref, :), 1).^2, 2)));
            normspeed_test = sum(sqrt(sum(diff(PSTH_test(1:ind_test, :), 1).^2, 2)));

            % compute the log ratio of those distances (0 is perfect match)
            map(ind_ref, ind_test) = log(normspeed_ref/normspeed_test);
        end
    end

    % put zeros for t_ref = 0
    map(1, :)=zeros(1, size(map, 2));

    % Handle degenerate cases
    for ind_ref = 1:size(map, 1)
        [~, ind_min_test] = min(abs(map(ind_ref, :)));
        if ind_ref>1 && ind_min_test==1 % this is in case there is a near zero value for t_test = 0 (which doesn't make sense to keep)
            [~, ind_min_test] = min(abs(map(ind_ref, 2:end))); % exclude t_test = 0
            vec_min(ind_ref)=ind_min_test+1;
        else
            vec_min(ind_ref)=ind_min_test;
        end
    end

    % Compute when t_ref and t_test start diverging and use this as starting
    % point for interpolation
    [~, onset] = find(abs(vec_min-(1:length(vec_min))), 1);
    if isempty(onset)
        onset=1;
    end

    % Compute when mapping starts saturating and use this as ending
    % point for interpolation
    offset = find(vec_min==ind_bin_max_test, 1);
    if isempty(offset)
        offset=ind_bin_max_ref;
    end

    fun_mean = @(x) sum(((x(1)*edges(onset:offset)+x(2))-edges(vec_min(onset:offset))).^2);
    x_min = fminsearch(fun_mean, [1 0]);
    slope = x_min(1);

end

function speed = computeInstantaneousSpeed(PSTH)
% Compute the speed as sqrt of squared difference of firing rate in consecutive
% bins
% PSTH [time x neurons]

    speed = sqrt(sum(diff(PSTH, 1).^2, 2));
    
end
