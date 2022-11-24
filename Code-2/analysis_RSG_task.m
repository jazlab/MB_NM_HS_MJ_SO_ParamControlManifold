% This code is used to generate the ephys results from:
% 
% - Figure 6A
% - Figure 6B
% - Figure 6C
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

% Specify the monkey (G versus H) and condition type (Eye versus Hand)
% (Left versus Right) to be analyzed
id_monkey = 'G'; % 'H' for monkey H, 'G' for monkey G
id_eye = true; % always true since we only analyze Eye trials
id_left = false; % false for Right target trials

% Specify fixed parameters for the analysis
wbin = 20; % bin size
t_max_short = 800; % max interval from short prior
t_max_long = 1200; % max interval from long prior
buffer_pre_ready = 200; % value of buffer pre Ready

t_s_unique_short = 480:80:800; % range of ts values for Short
t_s_unique_long = 800:100:1200; % range of ts values for Long
t_short = (0:wbin:t_max_short)'; % discretized time vector for Short
t_long = (0:wbin:t_max_long)'; % discretized time vector for Long
% find indices in time vector corresponding to each ts value
ind_t_s_unique_short = find(ismember(t_short, t_s_unique_short));
ind_t_s_unique_long = find(ismember(t_long, t_s_unique_long));

% Load data
load(['../Data/' id_monkey '_2prior_ReadyM200ms-Set_bin20ms_bootstrap'])

% Use the dataset corresponding to the desired condition
if id_eye
    if id_left
        PSTH_short = PSTH_left_eye_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_left_eye_long(buffer_pre_ready/wbin:end, :, :);
    else
        PSTH_short = PSTH_right_eye_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_right_eye_long(buffer_pre_ready/wbin:end, :, :);
    end
else
    if id_left
        PSTH_short = PSTH_left_hand_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_left_hand_long(buffer_pre_ready/wbin:end, :, :);
    else
        PSTH_short = PSTH_right_hand_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_right_hand_long(buffer_pre_ready/wbin:end, :, :);
    end
end

% First extract neuron_id from session data
neuron_ids = [];
for iSession=1:length(tag_all_neurons)
    neuron_ids = [neuron_ids; iSession*ones(size(tag_all_neurons{iSession})) tag_all_neurons{iSession}];
end

% Remove neurons with low firing rate (< 1 spk/s)
neurons2keep_short = find( logical( (mean(mean(PSTH_short, 3), 1)>1) .* (~isnan(mean(mean(PSTH_short, 3), 1)))) );
neurons2keep_long = find( logical( (mean(mean(PSTH_long, 3), 1)>1) .* (~isnan(mean(mean(PSTH_long, 3), 1)))) );
neurons2keep = intersect(neurons2keep_short, neurons2keep_long);
PSTH_short = PSTH_short(:, neurons2keep, :);
PSTH_long = PSTH_long(:, neurons2keep, :);
neuron_ids = neuron_ids(neurons2keep, :);

% Plot trajectories in top 3 PC space
% > Figure 6A and 6B (left panels)
[score_short, score_long, explained] = plotPCtrajs3D(mean(PSTH_short, 3), mean(PSTH_long, 3), ind_t_s_unique_short, ind_t_s_unique_long);

% Plot the speed mapping between short and long (with control)
% > Figure 6C (right panel)
plotSpeedMapping(PSTH_short, PSTH_long, t_long);

% Plot the distance between trajectories after removing the ctxt dimension
% or a random dimension (control)
% > Figure 6B (right panel)
plotDistProjOutCtxt(PSTH_short, PSTH_long)

% Plot the speed of trajectories versus the projection onto the ctxt dim
% > Figure 6C (left panel)
plotProjxSpeed(PSTH_short, PSTH_long)

% Plot the average angle between neural trajectories 
% > Figure 6A (right panel)
plotAngleTraj(id_monkey, id_left, id_eye, buffer_pre_ready, wbin)

%% FUNCTIONS

% Plot neural trajectories in the space spanned by top 3 PCs
function [score_short, score_long, explained, mat_coeff] = plotPCtrajs3D(PSTH_short, PSTH_long, ind_t_s_unique_short, ind_t_s_unique_long)
    
    nDims = 3; % nb of PCs to keep
    PSTH_concat = [PSTH_short; PSTH_long];
    data_mean = mean(PSTH_concat, 1);
    [coeff, ~, ~, ~, explained] = pca(PSTH_concat);
    mat_coeff = coeff(:, 1:nDims);
    score_short = (PSTH_short-data_mean)*mat_coeff;
    score_long = (PSTH_long-data_mean)*mat_coeff;
    
    figure; 
    plot3(score_short(:, 1), score_short(:, 2), score_short(:, 3), 'r-', 'markersize', 12)
    hold on
    plot3(score_short(1, 1), score_short(1, 2), score_short(1, 3), 'rs', 'markerfacecolor', 'r', 'markersize', 8)
    hold on
    plot3(score_short(ind_t_s_unique_short, 1), score_short(ind_t_s_unique_short, 2), score_short(ind_t_s_unique_short, 3), 'ro', 'markerfacecolor', 'r', 'markersize', 8)
    hold on
    plot3(score_long(:, 1), score_long(:, 2), score_long(:, 3), 'b-', 'markersize', 12)
    hold on
    plot3(score_long(1, 1), score_long(1, 2), score_long(1, 3), 'bs', 'markerfacecolor', 'b', 'markersize', 8)
    hold on
    plot3(score_long(ind_t_s_unique_long, 1), score_long(ind_t_s_unique_long, 2), score_long(ind_t_s_unique_long, 3), 'bo', 'markerfacecolor', 'b', 'markersize', 8)
    grid on
    xlabel('PC1')
    ylabel('PC2')
    zlabel('PC3')
    
%     % Uncomment to plot variance explained
%     % Plot scree plot
%     nPlotPCs = 10;
%     figure; plot(cumsum(explained(1:nPlotPCs)), 'k.', 'markersize', 36)
%     hold on
%     plot(1:nPlotPCs, 100*ones(size(1:nPlotPCs)), 'k--', 'linewidth', 3)
%     axis([0.8 nPlotPCs 0 105]);
%     xticks(0:nPlotPCs)
%     yticks(0:20:100)
%     xlabel('# PCs')
%     ylabel('% Var')
%     fixTicks
    
end

% Plot the mapping between trajectories to assess speed differences
function [] = plotSpeedMapping(PSTH_short, PSTH_long, t_long)

    nBootstrap = size(PSTH_short, 3);
    for iBootstrap = 1:nBootstrap
        ind_t_test_data = trajMapping(nanmean(PSTH_long, 3), PSTH_short(:, :, iBootstrap), 0);
        ind_t_test_null = trajMapping(nanmean(PSTH_long, 3), PSTH_long(:, :, iBootstrap), 0);
        t_test_data_bootstrap(iBootstrap, :) = t_long(ind_t_test_data);
        t_test_null_bootstrap(iBootstrap, :) = t_long(ind_t_test_null);
    end
    t_test_data_bootstrap = sort(t_test_data_bootstrap);
    t_test_null_bootstrap = sort(t_test_null_bootstrap);

    figure; 
    p(1) = plot(t_long, mean(t_test_data_bootstrap), 'r-');
    hold on
    ciplot(t_test_data_bootstrap(1, :), t_test_data_bootstrap(99, :), t_long, 'r-')
    hold on
    p(2) = plot(t_long, mean(t_test_null_bootstrap), 'b--');
    hold on
    ciplot(t_test_null_bootstrap(1, :), t_test_null_bootstrap(99, :), t_long, 'b--')
    plot(t_long, t_long, 'k--', 'linewidth', 2)
    xlabel('t_{ref} (ms)')
    ylabel('t_{test} (ms)')
    legend(p, {'diff. cxt', 'same ctx'}, 'location', 'southeast')
    fixTicks
    axis([0 800 0 800])

end

% Compute the mapping between ref and test to measure speed differences
function [ind_t_test, varexp] = trajMapping(PSTH_ref, PSTH_test, nDim) 
    
    varexp=nan;
    if nDim~=0
        [PSTH_ref, PSTH_test, varexp] = applyPCA(PSTH_ref, PSTH_test, nDim);
    end
    % Compute speed difference using kinet and using long condition as ref
    ind_t_test=[];
    for ind_t_ref = 1:size(PSTH_ref, 1) % for all ref times
        x_ref = PSTH_ref(ind_t_ref, :); % ref state short (1xN)
        dist = sum((PSTH_test-x_ref).^2, 2); % Tx1
        [~, ind_x_closest] = min(dist);
        ind_t_test(ind_t_ref) = ind_x_closest;
    end
end

% Plot distance between trajectories after removing the ctxt dimension
function [] = plotDistProjOutCtxt(PSTH_short, PSTH_long)
    
    dist_bootstrap = [];
    nDim = 3;
    nBootstrap = size(PSTH_short, 3);
    for iBootstrap = 1:nBootstrap
        dist_data_bootstrap(:, iBootstrap) = computeDistAfterProjOutCtxtDim(nanmean(PSTH_long, 3), PSTH_short(:, :, iBootstrap), nDim);
        dist_null_bootstrap(:, iBootstrap) = computeDistAfterProjOutRandDim(nanmean(PSTH_long, 3), PSTH_short(:, :, iBootstrap), nDim);
        dist_bound_bootstrap(:, iBootstrap) = computeDistAfterProjOutCtxtDim(nanmean(PSTH_short, 3), PSTH_short(:, :, iBootstrap), nDim);
    end

    figure;
    alpha_dist = 1/mean(mean(dist_null_bootstrap));
    plot([1.9 2.1], [mean(mean(dist_bound_bootstrap)) mean(mean(dist_null_bootstrap))]*alpha_dist, 'k--', 'linewidth', 1.5)
    hold on
    plot([0.9 1.1], [mean(mean(dist_bound_bootstrap)) mean(mean(dist_data_bootstrap))]*alpha_dist, 'k--', 'linewidth', 1.5)
    plot(0.9, mean(mean(dist_bound_bootstrap))*alpha_dist, 'o', 'markeredgecolor', 'k', 'markerfacecolor', 'b', 'markersize', 10)% 0.2, 'FaceColor', 'w', 'EdgeColor', 'k')
    hold on
    errorbar(0.9, mean(mean(dist_bound_bootstrap))*alpha_dist, ...
        abs(mean(dist_bound_bootstrap(:, 1))-mean(mean(dist_bound_bootstrap)))*alpha_dist, ...
        abs(mean(dist_bound_bootstrap(:, 99))-mean(mean(dist_bound_bootstrap)))*alpha_dist, 'color', 'b', 'linewidth', 1.5)
    hold on
    plot(1.1, mean(mean(dist_data_bootstrap))*alpha_dist, 'o', 'markeredgecolor', 'k', 'markerfacecolor', 'r', 'markersize', 10)%0.2, 'FaceColor', 'k', 'EdgeColor', 'k')
    hold on
    errorbar(1.1, mean(mean(dist_data_bootstrap))*alpha_dist, ...
        abs(mean(dist_data_bootstrap(:, 1))-mean(mean(dist_data_bootstrap)))*alpha_dist, ...
        abs(mean(dist_data_bootstrap(:, 99))-mean(mean(dist_data_bootstrap)))*alpha_dist, 'color', 'r', 'linewidth', 1.5)
    hold on
    errorbar(1.9, mean(mean(dist_bound_bootstrap))*alpha_dist, ...
        abs(mean(dist_bound_bootstrap(:, 1))-mean(mean(dist_bound_bootstrap)))*alpha_dist, ...
        abs(mean(dist_bound_bootstrap(:, 99))-mean(mean(dist_bound_bootstrap)))*alpha_dist, 'color', 'b', 'linewidth', 1.5)
    hold on
    plot(1.9, mean(mean(dist_bound_bootstrap))*alpha_dist, 'o', 'markeredgecolor', 'b', 'markerfacecolor', 'w', 'markersize', 10)%, 0.2, 'FaceColor', 'w', 'EdgeColor', 'k')
    hold on
    errorbar(2.1, mean(mean(dist_null_bootstrap))*alpha_dist, ...
        abs(mean(dist_null_bootstrap(:, 1))-mean(mean(dist_null_bootstrap)))*alpha_dist, ...
        abs(mean(dist_null_bootstrap(:, 99))-mean(mean(dist_null_bootstrap)))*alpha_dist, 'color', 'r', 'linewidth', 1.5)
    plot(2.1, mean(mean(dist_null_bootstrap))*alpha_dist, 'o', 'markeredgecolor', 'r', 'markerfacecolor', 'w', 'markersize', 10)%, 0.2, 'FaceColor', 'k')
    xlim([0 3])
    xticks([1 2])
    xticklabels({'no ctxt','control'})
    ylabel('Distance (a.u.)')
    fixTicks

end

% Compute distance between trajectories after removing the ctxt dimension
function dist = computeDistAfterProjOutCtxtDim(PSTH_short, PSTH_long, nDim)
    if nDim~=0
        [PSTH_short, PSTH_long] = applyPCA(PSTH_short, PSTH_long, nDim);
    end
    % define the context dimension at each time point of Long
    for ind_t_long = 1:size(PSTH_long, 1) % for all times in Long
            x_long = PSTH_long(ind_t_long, :); % test state Long (1xN)
            dd = sum((PSTH_short-x_long).^2, 2); % Tx1
            [~, ind_x_closest] = min(dd);
            x_short = PSTH_short(ind_x_closest, :); % test state Short (1xN)
            % test vector from ref Long state to ref Short state
            v_loc = (x_short-x_long)'; % Nx1
            v_loc = v_loc/norm(v_loc); % normalized ref vector
            v_all(:, ind_t_long) = v_loc;
    end
    % compute the average context dimension
    v = mean(v_all, 2);
    dist = distProjectOut(PSTH_short, PSTH_long, v);
end

% Compute distance between trajectories after removing the random dimension
function dist = computeDistAfterProjOutRandDim(PSTH_short, PSTH_long, nDim)
    if nDim~=0
        [PSTH_short, PSTH_long] = applyPCA(PSTH_short, PSTH_long, nDim);
    end
    nNeuron = size(PSTH_short, 2);
    dist_all = [];
    for iBootstrap = 1:100
        v = normrnd(0, 1, nNeuron, 1); % Nx1
        v = v/norm(v); % Nx1
        dist_loc = distProjectOut(PSTH_short, PSTH_long, v);
        dist_all = [dist_all dist_loc];
    end
    dist = mean(dist_all, 2);
end

% Apply PCA to concatenated PSTH data
function [score_short, score_long, varexp] = applyPCA(PSTH_short, PSTH_long, nDim)
    
    PSTH_concat = [PSTH_short; PSTH_long];
    data_mean = mean(PSTH_concat, 1);
    [coeff, ~, ~, ~, explained] = pca(PSTH_concat);
    varexp = cumsum(explained(1:nDim));
    mat_coeff = coeff(:, 1:nDim);
    score_short = (PSTH_short-data_mean)*mat_coeff;
    score_long = (PSTH_long-data_mean)*mat_coeff;

end

% Compute distance after projecting out the v dimension
function dist = distProjectOut(PSTH_short, PSTH_long, v)
    PSTH_short_proj = PSTH_short - (PSTH_short*v)*v';
    PSTH_long_proj = PSTH_long - (PSTH_long*v)*v';
    dist = [];
    for ind_t_test_long = 1:size(PSTH_long_proj, 1) % for all times in Long
        x_test_long = PSTH_long_proj(ind_t_test_long, :); % test state Long (1xN)
        d = sum((PSTH_short_proj-x_test_long).^2, 2); % Tx1
        dist(ind_t_test_long) = min(d);
    end
    dist = dist';
end

function plotProjxSpeed(PSTH_short, PSTH_long)
    
    % Define the normalized context dimension at the time of Ready
    ctx_dim = nanmean(PSTH_short(1, :, :), 3)-nanmean(PSTH_long(1, :, :), 3);
    ctx_dim = ctx_dim/norm(ctx_dim);

    % Project activity onto context dimension
    x_center = (nanmean(PSTH_short(1, :, :), 3)+nanmean(PSTH_long(1, :, :), 3))'/2;
    proj_short = ctx_dim*(squeeze(PSTH_short(1, :, :))-x_center);
    proj_long = ctx_dim*(squeeze(PSTH_long(1, :, :))-x_center);

    % Compute the speed for short and long
    for ind_bootstrap = 1:size(PSTH_short, 3)
        speed_short(:, ind_bootstrap) = computeInstantaneousSpeed(PSTH_short(:, :, ind_bootstrap));
        speed_long(:, ind_bootstrap) = computeInstantaneousSpeed(PSTH_long(:, :, ind_bootstrap));
    end

    % Plot average speed versus projection
    figure
    alpha_proj = 1/(mean(proj_short)-mean(proj_long));
    alpha_speed = 1/mean(mean(speed_short));
    errorbar(mean(proj_short)*alpha_proj, mean(mean(speed_short))*alpha_speed, abs(min(mean(speed_short))-mean(mean(speed_short)))*alpha_speed, abs(max(mean(speed_short))-mean(mean(speed_short)))*alpha_speed, abs(min(proj_short)-mean(proj_short))*alpha_proj, abs(mean(proj_short)-max(proj_short))*alpha_proj,  'ro')
    hold on
    errorbar(mean(proj_long)*alpha_proj, mean(mean(speed_long))*alpha_speed, abs(min(mean(speed_long))-mean(mean(speed_long)))*alpha_speed, abs(max(mean(speed_long))-mean(mean(speed_long)))*alpha_speed, abs(min(proj_long)-mean(proj_long))*alpha_proj, abs(mean(proj_long)-max(proj_long))*alpha_proj,  'bo')
    xlabel('projection onto ctx dim (a.u.)')
    ylabel('speed (a.u.)')
    axis([-1 1 0.7 1.2])
    fixTicks

end

function speed = computeInstantaneousSpeed(PSTH)
% Compute the speed as sqrt of squared difference of firing rate in consecutive
% bins
% PSTH [time x neurons]

    speed = sqrt(sum(diff(PSTH, 1).^2, 2));
    
end

function [] = fixTicks()
% fixTicks removes upper and yy ticks, and set the ticks outside the figure
    set(gca, 'TickDir', 'out');
    set(gca, 'box', 'off')
    set(gca, 'FontSize', 14)
end

function [] = plotAngleTraj(id_monkey, id_left, id_eye, buffer_pre_ready, wbin)
    
    % Load data (unsmoothed data to avoid smoothing confound in angle computation)
    load(['../Data/' id_monkey '_2prior_Ready-Set_singleTrial_unsmoothPSTH_bin20ms'])

    % Use the dataset corresponding to the desired condition
    if id_eye
        if id_left
            PSTH_short = PSTH_left_eye_short(buffer_pre_ready/wbin:end, :, :);
            PSTH_long = PSTH_left_eye_long(buffer_pre_ready/wbin:end, :, :);
            PSTH_short_null = PSTH_left_eye_short_null(buffer_pre_ready/wbin:end, :, :);
            PSTH_long_null = PSTH_left_eye_long_null(buffer_pre_ready/wbin:end, :, :);
        else
            PSTH_short = PSTH_right_eye_short(buffer_pre_ready/wbin:end, :, :);
            PSTH_long = PSTH_right_eye_long(buffer_pre_ready/wbin:end, :, :);
            PSTH_short_null = PSTH_right_eye_short_null(buffer_pre_ready/wbin:end, :, :);
            PSTH_long_null = PSTH_right_eye_long_null(buffer_pre_ready/wbin:end, :, :);
        end
    else
        if id_left
            PSTH_short = PSTH_left_hand_short(buffer_pre_ready/wbin:end, :, :);
            PSTH_long = PSTH_left_hand_long(buffer_pre_ready/wbin:end, :, :);
            PSTH_short_null = PSTH_left_hand_short_null(buffer_pre_ready/wbin:end, :, :);
            PSTH_long_null = PSTH_left_hand_long_null(buffer_pre_ready/wbin:end, :, :);
        else
            PSTH_short = PSTH_right_hand_short(buffer_pre_ready/wbin:end, :, :);
            PSTH_long = PSTH_right_hand_long(buffer_pre_ready/wbin:end, :, :);
            PSTH_short_null = PSTH_right_hand_short_null(buffer_pre_ready/wbin:end, :, :);
            PSTH_long_null = PSTH_right_hand_long_null(buffer_pre_ready/wbin:end, :, :);
        end
    end

    % Compute the angle between the trajectories
    nDim = 6; % nb of dimension to compute angles in lower-dimensional space
    angle_ref_test = angleBtwTraj(nanmean(PSTH_short, 3), nanmean(PSTH_long, 3), nDim);
    angle_ref_test_null = angleBtwTraj(nanmean(PSTH_short_null, 3), nanmean(PSTH_long_null, 3), nDim);

    % Plot angle versus ttest-tref
    maxDelay = 400; % ms
    for ind_t_test = 1:(maxDelay/wbin)%(angle_ref_test_left_eye, 2)
        mean_angle(ind_t_test) = mean(diag(angle_ref_test, ind_t_test));
        mean_angle_null(ind_t_test) = mean(diag(angle_ref_test_null, ind_t_test));
    end

    % Compute average angle between trajectories
    mean_angle_acrossDelays = mean(mean_angle);
    sem_angle_acrossDelays = std(mean_angle)/sqrt(size(mean_angle, 1));
    mean_angle_acrossDelays_null = mean(mean_angle_null);
    sem_angle_acrossDelays_null = std(mean_angle_null)/sqrt(size(mean_angle_null, 1));

    % Plot the results
    % Figure 6A (right panel)
    figure
    errorbar(0.5, mean_angle_acrossDelays, sem_angle_acrossDelays, 'color', 'k', 'linewidth', 1.5, 'capsize', 0)
    hold on
    plot(0.5, mean_angle_acrossDelays, 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'markersize', 9)
    hold on
    errorbar(1.5, mean_angle_acrossDelays_null, sem_angle_acrossDelays_null, 'color', 'k', 'linewidth', 1.5, 'capsize', 0)
    hold on
    plot(1.5, mean_angle_acrossDelays_null, 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'markersize', 9)
    hold on
    plot(0:0.1:2, 90*ones(size(0:0.1:2)), 'k--')
    axis([0 2 0 100])
    xticks([0.5 1.5])
    xticklabels({'2-ctxt','control'})
    yticks([0 45 90])
    yticklabels({'0','\pi/4','\pi/2'})
    fixTicks

end

% Compute angle between trajectories
function angle_ref_test = angleBtwTraj(PSTH_short, PSTH_long, nDim) 
    
    if nDim~=0
        [PSTH_short, PSTH_long] = applyPCA(PSTH_short, PSTH_long, nDim);
    end
    
    for ind_t_ref_long = 1:size(PSTH_long, 1) % for all times in Long
        x_ref_long = PSTH_long(ind_t_ref_long, :); % ref state Long (1xN)
        dist = sum((mean(PSTH_short, 3)-x_ref_long).^2, 2); % Tx1
        [~, ind_x_closest] = min(dist);
        x_ref_short = PSTH_short(ind_x_closest, :); % ref state Short (1xN)
        % ref vector from ref Long state to ref Short state
        v_ref = (x_ref_short-x_ref_long)';
        v_ref = v_ref/norm(v_ref); % normalized ref vector

        for ind_t_test_long = 1:size(PSTH_long, 1) % for all times in Long
            x_test_long = PSTH_long(ind_t_test_long, :); % test state Long (1xN)
            dist = sum((PSTH_short-x_test_long).^2, 2); % Tx1
            [~, ind_x_closest] = min(dist);
            x_test_short = PSTH_short(ind_x_closest, :); % test state Short (1xN)
            % test vector from ref Long state to ref Short state
            v_test = (x_test_short-x_test_long)';
            v_test = v_test/norm(v_test); % normalized ref vector
            % compute angle between ref and test vectors
            angle_ref_test(ind_t_ref_long, ind_t_test_long) = real(acos(v_ref'*v_test))*180/pi;
        end
    end
 
end