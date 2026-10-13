%% ========================================================================
%% Hysteresis Test Across Many Communities: Bidirectional Leakage Sweep
%% ========================================================================
% Extends the single-community hysteresis test to n_communities
% independently-drawn random communities, so we can report how PREVALENT
% hysteresis is in this model class, not just whether one example shows it.
%
% For each community:
%   1) Draw C, W, D, and a reference resource level R_ref, all randomly.
%      A reference leakage l_ref (also random, NOT fixed to the sweep
%      midpoint) is used once to set mortality m so that growth is zero
%      at R_ref -- the one quantity constructed to make coexistence
%      plausible. External supply S is set from the depletion and
%      recycling expected at R_ref (using a unit reference abundance),
%      floored at 0.05. m and S are then held FIXED -- only l is varied
%      for the rest of that community's test.
%   2) UP-SWEEP: continuation from low l to high l, each step starting
%      from where the previous step settled.
%   3) DOWN-SWEEP: continuation from high l back down to low l.
%   4) Compare the two sweeps at matching l values. Flag this community
%      as showing hysteresis if the settled abundances differ beyond
%      tolerance, OR the dominance pattern (who's present) differs,
%      anywhere along the sweep.
%
% NOTE ON RUNTIME: n_communities x n_steps x 2 sweep directions x ODE
% integration is a lot of work. Defaults below (500 communities x 25
% steps) already represent a substantial run. Reduce n_communities for
% a quick test before committing to the full run.
%% ========================================================================
clear; close all; clc;
rng('shuffle');

%% ============================ CONFIGURATION ============================
cfg.N_sp    = 2;
cfg.R_res   = 2;
cfg.l_low   = 0.05;
cfg.l_high  = 0.95;
cfg.n_steps = 25;          % l-grid resolution per sweep direction
cfg.T_step  = 150;         % integration time per step (let it settle)
cfg.ext_thresh = 1e-5;
cfg.diff_tol   = 1e-3;     % abundance difference counted as "diverged"
cfg.n_communities = 500;
cfg.rhs_tol = 1e-6;        % ||RHS|| below this counts as "settled" at a step
cfg.max_extend_attempts = 5; % how many times to double T_step if not settled
odeopt = odeset('NonNegative', 1:(cfg.N_sp+cfg.R_res), ...
                 'RelTol', 1e-9, 'AbsTol', 1e-11);

%% ==================== RUN ACROSS MANY COMMUNITIES ========================
hysteresis_flag   = false(cfg.n_communities, 1);
max_n1_diff_all   = nan(cfg.n_communities, 1);
max_n2_diff_all   = nan(cfg.n_communities, 1);
n_dom_mismatch_all = nan(cfg.n_communities, 1);
l_onset_all  = nan(cfg.n_communities, 1);  % lowest l where genuine (converged) divergence appears
l_offset_all = nan(cfg.n_communities, 1);  % highest l where genuine (converged) divergence appears
all_hyst_l_values = [];                    % pooled, across every community, every l where genuine divergence occurred
n_unconverged_steps_total = 0;             % diagnostic: how many steps failed to settle even after extension
n_steps_total = 0;

% Keep full traces for a few example communities so we have something
% concrete to plot: first hysteresis-positive found, plus a "no hysteresis"
% example. For the negative example we specifically prefer one where EVERY
% step in both directions genuinely converged -- a fully clean null result
% -- over one where the lack of detected divergence might just be because
% an unconverged (untrustworthy) point got excluded from the comparison.
% example_neg_any is kept only as a fallback in case no clean one exists.
example_pos = []; example_neg_clean = []; example_neg_any = [];
all_converged_all = false(cfg.n_communities, 1); % track, per community, whether
                                                   % EVERY step in both sweep
                                                   % directions genuinely settled

for c_idx = 1:cfg.n_communities
    if mod(c_idx, 25) == 0
        fprintf('Community %4d / %4d...\n', c_idx, cfg.n_communities);
    end

    [base, x0_ref, l_ref] = draw_baseline_community(cfg); %#ok<ASGLU>

    l_grid_up = linspace(cfg.l_low, cfg.l_high, cfg.n_steps);
    [N1_up, N2_up, dom_up, conv_up] = run_sweep(l_grid_up, x0_ref, base, cfg, odeopt);

    l_grid_down = fliplr(l_grid_up);
    [N1_down, N2_down, dom_down, conv_down] = run_sweep(l_grid_down, x0_ref, base, cfg, odeopt);
    [l_grid_down_sorted, sort_idx] = sort(l_grid_down);
    N1_down_sorted = N1_down(sort_idx);
    N2_down_sorted = N2_down(sort_idx);
    dom_down_sorted = dom_down(sort_idx, :);
    conv_down_sorted = conv_down(sort_idx);

    n_unconverged_steps_total = n_unconverged_steps_total + sum(~conv_up) + sum(~conv_down);
    n_steps_total = n_steps_total + numel(conv_up) + numel(conv_down);
    all_converged_all(c_idx) = all(conv_up) && all(conv_down_sorted);

    n1_diff = abs(N1_up - N1_down_sorted);
    n2_diff = abs(N2_up - N2_down_sorted);
    dom_mismatch = any(dom_up ~= dom_down_sorted, 2);

    max_n1_diff_all(c_idx) = max(n1_diff);
    max_n2_diff_all(c_idx) = max(n2_diff);
    n_dom_mismatch_all(c_idx) = sum(dom_mismatch);

    % Only trust divergence at points where BOTH sweeps genuinely settled.
    % This is the fix: without it, unconverged residual drift can look
    % identical to real direction-dependence.
    both_converged = conv_up(:) & conv_down_sorted(:);
    diverged_raw = (n1_diff > cfg.diff_tol) | (n2_diff > cfg.diff_tol) | dom_mismatch';
    diverged = diverged_raw & both_converged';
    is_hyst = any(diverged);
    hysteresis_flag(c_idx) = is_hyst;

    if is_hyst
        l_hit = l_grid_up(diverged);
        l_onset_all(c_idx)  = min(l_hit);
        l_offset_all(c_idx) = max(l_hit);
        all_hyst_l_values = [all_hyst_l_values, l_hit]; %#ok<AGROW>
    end

    if is_hyst && isempty(example_pos)
        example_pos.l_up = l_grid_up; example_pos.N1_up = N1_up; example_pos.N2_up = N2_up;
        example_pos.l_down = l_grid_down_sorted; example_pos.N1_down = N1_down_sorted; example_pos.N2_down = N2_down_sorted;
        example_pos.l_range = [l_onset_all(c_idx), l_offset_all(c_idx)];
    end
    if ~is_hyst && all_converged_all(c_idx) && isempty(example_neg_clean)
        example_neg_clean.l_up = l_grid_up; example_neg_clean.N1_up = N1_up; example_neg_clean.N2_up = N2_up;
        example_neg_clean.l_down = l_grid_down_sorted; example_neg_clean.N1_down = N1_down_sorted; example_neg_clean.N2_down = N2_down_sorted;
    end
    if ~is_hyst && isempty(example_neg_any)
        example_neg_any.l_up = l_grid_up; example_neg_any.N1_up = N1_up; example_neg_any.N2_up = N2_up;
        example_neg_any.l_down = l_grid_down_sorted; example_neg_any.N1_down = N1_down_sorted; example_neg_any.N2_down = N2_down_sorted;
        example_neg_any.all_converged = all_converged_all(c_idx);
    end
end

% Resolve which "no hysteresis" example to actually plot: the clean one if
% we found one, otherwise fall back to any no-hysteresis community (and say
% so plainly, since that fallback is a weaker illustration).
if ~isempty(example_neg_clean)
    example_neg = example_neg_clean;
    example_neg_is_clean = true;
elseif ~isempty(example_neg_any)
    example_neg = example_neg_any;
    example_neg_is_clean = false;
    fprintf(['\nNOTE: no fully-converged "no hysteresis" example was found among these\n' ...
             '%d communities; the plotted negative example below had at least one\n' ...
             'unconverged (excluded) point, so treat it as a weaker null-result\n' ...
             'illustration -- see the reliability breakdown below.\n'], cfg.n_communities);
else
    example_neg = [];
    example_neg_is_clean = false;
end

%% ==================== SUMMARY ============================================
n_hyst = sum(hysteresis_flag);
frac_hyst = n_hyst / cfg.n_communities;

fprintf('\n=== CONVERGENCE DIAGNOSTIC ===\n');
fprintf('Steps that needed extension beyond the base T_step to settle: check below.\n');
fprintf('Steps that FAILED to settle even after %d doublings (max T = %.0f): %d / %d (%.2f%%)\n', ...
    cfg.max_extend_attempts, cfg.T_step * 2^cfg.max_extend_attempts, ...
    n_unconverged_steps_total, n_steps_total, 100*n_unconverged_steps_total/n_steps_total);
fprintf('These unconverged points are EXCLUDED from the hysteresis comparison below,\n');
fprintf('so the results that follow reflect only genuinely-settled states.\n');

n_no_hyst = cfg.n_communities - n_hyst;
n_no_hyst_clean = sum(~hysteresis_flag & all_converged_all);
n_no_hyst_dirty = n_no_hyst - n_no_hyst_clean;

fprintf('\n=== "NO HYSTERESIS" RELIABILITY BREAKDOWN ===\n');
fprintf('Of the %d communities with no detected hysteresis:\n', n_no_hyst);
fprintf('  %d (%.1f%%) had EVERY step in both directions genuinely converge --\n', ...
    n_no_hyst_clean, 100*n_no_hyst_clean/max(n_no_hyst,1));
fprintf('      a clean, trustworthy null result (single attractor confirmed at\n');
fprintf('      every l tested, nothing excluded from the comparison).\n');
fprintf('  %d (%.1f%%) had at least one unconverged point excluded from the\n', ...
    n_no_hyst_dirty, 100*n_no_hyst_dirty/max(n_no_hyst,1));
fprintf('      comparison -- here "no hysteresis" means no CONFIRMED divergence,\n');
fprintf('      not that up- and down-sweeps agreed everywhere. Some of these may\n');
fprintf('      be masking real divergence at points that never settled (e.g.\n');
fprintf('      unbounded/blown-up trajectories). Treat this subset with caution\n');
fprintf('      and consider re-running them with a longer max integration time.\n');

fprintf('\n=== HYSTERESIS PREVALENCE ACROSS %d COMMUNITIES ===\n', cfg.n_communities);
fprintf('Communities showing hysteresis: %d / %d (%.1f%%)\n', n_hyst, cfg.n_communities, 100*frac_hyst);
fprintf('Mean max|N1_up-N1_down| across all communities: %.5f\n', mean(max_n1_diff_all));
fprintf('Mean max|N2_up-N2_down| across all communities: %.5f\n', mean(max_n2_diff_all));
fprintf('Mean number of l-steps with a dominance mismatch: %.2f / %d\n', ...
    mean(n_dom_mismatch_all), cfg.n_steps);

if n_hyst > 0
    fprintf('\n=== WHERE ALONG THE LEAKAGE AXIS DOES HYSTERESIS OCCUR? ===\n');
    fprintf('(Onset/offset = lowest/highest l at which that community''s up- and\n');
    fprintf(' down-sweeps diverged; a community can diverge over a sub-range of l,\n');
    fprintf(' not necessarily the full [%.2f, %.2f] sweep range.)\n\n', cfg.l_low, cfg.l_high);
    onset_vals  = l_onset_all(hysteresis_flag);
    offset_vals = l_offset_all(hysteresis_flag);
    fprintf('Onset l:  mean=%.3f  median=%.3f  range=[%.3f, %.3f]\n', ...
        mean(onset_vals), median(onset_vals), min(onset_vals), max(onset_vals));
    fprintf('Offset l: mean=%.3f  median=%.3f  range=[%.3f, %.3f]\n', ...
        mean(offset_vals), median(offset_vals), min(offset_vals), max(offset_vals));
    fprintf('Pooled across all divergent points from all communities:\n');
    fprintf('  mean l = %.3f, median l = %.3f\n', mean(all_hyst_l_values), median(all_hyst_l_values));
end

if frac_hyst == 0
    fprintf(['\nNo hysteresis detected in any of the %d communities tested.\n' ...
             'This is consistent with this model class having a unique attracting\n' ...
             'equilibrium for each fixed parameter set -- i.e. no bistability, so\n' ...
             'there is no alternative basin of attraction for a sweep to get trapped\n' ...
             'in. This supports attributing the irreversibility seen in the\n' ...
             'transient-perturbation experiments to the system crossing a genuine\n' ...
             'stability boundary as a parameter changes, not to pre-existing\n' ...
             'multistability at fixed parameters.\n'], cfg.n_communities);
else
    fprintf(['\nHysteresis was detected in a nonzero fraction of communities, indicating\n' ...
             'genuine multistability exists in at least part of this model''s parameter\n' ...
             'space. See Figure 3 for an example trace.\n']);
end

%% ============================ PLOTTING ====================================
figure('Name','Hysteresis prevalence summary','Position',[100 100 500 400]);
bar([frac_hyst, 1-frac_hyst]*100);
set(gca, 'XTickLabel', {'Hysteresis detected','No hysteresis'});
ylabel('% of communities'); ylim([0 100]);
title(sprintf('Hysteresis prevalence across %d communities', cfg.n_communities));
grid on; box off;

figure('Name','Distribution of max abundance divergence','Position',[650 100 900 400]);
subplot(1,2,1);
histogram(max_n1_diff_all, 30, 'FaceColor', [0.20 0.40 0.68], 'EdgeColor', 'none');
xlabel('max |N_1 up - N_1 down|'); ylabel('Count');
title('Species 1 divergence across sweep directions'); box off;
subplot(1,2,2);
histogram(max_n2_diff_all, 30, 'FaceColor', [0.89 0.30 0.29], 'EdgeColor', 'none');
xlabel('max |N_2 up - N_2 down|'); ylabel('Count');
title('Species 2 divergence across sweep directions'); box off;

if n_hyst > 0
    figure('Name','Where hysteresis occurs along the leakage axis','Position',[100 1000 900 400]);
    histogram(all_hyst_l_values, linspace(cfg.l_low, cfg.l_high, cfg.n_steps+1), ...
        'FaceColor', [0.55 0.25 0.65], 'EdgeColor', 'none');
    xlabel('Leakage, l'); ylabel('Number of divergent sweep points (pooled across all communities)');
    title('Where along the leakage axis do up- and down-sweeps diverge?');
    grid on; box off;
end

if ~isempty(example_pos)
    figure('Name','Example: community showing hysteresis','Position',[100 550 900 420]);
    plot(example_pos.l_up, example_pos.N1_up, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 12); hold on;
    plot(example_pos.l_down, example_pos.N1_down, 'r.--', 'LineWidth', 1.5, 'MarkerSize', 12);
    xline(example_pos.l_range(1), 'k:', 'LineWidth', 1.2, 'Label', 'onset');
    xline(example_pos.l_range(2), 'k:', 'LineWidth', 1.2, 'Label', 'offset');
    xlabel('Leakage, l'); ylabel('N_1 at settled state');
    legend({'Up-sweep','Down-sweep'}, 'Location', 'best');
    title(sprintf('Example community WITH detected hysteresis (diverges over l \\in [%.2f, %.2f])', ...
        example_pos.l_range(1), example_pos.l_range(2)));
    grid on; box off; hold off;
else
    fprintf('\n(No example hysteresis-positive community to plot -- none was found.)\n');
end

if ~isempty(example_neg)
    figure('Name','Example: community without hysteresis','Position',[1050 550 900 420]);
    plot(example_neg.l_up, example_neg.N1_up, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 12); hold on;
    plot(example_neg.l_down, example_neg.N1_down, 'r.--', 'LineWidth', 1.5, 'MarkerSize', 12);
    xlabel('Leakage, l'); ylabel('N_1 at settled state');
    legend({'Up-sweep','Down-sweep'}, 'Location', 'best');
    if example_neg_is_clean
        title('Example community WITHOUT detected hysteresis (fully converged, clean null result)');
    else
        title('Example community WITHOUT detected hysteresis (NOTE: fallback -- had unconverged points excluded; see reliability breakdown)');
    end
    grid on; box off; hold off;
else
    fprintf('\n(No example hysteresis-negative community to plot -- none was found.)\n');
end

save('hysteresis_test_500_results.mat', 'hysteresis_flag', 'max_n1_diff_all', ...
     'max_n2_diff_all', 'n_dom_mismatch_all', 'frac_hyst', 'example_pos', ...
     'example_neg', 'example_neg_is_clean', 'all_converged_all', ...
     'n_no_hyst_clean', 'n_no_hyst_dirty', ...
     'l_onset_all', 'l_offset_all', 'all_hyst_l_values', 'cfg');

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function [base, x0_ref, l_ref] = draw_baseline_community(cfg)
    % No l_ref is hardcoded. Everything is drawn randomly except m (and S,
    % which follows deterministically from the same reference values) --
    % this is what "fixes m so that coexistence is plausible" while leaving
    % the rest of the community genuinely random, rather than exactly
    % constructed around one specific point.
    N_sp = cfg.N_sp; R_res = cfg.R_res;

    C = rand(N_sp, R_res);                      % consumption matrix, Uniform(0,1)
    W = 0.5 + rand(R_res, 1);                    % energy values, Uniform(0.5, 1.5)
    D = rand(R_res, R_res);                      % metabolic conversion matrix, Uniform(0,1)
    R_ref = 0.5 + rand(R_res, 1);                % reference resource level, Uniform(0.5, 1.5)

    % Leakage used to construct m: drawn the same way as every other
    % parameter here (Uniform(0,1), like C and D), with no special
    % treatment at all -- not fixed, not anchored to the sweep midpoint,
    % and not even restricted to the sweep's [l_low, l_high] range. It is
    % simply "one of the random parameters" at construction time. The
    % actual sweep afterward is a separate, independent operation: it
    % varies l over l_grid regardless of what value was used here, since
    % m and S are frozen once construction is done. There is nothing left
    % that ties community construction to the sweep's own l range.
    l_ref = rand(R_res, 1);

    % m_i = sum_alpha c_ia (1-l_a) w_a R_ref,a : zero growth at R_ref, for
    % this reference leakage. This is the ONE quantity constructed to make
    % coexistence plausible; everything else is free.
    m = C * ((1 - l_ref) .* W .* R_ref);

    % Reference abundance used only to define a plausible external supply
    % level; no species abundance is otherwise drawn or fixed in this
    % construction, so we use a unit reference (N_ref = 1 per species).
    N_ref = ones(N_sp, 1);
    u_ref = C' * N_ref;                          % per-resource depletion driver at N_ref, R_ref
    v_ref = l_ref .* W .* R_ref .* u_ref;
    S_raw = R_ref .* u_ref - (D * v_ref) ./ W;    % depletion expected at R_ref minus recycling return expected at R_ref
    S = max(S_raw, 0.05);                         % floor to keep supply non-negative

    base.N_sp = N_sp; base.R_res = R_res;
    base.C = C; base.D = D; base.W = W; base.m = m; base.S = S; % FIXED for entire sweep

    x0_ref = [N_ref; R_ref]; % starting point for the sweep's first step
end

function [N1_trace, N2_trace, dom_trace, converged_trace] = run_sweep(l_grid, x0_start, base, cfg, odeopt)
    n = numel(l_grid);
    N1_trace = nan(1, n);
    N2_trace = nan(1, n);
    dom_trace = false(n, cfg.N_sp);
    converged_trace = false(1, n);
    x_current = x0_start; % continuation: each step starts where the last one ended
    for k = 1:n
        p = base;
        p.l = l_grid(k) * ones(cfg.R_res, 1);

        % Integrate, then verify the state has actually settled by checking
        % the norm of the RHS there. If not settled, extend the integration
        % time (doubling) and retry from the SAME starting point, rather
        % than silently accepting a possibly-unconverged state. This is
        % what prevents small residual drift from compounding across the
        % sweep and masquerading as direction-dependence (false hysteresis).
        T_this = cfg.T_step;
        x_end = x_current;
        is_converged = false;
        for attempt = 1:cfg.max_extend_attempts
            [~, xtraj] = ode15s(@(t,x) crm_rhs(t, x, p), [0 T_this], x_current, odeopt);
            x_end = xtraj(end,:)';
            rhs_norm = norm(crm_rhs(0, x_end, p));
            if rhs_norm < cfg.rhs_tol
                is_converged = true;
                break;
            end
            T_this = T_this * 2; % retry with double the integration time
        end

        x_current = x_end;
        N1_trace(k) = x_current(1);
        N2_trace(k) = x_current(2);
        dom_trace(k,:) = (x_current(1:cfg.N_sp) > cfg.ext_thresh)';
        converged_trace(k) = is_converged;
    end
end

function dx = crm_rhs(~, x, p)
    N = x(1:p.N_sp);
    R = x(p.N_sp+1:end);
    u = p.C' * N;
    dN = N .* (p.C * ((1 - p.l) .* p.W .* R) - p.m);
    v = p.l .* p.W .* R .* u;
    dR = p.S - R .* u + (p.D * v) ./ p.W;
    dx = [dN; dR];
end
