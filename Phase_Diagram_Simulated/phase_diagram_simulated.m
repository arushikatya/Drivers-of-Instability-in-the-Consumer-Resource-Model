clear all; close all; clc;

%% =========================================================================
% PHASE DIAGRAM: fraction of communities reaching stable coexistence,
% as a joint function of niche overlap and leakage (Fig 4 structure).
%
% Every trial: consumption C is constructed to hit a target niche
% overlap (cosine similarity between the two species' consumption
% vectors); energy values W and the metabolic matrix D are drawn fully
% at random; leakage l is set to the grid's target level. Mortality m
% and external supply S are NOT drawn arbitrarily at random (which
% would make most systems nonsensical regardless of leakage or
% overlap) -- they are built from a random reference resource level
% Rref, the same convention used elsewhere in this thesis, so that the
% system starts from a biologically FEASIBLE point. No equilibrium is
% assumed or solved for analytically anywhere in this script: every
% trial is then simulated forward via numerical integration of the
% full model equations until it settles (or fails to), exactly as in
% leakage_baseline_simulated.m.
%
%   dN_i/dt = N_i [ sum_alpha (1-l_alpha) c_ialpha w_alpha R_alpha - m_i ]
%   dR_alpha/dt = S_alpha - R_alpha sum_i c_ialpha N_i
%                 + sum_{i,beta} l_beta c_ibeta D_alphabeta (w_beta/w_alpha) N_i R_beta
% =========================================================================

rng(2033);
N_sp = 2;
R_res = 2;
T_settle = 5000;
tol_eq = 1e-4;
tol_coex = 0.05;

overlap_grid = linspace(0.1, 0.98, 8);
leakage_grid = linspace(0.05, 0.9, 8);
n_trials_per_cell = 60;
overlap_tol = 0.04;
max_overlap_attempts = 300;

ODE_OPTS = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',10,'NonNegative',1:4);

frac_stable = nan(length(leakage_grid), length(overlap_grid));

for li = 1:length(leakage_grid)
    l_val = leakage_grid(li);
    fprintf('Leakage = %.2f (%d/%d)\n', l_val, li, length(leakage_grid));

    for oi = 1:length(overlap_grid)
        target_overlap = overlap_grid(oi);

        n_stable = 0;

        for trial = 1:n_trials_per_cell
            % STEP 1: construct C to hit the target niche overlap
            c1 = rand(1,R_res) + 0.1;
            found = false;
            for attempt = 1:max_overlap_attempts
                c2_try = rand(1,R_res) + 0.1;
                cs = dot(c1,c2_try)/(norm(c1)*norm(c2_try));
                if abs(cs - target_overlap) <= overlap_tol
                    c2 = c2_try; found = true; break;
                end
            end
            if ~found
                if target_overlap > 0.95
                    c2 = abs(c1 + 0.01*randn(1,R_res)) + 0.01;
                else
                    c2 = c1 + (1-target_overlap)*randn(1,R_res)*0.3;
                    c2 = abs(c2) + 0.01;
                end
            end
            C = [c1; c2];

            % STEP 2: remaining parameters drawn randomly; leakage set
            % to this cell's grid value
            W = 0.5 + rand(R_res,1);
            D = rand(R_res, R_res);
            l = l_val*ones(R_res,1);

            % STEP 3: mortality and supply built to be FEASIBLE, but
            % mortality is now drawn INDEPENDENTLY per species -- each
            % species' break-even growth rate at Rref is computed from
            % its own row of C, then scaled by its own independent
            % random multiplier (Uniform(0.5,1.5), matching this
            % thesis's "community screen" Methods convention). This
            % means m1 and m2 do NOT automatically become nearly equal
            % just because c1 and c2 are nearly equal (high overlap),
            % unlike a construction that uses no independent
            % per-species randomness on top of C.
            Rref = 0.5 + rand(R_res,1);
            m = zeros(N_sp,1);
            for i = 1:N_sp
                breakeven_i = sum(C(i,:)'.*(1-l).*W.*Rref);
                mult_i = 0.5 + rand()*1.0;   % independent Uniform(0.5,1.5) per species
                m(i) = mult_i * breakeven_i;
            end
            S = zeros(R_res,1);
            for a = 1:R_res
                dep = Rref(a)*sum(C(:,a))*0.5;
                rec = 0;
                for b = 1:R_res
                    rec = rec + l(b)*D(a,b)*(W(b)/W(a))*C(a,b)*Rref(b)*sum(C(:,b))*0.5;
                end
                S(a) = max(dep-rec, 0.05);
            end

            params = struct('W',W,'l',l,'C',C,'D',D,'m',m,'S',S);

            % STEP 4: SIMULATE FORWARD to completion -- no analytic
            % equilibrium, no eigenvalue shortcut
            x0 = [0.1+rand(N_sp,1)*2; 0.1+rand(R_res,1)*2];
            is_stable_coexist = false;
            try
                [~,Y] = ode15s(@(t,y)crm_ode(t,y,params), [0 T_settle], x0, ODE_OPTS);
                x_eq = max(Y(end,:)',0);
                if all(isfinite(x_eq))
                    dxdt = crm_ode(0, x_eq, params);
                    if max(abs(dxdt)) <= tol_eq
                        N_eq = x_eq(1:N_sp);
                        R_eq = x_eq(N_sp+1:end);
                        if all(R_eq > 0) && N_eq(1) > tol_coex && N_eq(2) > tol_coex
                            is_stable_coexist = true;
                        end
                    end
                end
            catch
                % ODE solver failure counts as not reaching coexistence
            end

            n_stable = n_stable + is_stable_coexist;
        end

        frac_stable(li, oi) = n_stable / n_trials_per_cell;
    end
end

save('phase_diagram_simulated_results.mat', 'frac_stable', 'overlap_grid', 'leakage_grid');

%% -- Figure ---------------------------------------------------------------
figure('Position',[100 100 750 600]);
imagesc(overlap_grid, leakage_grid, frac_stable);
set(gca,'YDir','normal');
colormap(flipud(parula));
cb = colorbar;
set(get(cb,'Label'), 'String', 'Fraction reaching stable coexistence');
xlabel('Niche overlap (cosine similarity between c_1, c_2)');
ylabel('Leakage level, l');
title('Phase diagram: stable coexistence vs. niche overlap \times leakage');
saveas(gcf, 'phase_diagram_simulated.png');

fprintf('\nDone. Grid saved to phase_diagram_simulated_results.mat, figure saved to phase_diagram_simulated.png\n');

%% =========================================================================
function dydt = crm_ode(~, y, p)
    N=max(y(1:2),0); R=max(y(3:4),0);
    N=min(N,1e6); R=min(R,1e6);
    dN=zeros(2,1);
    for i=1:2
        g=sum((1-p.l).*p.C(i,:)'.*p.W.*R)-p.m(i);
        dN(i)=N(i)*max(min(g,100),-100);
    end
    dR=zeros(2,1);
    for a=1:2
        dep = R(a)*sum(p.C(:,a).*N);
        rec = 0;
        for b=1:2
            rec = rec + p.l(b)*p.D(a,b)*(p.W(b)/p.W(a))*sum(p.C(:,b).*N)*R(b);
        end
        dR(a)=p.S(a)-dep+rec;
    end
    dydt=[dN;dR]; dydt(~isfinite(dydt))=0;
end
