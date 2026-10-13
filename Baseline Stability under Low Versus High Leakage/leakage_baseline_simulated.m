clear all; close all; clc;

%% =========================================================================
% Does high recycling reduce the chance that a stable coexisting
% community forms IN THE FIRST PLACE -- before any perturbation at all?
%
% For each of 1000 random 2-species, 2-resource systems per leakage
% regime, parameters are drawn randomly, mortality is set analytically
% so a reference resource level Rref is a zero-growth point (matching
% this thesis's own Methods convention), and the system is then
% SIMULATED FORWARD via numerical integration from a random initial
% condition until it settles. The settled endpoint is checked: did the
% system reach a genuine, stable, COEXISTING equilibrium, or did it
% fail to (one species excluded, both extinct, or no settling at all)?
%
%   dN_i/dt = N_i [ sum_alpha (1-l_alpha) c_ialpha w_alpha R_alpha - m_i ]
%   dR_alpha/dt = S_alpha(t) - R_alpha sum_i c_ialpha N_i
%                 + sum_{i,beta} l_beta c_ibeta D_alphabeta (w_beta/w_alpha) N_i R_beta
% =========================================================================

rng(2031);
N_sp = 2;
R_res = 2;
n_trials = 1000;
T_settle = 5000;
tol_eq = 1e-4;
tol_coex = 0.05;

ODE_OPTS = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',10,'NonNegative',1:4);

regimes(1).name = 'Low leakage (l in [0.1, 0.3])';
regimes(1).l_lo = 0.1; regimes(1).l_hi = 0.3;
regimes(2).name = 'High leakage (l in [0.6, 0.9])';
regimes(2).l_lo = 0.6; regimes(2).l_hi = 0.9;

results = struct('name',{},'failed',{},'frac_failed',{},'reason',{});

for reg = 1:2
    failed = false(n_trials,1);
    reason = cell(n_trials,1);

    for t = 1:n_trials
        % STEP 1: draw random parameters
        C = rand(N_sp, R_res);                                  % Uniform(0,1)
        W = 0.5 + rand(R_res,1);                                % Uniform(0.5,1.5)
        D = rand(R_res, R_res);                                 % Uniform(0,1)
        l = regimes(reg).l_lo + rand(R_res,1)*(regimes(reg).l_hi - regimes(reg).l_lo);

        % STEP 2: set mortality analytically so a reference resource
        % level Rref is a zero-growth point (this thesis's own
        % convention: m = C * ((1-l).*W.*Rref) )
        Rref = 0.5 + rand(R_res,1);
        m = zeros(N_sp,1);
        for i = 1:N_sp
            m(i) = sum(C(i,:)'.*(1-l).*W.*Rref);
        end

        % STEP 3: set external supply S from that same reference level
        % (depletion-minus-recycling at Rref, with a floor), matching
        % this thesis's convention used elsewhere for resource supply
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

        % STEP 4: SIMULATE FORWARD from a random initial condition
        % until the system settles
        x0 = [0.1+rand(N_sp,1)*2; 0.1+rand(R_res,1)*2];
        try
            [~,Y] = ode15s(@(t,y)crm_ode(t,y,params), [0 T_settle], x0, ODE_OPTS);
            x_eq = max(Y(end,:)',0);
        catch
            failed(t) = true; reason{t} = 'ode_failed';
            continue;
        end
        if any(~isfinite(x_eq))
            failed(t) = true; reason{t} = 'nonfinite';
            continue;
        end

        % STEP 5: check whether it actually settled to a fixed point
        dxdt = crm_ode(0, x_eq, params);
        if max(abs(dxdt)) > tol_eq
            failed(t) = true; reason{t} = 'did_not_settle';
            continue;
        end

        % STEP 6: check whether that settled point is a genuine,
        % positive, COEXISTING equilibrium
        N_eq = x_eq(1:N_sp);
        R_eq = x_eq(N_sp+1:end);
        if any(R_eq <= 0)
            failed(t) = true; reason{t} = 'resource_collapsed';
            continue;
        end
        if ~(N_eq(1) > tol_coex && N_eq(2) > tol_coex)
            failed(t) = true; reason{t} = 'exclusion_or_extinction';
            continue;
        end

        % if we reach here: a stable coexisting equilibrium was reached
        failed(t) = false; reason{t} = 'coexistence';

        if mod(t,100)==0, fprintf('  %d/%d done\n', t, n_trials); end
    end

    frac_failed = mean(failed);
    results(reg).name = regimes(reg).name;
    results(reg).failed = failed;
    results(reg).frac_failed = frac_failed;
    results(reg).reason = reason;

    fprintf('\n%s:\n', regimes(reg).name);
    fprintf('  Fraction of draws that FAILED to produce a stable coexisting equilibrium: %.1f%%\n', 100*frac_failed);
    cats = unique(reason);
    for c = 1:length(cats)
        fprintf('    %-22s %d\n', cats{c}, sum(strcmp(reason,cats{c})));
    end
    fprintf('\n');
end

save('leakage_baseline_simulated.mat', 'results');

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
