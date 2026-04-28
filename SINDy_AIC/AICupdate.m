% run_AIC_pipeline_sweep.m
% Sweeps polyorder 1–4, plots relative AICc per order, and time vs Speed (RPM)
% for the best model in each order.

addpath('utils');
addpath('models');
changeplot;
clearvars;
close all;
clc;

%% ===== USER SETTINGS =====
matfile        = 'mydataupdated.mat';
plottag        = 0;          % set 0 to suppress per-model plots during sweep
numvalidation  = 100;
eps_noise      = 0.0;
usesine        = 0;
laurent        = 0;
horizon_sec    = 5;
polyorders     = [1 2];  % <-- sweep these orders
top_k_orders   = 2;          % plot time-vs-speed for K lowest-AIC orders

%% ===== LOAD DATA =====
if ~isfile(matfile)
    error('MAT file "%s" not found.', matfile);
end
S = load(matfile);
if ~isfield(S,'t') || ~isfield(S,'x')
    error('MAT file must contain t (Nx1) and x (Nx2).');
end
t = S.t;
x = S.x;
[N, n] = size(x);
dt = median(diff(t));
fprintf('Loaded: N=%d samples, n=%d states, dt=%.6g\n', N, n, dt);

%% ===== DERIVATIVE ESTIMATION (done once, shared across orders) =====
framelen = min(N, 11);
if mod(framelen,2)==0, framelen = framelen-1; end
poly_sg = 3;
try
    x_smooth = zeros(size(x));
    for k = 1:n
        x_smooth(:,k) = sgolayfilt(x(:,k), poly_sg, framelen);
    end
catch
    warning('sgolayfilt unavailable, using moving average.');
    b = ones(framelen,1)/framelen;
    x_smooth = filter(b,1,x);
end
dx = zeros(size(x_smooth));
for k = 1:n
    dx(:,k) = gradient(x_smooth(:,k), dt);
end

%% ===== VALIDATION SEGMENTS (fixed across all orders) =====
L = round(horizon_sec / dt);
if L >= N
    error('Horizon too long. Reduce horizon_sec or use more data.');
end
rng(10);
valid_starts = randi([1, N-L], numvalidation, 1);

tA = cell(1,numvalidation);
xA = cell(1,numvalidation);
for ii = 1:numvalidation
    idx = valid_starts(ii):(valid_starts(ii)+L-1);
    tA{ii} = t(idx);
    xA{ii} = x(idx,:) + eps_noise*randn(L,n);
end

val.tA = tA;
val.xA = xA;
val.options = [];

%% ===== SWEEP OVER POLYNOMIAL ORDERS =====
results = struct();  % store per-order results

for oi = 1:length(polyorders)
    polyorder = polyorders(oi);
    fprintf('\n===== polyorder = %d =====\n', polyorder);

    %% Build library
    Theta = poolData(x, n, polyorder, usesine, laurent);
    m = size(Theta,2);
    fprintf('Theta: %d x %d\n', size(Theta,1), m);

    Thetalib.Theta    = Theta;
    Thetalib.normTheta= 0;
    Thetalib.dx       = dx;
    Thetalib.polyorder= polyorder;
    Thetalib.usesine  = usesine;

    %% Sparse regression
    lambdavals.numlambda   = 30;
    lambdavals.lambdastart = -6;
    lambdavals.lambdaend   = 2;
    [Xicomb, numcoeff, ~] = multiD_Xilib(Thetalib, lambdavals);
    fprintf('Candidate models: %d\n', length(Xicomb));

    %% Rebuild val.x0 (same segments, same ICs)
    val.x0 = zeros(n, numvalidation);
    for ii = 1:numvalidation
        val.x0(:,ii) = xA{ii}(1,:)';
    end

    %% Validate all candidates
    abserror = [];
    RMSE     = [];
    IC       = [];
    tB_all   = {};
    xB_all   = {};

    for nn = 1:length(Xicomb)
        Xi = Xicomb{nn};
        [error, RMSE1, savetB, savexB] = validateXi(Xi, Thetalib, val, plottag);
        ICtemp = ICcalculations(error', numcoeff(nn), numvalidation);
        abserror(:,nn) = error';
        RMSE(:,nn)     = RMSE1;
        tB_all{nn}     = savetB;
        xB_all{nn}     = savexB;
        IC{nn}         = ICtemp;
        fprintf('  Model %d/%d | #coeff=%d | mean RMSE=%.5g\n', ...
            nn, length(Xicomb), numcoeff(nn), mean(RMSE1));
    end

    %% Relative AICc for this order
    % NEW
    aic_vals = cellfun(@(s) s.aic_c, IC);
    AIC_rel  = aic_vals - min(aic_vals);

    %% Best model for this order
    [~, best_idx] = min(aic_vals);

    %% Store results
    results(oi).polyorder = polyorder;
    results(oi).AIC_rel   = AIC_rel;
    results(oi).best_idx  = best_idx;
    results(oi).best_Xi   = Xicomb{best_idx};
    results(oi).best_tB   = tB_all{best_idx};
    results(oi).best_xB   = xB_all{best_idx};
    results(oi).numcoeff  = numcoeff;
    results(oi).Thetalib  = Thetalib;
    results(oi).IC        = IC;
end

%% ===== FIGURE 1: Relative AICc bar plots (one subplot per order) =====
figure('Name','Relative AICc by Polynomial Order','NumberTitle','off');
for oi = 1:length(polyorders)
    subplot(2,2,oi);
    bar(results(oi).AIC_rel);
    xlabel('Model Index');
    ylabel('Relative AICc');
    title(sprintf('Poly Order %d', results(oi).polyorder));
    grid on;
end
sgtitle('Relative AICc — Candidate Models per Polynomial Order');

%% ===== FIGURE 1B: Relative AICc across polynomial orders =====
% Compare orders using the best candidate model from each order.
order_best_aic = zeros(1, length(polyorders));
for oi = 1:length(polyorders)
    order_best_aic(oi) = min(cellfun(@(s) s.aic_c, results(oi).IC));
end
order_rel_aic = order_best_aic - min(order_best_aic);

% Rank orders from best (lowest AICc) to worst.
[~, sort_idx] = sort(order_best_aic, 'ascend');
top_k = min(top_k_orders, length(polyorders));
best_order_idx = sort_idx(1:top_k);

figure('Name','Relative AICc Across Polynomial Orders','NumberTitle','off');
bar(polyorders, order_rel_aic, 0.6, 'FaceColor',[0.2 0.6 0.8]); hold on;
scatter(polyorders(best_order_idx), order_rel_aic(best_order_idx), ...
    80, 'r', 'filled');
xlabel('Polynomial Order');
ylabel('Relative AICc (best model per order)');
title('Order Selection by Relative AICc');
legend('Relative AICc', sprintf('Top %d order(s)', top_k), ...
    'Location','best');
grid on;

fprintf('\nBest polynomial orders by AICc:\n');
for kk = 1:top_k
    oi = best_order_idx(kk);
    fprintf('  Rank %d: order=%d | min AICc=%.6g | rel AICc=%.6g\n', ...
        kk, results(oi).polyorder, order_best_aic(oi), order_rel_aic(oi));
end

%% ===== FIGURE 2: Time vs Speed (RPM) — lowest-AIC orders only =====
% Pick one representative validation segment to show (segment 1)
seg = 1;

figure('Name','Time vs Speed — Lowest-AIC Orders','NumberTitle','off');
tiledlayout(top_k,1);
for kk = 1:top_k
    oi = best_order_idx(kk);
    nexttile;

    % Ground truth for segment
    t_true = val.tA{seg};
    x_true = val.xA{seg};          % columns: [duty/voltage, speed]

    % Model prediction for segment
    % best_tB / best_xB are cell arrays over validation segments
    t_pred = results(oi).best_tB{seg};
    x_pred = results(oi).best_xB{seg};   % same column layout

    plot(t_true, x_true(:,2), 'k-',  'LineWidth', 1.5); hold on;
    plot(t_pred, x_pred(:,2), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Speed (RPM)');
    title(sprintf('Order %d (rank %d) — Best Model (idx=%d)', ...
        results(oi).polyorder, kk, results(oi).best_idx));
    legend('Measured', 'SINDy Prediction', 'Location','best');
    grid on;
end
sgtitle('Time vs Speed (RPM) — Best Orders by Relative AICc');

%% ===== cleanup =====
rmpath('utils');
rmpath('models');