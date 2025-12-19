% run_AIC_pipeline_with_mat.m
% Updated pipeline for user dataset (time + 2 states, MAT input)
% Keeps utils/ and models/ on path (functions required: poolData, multiD_Xilib,
% validateXi, ICcalculations, AnalyzeOutput).

addpath('utils');
addpath('models');
changeplot;
clearvars;
close all;
clc;

%% ===== USER SETTINGS =====
matfile = '20kdata.mat';  % <<-- set to your MAT filename if different
plottag = 2;                   % plotting option passed to validateXi
numvalidation = 100;           % number of cross-validation segments
eps_noise = 0.0;               % (optional) add measurement noise to validation segments
polyorder = 2;                 % library polynomial order
usesine = 0;
laurent = 0;

%% ===== LOAD MAT (expects variables t (Nx1) and x (Nx2)) =====
if ~isfile(matfile)
    error('MAT file "%s" not found in current folder.', matfile);
end
S = load(matfile);

if isfield(S,'t') && isfield(S,'x')
    t = S.t;
    x = S.x;
else
    error('MAT file must contain variables ''t'' (Nx1) and ''x'' (Nx2).');
end

% basic check
[N, n] = size(x);
if numel(t) ~= N
    error('Length(t) (=%d) does not match number of rows in x (=%d).', numel(t), N);
end
fprintf('Loaded %s: N=%d samples, n=%d states\n', matfile, N, n);

%% ===== Sampling interval dt =====
% If your data has real timestamps, compute dt from t. Otherwise change dt manually.
dt = median(diff(t));
if ~isfinite(dt) || dt <= 0
    error('Computed dt is invalid. Check your time vector t.');
end
fprintf('Using dt = %.6g (median of diff(t))\n', dt);

%% ===== Derivative estimation (Savitzky-Golay smoothing + gradient) =====
% Choose smoothing window & polynomial for Savitzky-Golay
framelen = min(N, 11);      % odd window length (increase for noisier signals)
if mod(framelen,2)==0, framelen = framelen-1; end
poly_sg = 3;

% Apply smoothing (prefer sgolayfilt if available)
try
    x_smooth = zeros(size(x));
    for k = 1:n
        x_smooth(:,k) = sgolayfilt(x(:,k), poly_sg, framelen);
    end
catch
    warning('sgolayfilt not available; falling back to simple moving-average smoothing.');
    win = max(3, framelen);
    b = ones(win,1)/win;
    x_smooth = filter(b,1,x);
end

% Numerical derivative (central differences via gradient)
dx = zeros(size(x_smooth));
for k = 1:n
    dx(:,k) = gradient(x_smooth(:,k), dt);
end

%% ===== Build Theta library =====
Theta = poolData(x, n, polyorder, usesine, laurent);  % poolData from utils/
m = size(Theta,2);

Thetalib.Theta = Theta;
Thetalib.normTheta = 0;
Thetalib.dx = dx;
Thetalib.polyorder = polyorder;
Thetalib.usesine = usesine;

fprintf('Built Theta: %d rows x %d columns\n', size(Theta,1), m);

%% ===== Sparse regression (multiD_Xilib) =====
lambdavals.numlambda   = 30;
lambdavals.lambdastart = -6;
lambdavals.lambdaend   = 2;

[Xicomb, numcoeff, lambdavec] = multiD_Xilib(Thetalib, lambdavals);
fprintf('Found %d candidate models from sparse regression.\n', length(Xicomb));

%% ===== Create validation segments from existing data =====
% We use data segments of length L corresponding to horizon_sec seconds
horizon_sec = 5;                     % validation horizon in seconds (same as original)
L = round(horizon_sec / dt);         % number of samples in validation horizon
if L >= N
    error('Validation horizon L=%d is too long for dataset length N=%d. Reduce horizon or provide more data.', L, N);
end

rng(10);  % reproducible
valid_starts = randi([1, N - L], numvalidation, 1);

tA = cell(1,numvalidation);
xA = cell(1,numvalidation);
for ii = 1:numvalidation
    idx = valid_starts(ii):(valid_starts(ii)+L-1);
    tA{ii} = t(idx);
    xA{ii} = x(idx,:) + eps_noise*randn(L,n);  % optional small noise for robustness
end

% Build val struct matching expected format (val.x0 is n x numvalidation)
val.x0 = zeros(n, numvalidation);
for ii = 1:numvalidation
    % initial condition: column vector
    val.x0(:,ii) = xA{ii}(1,:)';
end
val.tA = tA;
val.xA = xA;
val.options = [];  % not used for purely data-driven validation

%% ===== Validate candidate models and compute AIC =====
clear abserror RMSE tB xB IC
for nn = 1:length(Xicomb)
    Xi = Xicomb{nn};
    % validateXi should compare model predictions (from Xi) to data segments in val
    [error, RMSE1, savetB, savexB] = validateXi(Xi, Thetalib, val, plottag);
    ICtemp = ICcalculations(error', numcoeff(nn), numvalidation);
    abserror(:,nn) = error';
    RMSE(:,nn) = RMSE1;
    tB{nn} = savetB;
    xB{nn} = savexB;
    IC(nn) = ICtemp;
    fprintf('Model %d/%d: #coeff=%d, mean RMSE=%.5g\n', nn, length(Xicomb), numcoeff(nn), mean(RMSE1));
end

AIC_rel = cell2mat({IC.aic_c}) - min(cell2mat({IC.aic_c}));

%% ===== Analyze & output =====
AnalyzeOutput;

%% ===== cleanup =====
rmpath('utils');
rmpath('models');
