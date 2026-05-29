%% local_conditioning_project.m
% Conditioning for Multiscale Sampling with Overlapping Subdomains
%
% PROVENANCE: This file is a cleaned, faithful reconstruction of "Listing 1"
% in NLA-Project-Report.pdf (Arya Kalantari, MATH 6340, advisor L. F. Pereira,
% 2026-05-04). It was recovered from the PDF text layer; identifier spacing and
% smart quotes were repaired. It is the VALIDATED static local-conditioning core
% that produced the good results reported in the project (residual -> machine
% precision, interface-jump reduction, cond(A) growth table). Treat it as the
% numerical ground truth to port and to test the Python implementation against.
%
% IMPORTANT: this script implements ONLY the static local conditioning / local
% sampling step. It does NOT contain a Darcy forward solver, a likelihood, or an
% MCMC loop -- those are the new pieces the Python project must build (see SPEC.md).
%
% Pipeline:
%   1. Generate one global Gaussian log-permeability sample G_old.
%   2. Choose one coarse subdomain.
%   3. Enlarge it by an overlap / buffer.
%   4. Build a local KLE on the overlapping local region.
%   5. Choose Mb conditioning points in the buffer.
%   6. Build A*theta = c.
%   7. Compute one (minimum-norm) particular solution theta_p.
%   8. Compute Null(A).
%   9. Generate several conditioned local samples:
%          theta = theta_p + theta_n,   theta_n in Null(A).
%  10. Restrict the local update back to the non-overlapping subdomain.
%  11. Compare constrained vs unconstrained local updates.

clear; close all; clc;
rng(7);

%% Parameters
params.nx = 48;
params.ny = 48;

params.nCoarseX = 4;
params.nCoarseY = 4;

params.targetCol = 2;
params.targetRow = 2;

params.overlapCells = 2;

params.sigma = 1.0;
params.corrLength = 0.18;

params.globalModes = 90;
params.Nc = 30;

params.Mb_list = [1 2 4 8 16 32 64];
params.nSamples = 100;

params.localModesMax = params.Nc + max(params.Mb_list);

fprintf('\nLOCAL CONDITIONING PROJECT\n');
fprintf('Global grid: %d x %d\n', params.nx, params.ny);
fprintf('Coarse partition: %d x %d\n', params.nCoarseX, params.nCoarseY);
fprintf('Target subdomain: row %d, col %d\n', params.targetRow, params.targetCol);
fprintf('Overlap width: %d fine-grid cells\n', params.overlapCells);
fprintf('Base local stochastic dimension Nc = %d\n', params.Nc);
fprintf('Mb values tested: ');
fprintf('%d ', params.Mb_list);
fprintf('\n\n');

%% Build global grid and synthetic global Gaussian sample
[xVec, yVec, X, Y] = makeCellCenteredGrid(params.nx, params.ny);
globalPts = [X(:), Y(:)];

fprintf('Building global covariance matrix...\n');
Cglobal = covarianceMatrix(globalPts, params.sigma, params.corrLength);

fprintf('Computing global KLE eigenpairs...\n');
[PhiGlobal, lambdaGlobal] = topEigenpairs(Cglobal, params.globalModes);

thetaGlobal = randn(params.globalModes, 1);
GoldVec = PhiGlobal * (sqrt(lambdaGlobal) .* thetaGlobal);
Gold = reshape(GoldVec, params.ny, params.nx);

%% Choose target subdomain and overlap region
sub = makeSubdomain(params);

localGlobalIdx = sub.localGlobalIdx;
localPts = globalPts(localGlobalIdx, :);

coreLocalIdx = sub.coreLocalIdx;
coreGlobalIdx = localGlobalIdx(coreLocalIdx);

bufferLocalIdx = sub.bufferLocalIdx;

fprintf('Target core cells: %d\n', numel(coreLocalIdx));
fprintf('Overlap / local cells: %d\n', numel(localGlobalIdx));
fprintf('Buffer cells: %d\n\n', numel(bufferLocalIdx));

%% Build local KLE on overlapping subdomain
fprintf('Building local covariance matrix...\n');
Clocal = covarianceMatrix(localPts, params.sigma, params.corrLength);

fprintf('Computing local KLE eigenpairs...\n');
[PhiLocal, lambdaLocal] = topEigenpairs(Clocal, params.localModesMax);

%% Run experiments for different Mb
numMb = numel(params.Mb_list);

Mb_col = zeros(numMb, 1);
Next_col = zeros(numMb, 1);
rank_col = zeros(numMb, 1);
nullDim_col = zeros(numMb, 1);
cond_col = zeros(numMb, 1);
minSing_col = zeros(numMb, 1);
meanResidCond_col = zeros(numMb, 1);
meanResidUnc_col = zeros(numMb, 1);
meanJumpCond_col = zeros(numMb, 1);
meanJumpUnc_col = zeros(numMb, 1);

results = cell(numMb, 1);

for ib = 1:numMb

    Mb = params.Mb_list(ib);
    Next = params.Nc + Mb;

    fprintf('--------------------------------------------------\n');
    fprintf('Experiment with Mb = %d, Next = Nc + Mb = %d\n', Mb, Next);

    condLocalIdx = selectConditioningPoints(localPts, coreLocalIdx, bufferLocalIdx, Mb);

    PhiExt = PhiLocal(:, 1:Next);
    sqrtLam = sqrt(lambdaLocal(1:Next));

    A = buildConditioningMatrix(PhiExt, sqrtLam, condLocalIdx);
    c = GoldVec(localGlobalIdx(condLocalIdx));

    [theta_p, Z, linInfo] = solveConditioningSVD(A, c);

    residParticular = norm(A * theta_p - c) / max(1, norm(c));

    fprintf('rank(A) = %d\n', linInfo.rankA);
    fprintf('nullity(A) = %d\n', size(Z, 2));
    fprintf('effective cond(A) = %.4e\n', linInfo.condEffective);
    fprintf('relative residual particular = %.4e\n', residParticular);

    condJumps = zeros(params.nSamples, 1);
    uncJumps = zeros(params.nSamples, 1);

    condResiduals = zeros(params.nSamples, 1);
    uncResiduals = zeros(params.nSamples, 1);

    conditionedFields = cell(params.nSamples, 1);
    unconstrainedFields = cell(params.nSamples, 1);

    for m = 1:params.nSamples

        % Conditioned null-space sample
        eta = randn(Next, 1);

        if isempty(Z)
            theta_n = zeros(Next, 1);
        else
            theta_n = Z * (Z' * eta);
        end

        theta_cond = theta_p + theta_n;
        GlocalCond = PhiExt * (sqrtLam .* theta_cond);

        tempVec = GoldVec;
        tempVec(coreGlobalIdx) = GlocalCond(coreLocalIdx);
        Gcond = reshape(tempVec, params.ny, params.nx);

        condResiduals(m) = norm(A * theta_cond - c) / max(1, norm(c));
        condJumps(m) = interfaceJumpRMS(Gcond, sub);

        conditionedFields{m} = Gcond;

        % Unconstrained local sample
        theta_unc = randn(Next, 1);
        GlocalUnc = PhiExt * (sqrtLam .* theta_unc);

        tempVec = GoldVec;
        tempVec(coreGlobalIdx) = GlocalUnc(coreLocalIdx);
        Gunc = reshape(tempVec, params.ny, params.nx);

        uncResiduals(m) = norm(A * theta_unc - c) / max(1, norm(c));
        uncJumps(m) = interfaceJumpRMS(Gunc, sub);

        unconstrainedFields{m} = Gunc;
    end

    fprintf('mean constrained residual = %.4e\n', mean(condResiduals));
    fprintf('mean unconstrained residual = %.4e\n', mean(uncResiduals));
    fprintf('mean constrained interface jump = %.4e\n', mean(condJumps));
    fprintf('mean unconstrained interface jump = %.4e\n', mean(uncJumps));

    Mb_col(ib) = Mb;
    Next_col(ib) = Next;
    rank_col(ib) = linInfo.rankA;
    nullDim_col(ib) = size(Z, 2);
    cond_col(ib) = linInfo.condEffective;
    minSing_col(ib) = linInfo.minNonzeroSingular;
    meanResidCond_col(ib) = mean(condResiduals);
    meanResidUnc_col(ib) = mean(uncResiduals);
    meanJumpCond_col(ib) = mean(condJumps);
    meanJumpUnc_col(ib) = mean(uncJumps);

    results{ib}.Mb = Mb;
    results{ib}.Next = Next;
    results{ib}.A = A;
    results{ib}.c = c;
    results{ib}.theta_p = theta_p;
    results{ib}.Z = Z;
    results{ib}.condLocalIdx = condLocalIdx;
    results{ib}.conditionedFields = conditionedFields;
    results{ib}.unconstrainedFields = unconstrainedFields;
    results{ib}.condResiduals = condResiduals;
    results{ib}.uncResiduals = uncResiduals;
    results{ib}.condJumps = condJumps;
    results{ib}.uncJumps = uncJumps;
end

summaryTable = table( ...
    Mb_col, Next_col, rank_col, nullDim_col, cond_col, minSing_col, ...
    meanResidCond_col, meanResidUnc_col, meanJumpCond_col, meanJumpUnc_col, ...
    'VariableNames', {'Mb', 'Next', 'RankA', 'NullDim', 'CondA', 'MinSingularValue', ...
    'MeanResidualConditioned', 'MeanResidualUnconditioned', ...
    'MeanJumpConditioned', 'MeanJumpUnconditioned'});

fprintf('\n\nSUMMARY TABLE\n');
disp(summaryTable);

%% Visualization
chosenMb = 2;
chosenIdx = find(params.Mb_list == chosenMb, 1);

if isempty(chosenIdx)
    chosenIdx = 1;
end

R = results{chosenIdx};

figure('Name', 'Old global field and selected subdomain');
imagesc(xVec, yVec, Gold);
axis image; set(gca, 'YDir', 'normal');
colorbar;
title('Original global sample G_{old}');
xlabel('x'); ylabel('y');
hold on;
drawSubdomainRectangles(sub, xVec, yVec);
plot(localPts(R.condLocalIdx,1), localPts(R.condLocalIdx,2), ...
    'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
legend('Conditioning points');

figure('Name', 'Constrained vs unconstrained local updates');

subplot(2,2,1);
imagesc(xVec, yVec, Gold);
axis image; set(gca, 'YDir', 'normal');
colorbar;
title('Original G_{old}');
hold on;
drawSubdomainRectangles(sub, xVec, yVec);

subplot(2,2,2);
imagesc(xVec, yVec, R.unconstrainedFields{1});
axis image; set(gca, 'YDir', 'normal');
colorbar;
title(sprintf('Unconstrained update, M_b=%d', R.Mb));
hold on;
drawSubdomainRectangles(sub, xVec, yVec);

subplot(2,2,3);
imagesc(xVec, yVec, R.conditionedFields{1});
axis image; set(gca, 'YDir', 'normal');
colorbar;
title(sprintf('Conditioned update 1, M_b=%d', R.Mb));
hold on;
drawSubdomainRectangles(sub, xVec, yVec);

subplot(2,2,4);
imagesc(xVec, yVec, R.conditionedFields{min(2, params.nSamples)});
axis image; set(gca, 'YDir', 'normal');
colorbar;
title(sprintf('Conditioned update 2, M_b=%d', R.Mb));
hold on;
drawSubdomainRectangles(sub, xVec, yVec);

figure('Name', 'Numerical diagnostics');

subplot(1,3,1);
semilogy(summaryTable.Mb, summaryTable.CondA, '-o', 'LineWidth', 1.5);
xlabel('M_b');
ylabel('effective cond(A)');
title('Conditioning of A');
grid on;

subplot(1,3,2);
semilogy(summaryTable.Mb, summaryTable.MeanResidualConditioned, '-o', 'LineWidth', 1.5);
hold on;
semilogy(summaryTable.Mb, summaryTable.MeanResidualUnconditioned, '--s', 'LineWidth', 1.5);
xlabel('M_b');
ylabel('relative residual');
title('Constraint residual');
legend('conditioned', 'unconditioned', 'Location', 'best');
grid on;

subplot(1,3,3);
plot(summaryTable.Mb, summaryTable.MeanJumpConditioned, '-o', 'LineWidth', 1.5);
hold on;
plot(summaryTable.Mb, summaryTable.MeanJumpUnconditioned, '--s', 'LineWidth', 1.5);
xlabel('M_b');
ylabel('RMS interface jump');
title('Boundary mismatch');
legend('conditioned', 'unconditioned', 'Location', 'best');
grid on;

fprintf('\nDone. Use the summary table and figures in your report.\n');

%% Helper Functions

function [xVec, yVec, X, Y] = makeCellCenteredGrid(nx, ny)
    xVec = ((1:nx) - 0.5) / nx;
    yVec = ((1:ny) - 0.5) / ny;
    [X, Y] = meshgrid(xVec, yVec);
end

function C = covarianceMatrix(pts, sigma, ell)
    dx = pts(:,1) - pts(:,1).';
    dy = pts(:,2) - pts(:,2).';
    dist = sqrt(dx.^2 + dy.^2);

    C = sigma^2 * exp(-dist / ell);
    C = (C + C.') / 2;
    C = C + 1e-12 * eye(size(C));
end

function [V, lambda] = topEigenpairs(C, k)
    C = (C + C.') / 2;
    n = size(C, 1);
    k = min(k, n);

    if k < n
        opts.isreal = true;
        opts.tol = 1e-10;
        opts.maxit = 1000;

        try
            [V, D] = eigs(C, k, 'la', opts);
        catch
            [Vfull, Dfull] = eig(C);
            [lambdaFull, idx] = sort(diag(Dfull), 'descend');
            idx = idx(1:k);
            V = Vfull(:, idx);
            lambda = lambdaFull(1:k);
            lambda = max(lambda, 0);
            return;
        end

        lambda = diag(D);
        [lambda, idx] = sort(lambda, 'descend');
        V = V(:, idx);
        lambda = max(lambda, 0);
    else
        [Vfull, Dfull] = eig(C);
        [lambda, idx] = sort(diag(Dfull), 'descend');
        V = Vfull(:, idx);
        lambda = max(lambda, 0);
    end
end

function sub = makeSubdomain(params)
    nx = params.nx;
    ny = params.ny;

    blockX = nx / params.nCoarseX;
    blockY = ny / params.nCoarseY;

    if abs(blockX - round(blockX)) > 0 || abs(blockY - round(blockY)) > 0
        error('nx and ny must be divisible by the number of coarse cells.');
    end

    blockX = round(blockX);
    blockY = round(blockY);

    c = params.targetCol;
    r = params.targetRow;

    coreCols = (c-1)*blockX + 1 : c*blockX;
    coreRows = (r-1)*blockY + 1 : r*blockY;

    ov = params.overlapCells;

    hatCols = max(1, coreCols(1) - ov) : min(nx, coreCols(end) + ov);
    hatRows = max(1, coreRows(1) - ov) : min(ny, coreRows(end) + ov);

    [HatCols, HatRows] = meshgrid(hatCols, hatRows);
    localGlobalIdx = sub2ind([ny, nx], HatRows(:), HatCols(:));

    isCore = ...
        HatCols(:) >= coreCols(1) & HatCols(:) <= coreCols(end) & ...
        HatRows(:) >= coreRows(1) & HatRows(:) <= coreRows(end);

    sub.coreCols = coreCols;
    sub.coreRows = coreRows;
    sub.hatCols = hatCols;
    sub.hatRows = hatRows;
    sub.localGlobalIdx = localGlobalIdx;
    sub.coreLocalIdx = find(isCore);
    sub.bufferLocalIdx = find(~isCore);
end

function condLocalIdx = selectConditioningPoints(localPts, coreLocalIdx, bufferLocalIdx, Mb)
    if Mb > numel(bufferLocalIdx)
        error('Mb is larger than the number of available buffer cells.');
    end

    center = mean(localPts(coreLocalIdx, :), 1);
    bufferPts = localPts(bufferLocalIdx, :);

    angles = atan2(bufferPts(:,2) - center(2), bufferPts(:,1) - center(1));
    [~, order] = sort(angles);

    orderedBuffer = bufferLocalIdx(order);

    rawPositions = round(linspace(1, numel(orderedBuffer), Mb + 2));
    rawPositions = rawPositions(2:end-1);
    rawPositions = unique(rawPositions, 'stable');

    while numel(rawPositions) < Mb
        missing = setdiff(1:numel(orderedBuffer), rawPositions, 'stable');
        rawPositions(end+1) = missing(1);
    end

    condLocalIdx = orderedBuffer(rawPositions(1:Mb));
end

function A = buildConditioningMatrix(PhiExt, sqrtLam, condLocalIdx)
    % Row j of A corresponds to a conditioning point; column l to a KLE mode.
    % A(j,l) = sqrt(lambda_l) * phi_l(y_j). The .* sqrtLam.' scales columns.
    A = PhiExt(condLocalIdx, :) .* sqrtLam.';
end

function [theta_p, Z, info] = solveConditioningSVD(A, c)
    % MINIMUM-NORM particular solution via the economy SVD, plus a null-space
    % basis. This is the STABLE construction (theta_p is perpendicular to
    % Null(A)). The Python project must reproduce this exactly AND also provide
    % an "arbitrary" LU/pivot-column particular solution to reproduce the
    % instability (see SPEC.md, conditioning module).
    [U, S, V] = svd(A, 'econ');

    s = diag(S);

    if isempty(s)
        error('A has no singular values.');
    end

    tol = max(size(A)) * eps(max(s));
    r = sum(s > tol);

    if r == 0
        error('Numerical rank of A is zero.');
    end

    theta_p = V(:,1:r) * ((U(:,1:r)' * c) ./ s(1:r));
    Z = null(A);

    info.rankA = r;
    info.singularValues = s;
    info.minNonzeroSingular = s(r);
    info.condEffective = s(1) / s(r);
end

function jump = interfaceJumpRMS(G, sub)
    rows = sub.coreRows;
    cols = sub.coreCols;

    diffs = [];

    if cols(1) > 1
        inside = G(rows, cols(1));
        outside = G(rows, cols(1)-1);
        diffs = [diffs; inside(:) - outside(:)];
    end

    if cols(end) < size(G, 2)
        inside = G(rows, cols(end));
        outside = G(rows, cols(end)+1);
        diffs = [diffs; inside(:) - outside(:)];
    end

    if rows(1) > 1
        inside = G(rows(1), cols);
        outside = G(rows(1)-1, cols);
        diffs = [diffs; inside(:) - outside(:)];
    end

    if rows(end) < size(G, 1)
        inside = G(rows(end), cols);
        outside = G(rows(end)+1, cols);
        diffs = [diffs; inside(:) - outside(:)];
    end

    jump = sqrt(mean(diffs.^2));
end

function drawSubdomainRectangles(sub, xVec, yVec)
    dx = xVec(2) - xVec(1);
    dy = yVec(2) - yVec(1);

    coreX0 = xVec(sub.coreCols(1)) - dx/2;
    coreY0 = yVec(sub.coreRows(1)) - dy/2;
    coreW = numel(sub.coreCols) * dx;
    coreH = numel(sub.coreRows) * dy;

    hatX0 = xVec(sub.hatCols(1)) - dx/2;
    hatY0 = yVec(sub.hatRows(1)) - dy/2;
    hatW = numel(sub.hatCols) * dx;
    hatH = numel(sub.hatRows) * dy;

    rectangle('Position', [hatX0, hatY0, hatW, hatH], ...
        'EdgeColor', 'w', 'LineWidth', 2, 'LineStyle', '--');

    rectangle('Position', [coreX0, coreY0, coreW, coreH], ...
        'EdgeColor', 'k', 'LineWidth', 2);
end
