function [x_est, scale_power, theta] = scale_champagne( ...
    y, f, voxel, sigu, scale_list, nem, Adj)
% SCALE_CHAMPAGNE Multi-scale Champagne source reconstruction.
% Supports one or three source orientations with scalar voxel variance.
%
% Inputs:
%   y          - Sensor measurements [nk x nt]
%   f          - Leadfield [nk x nv] or [nk x 3*nv], with three
%                consecutive orientation columns per voxel; [nk x nv x 3]
%                is also accepted
%   voxel      - 3D voxel coordinates [nv x 3]
%   sigu       - Sensor noise covariance [nk x nk]
%   scale_list - Spatial scale candidates [1 x num_scales]
%   nem        - Number of EM iterations for Champagne
%   Adj        - Cortical mesh adjacency matrix [nv x nv]
%
% Outputs:
%   x_est       - Reconstructed source time-courses [nv*nd x nt]
%   scale_power - Total energy across each scale [1 x num_scales]
%   theta       - Estimated source coefficients in normalized scale space

% Internal algorithm parameters
p = 2;
min_weight = 0.1;
scale_energy_threshold_ratio = 0.05;
support_threshold_ratio = 0.01;

% Determine dimensions
nv = size(voxel, 1);
nt = size(y, 2);
if ndims(f) == 3
    if size(f, 2) ~= nv || size(f, 3) ~= 3
        error('A 3D leadfield must have size [nk x nv x 3].');
    end
    f = reshape(permute(f, [1 3 2]), size(f, 1), 3 * nv);
end
nvd = size(f, 2);
if nvd == nv
    nd = 1;
elseif nvd == 3 * nv
    nd = 3;
else
    error('Leadfield columns must equal nv or 3*nv.');
end
scale_list = scale_list(:)';
num_scales = numel(scale_list);

% Normalize data and noise covariance energy
y_scale = sqrt(mean(y(:).^2));
sigu = sigu ./ (y_scale^2);
y = y / y_scale;

% Build symmetric voxel adjacency graph and estimate median edge length (mm)
Adj_graph = spones(sparse(Adj) | sparse(Adj)');
Adj_list = local_adjacency_list(Adj_graph, nv);

[ei, ej] = find(triu(Adj_graph, 1));
edge_len = sqrt(sum((voxel(ei, :) - voxel(ej, :)).^2, 2));
graph_edge_mm = median(edge_len);

% Precompute local shortest-path graph distances for each voxel
max_graph_dist_mm = local_max_graph_steps(scale_list, p, min_weight, graph_edge_mm);
[graph_idx_list, graph_dist_list] = ...
    local_graph_distance_cache(Adj_list, nv, max_graph_dist_mm, voxel);

% Construct multi-scale spatial smoothing weights W and normalized leadfields F
W_list = cell(1, num_scales);
F_list = cell(1, num_scales);
col_norm_list = cell(1, num_scales);

for iscale = 1:num_scales
    W = local_build_scale_weight(nv, scale_list(iscale), p, min_weight, ...
        graph_idx_list, graph_dist_list, graph_edge_mm);
    if nd == 3
        W = kron(W, speye(3));
    end
    F = f * W;
    
    % L2-normalize leadfield columns to prevent scale amplitude bias
    col_norm = sqrt(sum(F.^2, 1));
    valid = col_norm > 0;
    F(:, valid) = F(:, valid) ./ col_norm(valid);
    col_norm(~valid) = 1;
    
    W_list{iscale} = W;
    F_list{iscale} = F;
    col_norm_list{iscale} = col_norm;
end

% Concatenate multi-scale leadfields and append identity matrix for sensor noise (auto_bg)
F_multi = cat(2, F_list{:});
n_task_cols = size(F_multi, 2);
F_multi = [F_multi, eye(size(f, 1))];

% Run Champagne solver and discard the appended noise-channel coefficients
theta_norm = champagne(y, F_multi, sigu, nem, nd, n_task_cols);
theta_norm = theta_norm(1:n_task_cols, :);

% Rescale coefficients back to original amplitude and compute scale energy
theta_by_scale = cell(1, num_scales);
center_scale_power = zeros(nv, num_scales);
scale_power = zeros(1, num_scales);

for iscale = 1:num_scales
    idx = (iscale - 1) * nvd + (1:nvd);
    % Undo leadfield column normalization
    theta_block = theta_norm(idx, :) ./ col_norm_list{iscale}';
    theta_by_scale{iscale} = theta_block;
    
    % Compute time-series energy per voxel at the current scale
    center_scale_power(:, iscale) = local_voxel_power(theta_block, nv, nd);
    scale_power(iscale) = sum(center_scale_power(:, iscale));
end

% Select active scales and dominant voxel scales to synthesize source activity
scale_energy_ratio = scale_power ./ sum(scale_power);
active_scale_mask = scale_energy_ratio >= scale_energy_threshold_ratio;

[center_power, center_best_scale_index] = max(center_scale_power, [], 2);
center_thr = scale_energy_threshold_ratio * max(center_power);

x_est = zeros(nvd, nt);
for iscale = 1:num_scales
    if ~active_scale_mask(iscale)
        continue;
    end
    
    theta_block = theta_by_scale{iscale};
    % Keep only voxel centers where the current scale is dominant and above threshold
    inactive_voxels = (center_best_scale_index ~= iscale) | (center_power < center_thr);
    theta_block(repelem(inactive_voxels, nd), :) = 0;
    
    % Map scale coefficients back to voxel space via spatial kernel W
    x_est = x_est + W_list{iscale} * theta_block;
end

% Hard-threshold final voxel power to prune weak background artifacts
if support_threshold_ratio > 0
    voxel_power_final = local_voxel_power(x_est, nv, nd);
    thr = support_threshold_ratio * max(voxel_power_final);
    x_est(repelem(voxel_power_final < thr, nd), :) = 0;
end

theta = theta_norm;
end


function x = champagne(y, f, sigu, nem, nd, n_task_cols)
% CHAMPAGNE Empirical Bayesian sparse source reconstruction.
eps1 = 1e-8;
[nk, nvd] = size(f);

% Initialize prior source variance via rough pseudo-inverse projection
f2 = sum(f.^2, 1);
invf2 = zeros(1, nvd);
valid = f2 > 0;
invf2(valid) = 1 ./ f2(valid);
w = spdiags(invf2', 0, nvd, nvd) * f';
inu0 = mean(mean((w * y).^2));
vvec = inu0 * ones(nvd, 1);

% EM iterations to update prior variances (vvec) and posterior sources (x)
for iem = 1:nem
    vmat = spdiags(vvec, 0, nvd, nvd);
    
    % Compute model data covariance and enforce symmetry
    c = f * vmat * f' + sigu;
    c = (c + c') / 2;
    
    % Regularized eigendecomposition inverse of data covariance
    [eigvec, d] = eig(c);
    d = max(real(diag(d)), 0);
    invd = zeros(nk, 1);
    ff = d >= eps1;
    invd(ff) = 1 ./ d(ff);
    invc = eigvec * spdiags(invd, 0, nk, nk) * eigvec';
    
    % Compute spatial filter weights w and posterior source estimate x
    fc = f' * invc;
    w = vmat * fc;
    x = w * y;
    
    % Convex-bound hyperparameter update rule for voxel variances
    x2 = mean(x.^2, 2);
    z = sum(fc .* f', 2);
    if nd == 1
        vvec = sqrt(x2 ./ z);
    else
        % Share one variance across the three orientations of each voxel.
        x2_voxel = sum(reshape(x2(1:n_task_cols), nd, []), 1);
        z_voxel = sum(reshape(z(1:n_task_cols), nd, []), 1);
        variance = zeros(size(z_voxel));
        valid = z_voxel > 0;
        variance(valid) = sqrt(max(x2_voxel(valid) ./ z_voxel(valid), 0));
        vvec = zeros(nvd, 1);
        vvec(1:n_task_cols) = reshape(repmat(variance, nd, 1), [], 1);

        % The appended sensor-noise columns remain independent.
        bg = (n_task_cols + 1):nvd;
        valid = z(bg) > 0;
        vvec(bg(valid)) = sqrt(max(x2(bg(valid)) ./ z(bg(valid)), 0));
    end
end
end


function power = local_voxel_power(x, nv, nd)
power = sum(reshape(sum(x.^2, 2), nd, nv), 1)';
end


function W = local_build_scale_weight(nv, scale, p, min_weight, ...
    graph_idx_list, graph_dist_list, graph_edge_mm)
% LOCAL_BUILD_SCALE_WEIGHT Construct sparse spatial smoothing kernel W for a given scale.
% Reduces to identity matrix (point sources) when scale <= 0.
if scale <= 0
    W = speye(nv);
    return;
end

sigma = scale * graph_edge_mm * 2;
rows = cell(nv, 1);
cols = cell(nv, 1);
vals = cell(nv, 1);

for iv = 1:nv
    keep = graph_idx_list{iv};
    d = graph_dist_list{iv};
    
    % Generalized Gaussian kernel weights truncated at min_weight
    w = exp(-(abs(d ./ sigma) .^ p));
    use = w >= min_weight;
    
    rows{iv} = keep(use);
    cols{iv} = iv * ones(sum(use), 1);
    vals{iv} = w(use);
end

% Assemble sparse kernel and normalize column energy
W = sparse(vertcat(rows{:}), vertcat(cols{:}), vertcat(vals{:}), nv, nv);
col_energy = sqrt(sum(W.^2, 1));
W = W * spdiags(1 ./ col_energy', 0, nv, nv);
end


function Adj_list = local_adjacency_list(Adj, nv)
% LOCAL_ADJACENCY_LIST Convert sparse adjacency matrix into a cell adjacency list.
[row, col] = find(Adj);
Adj_list = accumarray(row, col, [nv, 1], @(x) {x(:)'}, {[]});
end


function max_dist_mm = local_max_graph_steps(scale_list, p, min_weight, graph_edge_mm)
% LOCAL_MAX_GRAPH_STEPS Compute maximum physical search radius (mm) for the largest scale.
max_sigma = max(scale_list) * graph_edge_mm * 2;
max_dist_mm = round((-log(min_weight))^(1/p) * max_sigma) + 5 * graph_edge_mm;
end


function [idx_list, dist_list] = local_graph_distance_cache(Adj_list, nv, max_dist_mm, voxel)
% LOCAL_GRAPH_DISTANCE_CACHE Precompute neighborhood indices and geodesic distances per voxel.
idx_list = cell(nv, 1);
dist_list = cell(nv, 1);
for src = 1:nv
    [idx_list{src}, dist_list{src}] = gf_bfs_distance(Adj_list, src, nv, max_dist_mm, voxel);
end
end


function [idx, dist] = gf_bfs_distance(Adj_list, src, nv, max_dist_mm, voxel)
% GF_BFS_DISTANCE Compute local graph shortest-path distances from src within max_dist_mm.
dist_full = inf(nv, 1);
dist_full(src) = 0;
visited = false(nv, 1);
visited(src) = true;
frontier = src;

while ~isempty(frontier)
    nxt = [];
    for f = frontier(:)'
        for v_nb = Adj_list{f}
            if ~visited(v_nb)
                edge_d = norm(voxel(f, :) - voxel(v_nb, :));
                new_d = dist_full(f) + edge_d;
                % Relax edge and queue neighbor if within search radius
                if new_d < dist_full(v_nb)
                    dist_full(v_nb) = new_d;
                    if new_d < max_dist_mm
                        nxt = [nxt, v_nb];
                    end
                end
            end
        end
        visited(f) = true;
    end
    frontier = unique(nxt);
    frontier = frontier(~visited(frontier));
end

idx = find(isfinite(dist_full));
dist = dist_full(idx);
end
