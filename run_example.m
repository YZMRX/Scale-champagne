%% Scale-champ example: TRESP-0B
% Run this file from any working directory.
clear; clc;

% Setup paths and file locations
example_dir = fileparts(mfilename('fullpath'));
addpath(example_dir);
addpath(fullfile(example_dir, 'algorithm'));

task_file = fullfile(example_dir, 'data', 'taskdata.mat');
leadfield_file = fullfile(example_dir, 'data', 'leadfield.mat');
noise_file = fullfile(example_dir, 'data', 'noisecov.mat');

% Configure algorithm parameters
active_window_s = [0 0.5];
scale_list_input = 0:10;

% Load dataset, leadfield, and precomputed noise covariance
task = load(task_file, 'data');
lead = load(leadfield_file, 'leadfield_normal', 'labels', ...
    'source_rr', 'faces', 'Adj', 'display_voxel');
noise = load(noise_file, 'sigu');

% Align channel labels and extract active time window
time = task.data.time(:)';
data_labels = cellstr(string(task.data.label(:)));
lead_labels = cellstr(string(lead.labels(:)));
[~, data_idx, lead_idx] = intersect(data_labels, lead_labels, 'stable');

active_idx = time >= active_window_s(1) & time <= active_window_s(2);
y_active = task.data.avg(data_idx, active_idx);
leadfield = lead.leadfield_normal(lead_idx, :);

% Extract cortical surface geometry and convert 0-based faces if needed
voxel = lead.source_rr;
display_voxel = lead.display_voxel;
faces = double(lead.faces);
if min(faces(:)) == 0
    faces = faces + 1;
end
Adj = spones(sparse(lead.Adj) | sparse(lead.Adj)');

% Normalize leadfield columns and scale data/noise covariance energy
column_norm = sqrt(sum(leadfield .^ 2, 1));
leadfield = leadfield ./ column_norm;

y_scale = sqrt(mean(y_active(:) .^ 2));
y_active = y_active / y_scale;
sigu = noise.sigu / (y_scale^2);

% Convert physical input scales using median cortical mesh edge length
[edge_i, edge_j] = find(triu(Adj, 1));
edge_mm = sqrt(sum((voxel(edge_i, :) - voxel(edge_j, :)).^2, 2));
graph_edge_mm = median(edge_mm);

scale_list = scale_list_input / graph_edge_mm;

% Run Multi-scale Champagne source reconstruction
fprintf('%d channels, %d vertices\n', size(y_active, 1), size(leadfield, 2));
fprintf('Using precomputed noise covariance.\n');

x_est = scale_champagne( ...
    y_active, leadfield, voxel, sigu, scale_list, 100, Adj);

% Compute source power and render inflated cortical surface
source_power = sum(real(x_est).^2, 2);

plot_cfg = struct('surface_threshold_ratio', 0.1);
fig = plot_scalechamp_surface(display_voxel, faces, source_power, plot_cfg);
drawnow;
