function fig = plot_scalechamp_surface(voxel, faces, power, cfg)
% PLOT_SCALECHAMP_SURFACE Render cortical source power in superior view.
%
% Inputs:
%   voxel       - Inflated cortical surface vertices [nv x 3]
%   faces       - Surface mesh triangulation [n_faces x 3]
%   power       - Estimated source power per vertex [nv x 1]
%   cfg         - Plotting configuration struct (surface_threshold_ratio)

% 1. Normalize source amplitude and filter active mesh faces
[plot_power, overlay_faces] = local_prepare_plot_power( ...
    power, faces, cfg.surface_threshold_ratio);

% 2. Initialize figure and layout
fig = figure('Color', 'w', 'Name', 'TRESP-0B ScaleChamp surface preview', ...
    'Position', [80 80 675 475]);
tiledlayout(fig, 1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

% 3. Render cortical surface and active source overlay
ax = nexttile;
local_plot_one_view(ax, voxel, faces, overlay_faces, plot_power, ...
    [0 90]);

% 4. Configure colorbar and colormap
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'sqrt power / max';
colormap(fig, local_surface_energy_colormap());
end


function [plot_power, overlay_faces] = local_prepare_plot_power( ...
    power, faces, threshold_ratio)
% LOCAL_PREPARE_PLOT_POWER Compute normalized amplitude and active face subset.
plot_power = sqrt(power(:));
plot_power = plot_power ./ max(plot_power);

% Retain faces containing at least one supra-threshold vertex
hot_vertex = plot_power > 0 & plot_power >= threshold_ratio;
overlay_faces = faces(any(hot_vertex(faces), 2), :);
end


function cmap = local_surface_energy_colormap()
% LOCAL_SURFACE_ENERGY_COLORMAP Generate smooth neutral-to-hot energy colormap.
stops = [0; 0.12; 0.35; 0.65; 1];
colors = [ ...
    0.92 0.92 0.90; ...
    0.96 0.86 0.78; ...
    0.82 0.12 0.07; ...
    1.00 0.43 0.03; ...
    1.00 0.96 0.40];
cmap = interp1(stops, colors, linspace(0, 1, 256)', 'linear');
end


function local_plot_one_view(ax, voxel, faces, overlay_faces, plot_power, ...
    view_angle)
% LOCAL_PLOT_ONE_VIEW Draw base cortex mesh and interpolated activity.
hold(ax, 'on');

% Base neutral cortical surface
patch(ax, 'Faces', faces, 'Vertices', voxel, ...
    'FaceColor', [0.92 0.92 0.90], 'EdgeColor', 'none', ...
    'FaceLighting', 'gouraud', 'AmbientStrength', 0.55, ...
    'DiffuseStrength', 0.55, 'SpecularStrength', 0.05);

% Supra-threshold source activity overlay
patch(ax, 'Faces', overlay_faces, 'Vertices', voxel, ...
    'FaceVertexCData', plot_power, 'FaceColor', 'interp', ...
    'EdgeColor', 'none', 'FaceLighting', 'gouraud', ...
    'AmbientStrength', 0.55, 'DiffuseStrength', 0.65, ...
    'SpecularStrength', 0.05);

% Camera and lighting setup
axis(ax, 'equal');
axis(ax, 'off');
view(ax, view_angle(1), view_angle(2));
camlight(ax, 'headlight');
material(ax, 'dull');
clim(ax, [0 1]);
hold(ax, 'off');
end
