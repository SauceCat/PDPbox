import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import copy

import pdp_calc_utils
from sklearn.cluster import MiniBatchKMeans, KMeans


def _pdp_plot_title(n_grids, feature_name, ax, multi_flag, which_class, plot_params):
    """
    Draw pdp plot title

    :param n_grids: number of grids
    :param feature_name: name of the feature
    :param ax: axes to plot on
    :param multi_flag: whether it is a subplot of a multi-classes plot
    :param which_class: which class to plot
    :param plot_params: values of plot parameters
    """

    font_family = 'Arial'
    title = 'PDP for %s' % feature_name
    subtitle = "Number of unique grid points: %d" % n_grids

    title_fontsize = 15
    subtitle_fontsize = 12

    if plot_params is not None:
        if 'font_family' in plot_params.keys():
            font_family = plot_params['font_family']
        if 'title' in plot_params.keys():
            title = plot_params['title']
        if 'title_fontsize' in plot_params.keys():
            title_fontsize = plot_params['title_fontsize']
        if 'subtitle_fontsize' in plot_params.keys():
            subtitle_fontsize = plot_params['subtitle_fontsize']

    ax.set_facecolor('white')
    if multi_flag:
        ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
        ax.text(0, 0.45, "For Class %d" % which_class, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family)
        ax.text(0, 0.25, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    else:
        ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
        ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _axis_modify(font_family, ax):
    """
    Modify axes
    """

    for tick in ax.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_family)

    ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='#424242', colors='#9E9E9E')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.grid(True, 'major', 'x', ls='--', lw=.5, c='k', alpha=.3)
    ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)


def _pdp_plot(pdp_isolate_out, feature_name, center, plot_org_pts, plot_lines, frac_to_plot,
              cluster, n_cluster_centers, cluster_method, x_quantile, ax, plot_params):
    """
    Plot partial dependent plot

    :param pdp_isolate_out: instance of pdp_isolate_obj
        a calculated pdp_isolate_obj instance
    :param feature_name: string
        name of the feature, not necessary the same as the column name
    :param center: boolean, default=True
        whether to center the plot
    :param plot_org_pts: boolean, default=False
        whether to plot out the original points
    :param plot_lines: boolean, default=False
        whether to plot out the individual lines
    :param frac_to_plot: float or integer, default=1
        how many points or lines to plot, can be a integer or a float
    :param cluster: boolean, default=False
        whether to cluster the individual lines and only plot out the cluster centers
    :param n_cluster_centers: integer, default=None
        number of cluster centers
    :param cluster_method: string, default=None
        cluster method to use, default is KMeans, if 'approx' is passed, MiniBatchKMeans is used
    :param x_quantile: boolean, default=False
        whether to construct x axis ticks using quantiles
    :param ax: axes to plot on
    :param plot_params: dict, default=None
        values of plot parameters
    """

    font_family = 'Arial'
    xticks_rotation = 0

    if plot_params is not None:
        if 'font_family' in plot_params.keys():
            font_family = plot_params['font_family']
        if 'xticks_rotation' in plot_params.keys():
            xticks_rotation = plot_params['xticks_rotation']

    # modify axes
    _axis_modify(font_family, ax)
    ax.set_xlabel(feature_name, fontsize=10)

    feature_type = pdp_isolate_out.feature_type
    feature_grids = pdp_isolate_out.feature_grids
    display_columns = pdp_isolate_out.display_columns
    actual_columns = pdp_isolate_out.actual_columns

    if feature_type == 'binary' or feature_type == 'onehot' or x_quantile:
        x = range(len(feature_grids))
        ax.set_xticks(x)
        ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    else:
        # for numeric feature
        x = feature_grids

    ice_lines = copy.deepcopy(pdp_isolate_out.ice_lines)
    pdp_y = copy.deepcopy(pdp_isolate_out.pdp)

    # whether to fill between std upper and lower
    # whether to highlight pdp line
    std_fill = True
    pdp_hl = False

    # whether to center the plot
    if center:
        pdp_y -= pdp_y[0]
        for col in feature_grids[1:]:
            ice_lines[col] -= ice_lines[feature_grids[0]]
        ice_lines['actual_preds'] -= ice_lines[feature_grids[0]]
        ice_lines[feature_grids[0]] = 0

    if cluster or plot_lines:
        std_fill = False
        pdp_hl = True
        if cluster:
            _ice_cluster_plot(x=x, ice_lines=ice_lines, feature_grids=feature_grids, n_cluster_centers=n_cluster_centers,
                              cluster_method=cluster_method, ax=ax, plot_params=plot_params)
        else:
            ice_plot_data = pdp_calc_utils._sample_data(ice_lines=ice_lines, frac_to_plot=frac_to_plot)
            _ice_line_plot(x=x, ice_plot_data=ice_plot_data, feature_grids=feature_grids, ax=ax, plot_params=plot_params)

    if plot_org_pts:
        ice_lines_temp = ice_lines.copy()
        if feature_type == 'onehot':
            ice_lines_temp['x'] = ice_lines_temp[actual_columns].apply(lambda x: pdp_calc_utils._find_onehot_actual(x), axis=1)
            ice_lines_temp = ice_lines_temp[ice_lines_temp['x'].isnull() == False].reset_index(drop=True)
        elif feature_type == 'numeric':
            feature_grids = pdp_isolate_out.feature_grids
            ice_lines_temp = ice_lines_temp[(ice_lines_temp[actual_columns[0]] >= feature_grids[0])
                                            & (ice_lines_temp[actual_columns[0]] <= feature_grids[-1])]
            if x_quantile:
                ice_lines_temp['x'] = ice_lines_temp[actual_columns[0]].apply(lambda x : pdp_calc_utils._find_closest(x, feature_grids))
            else:
                ice_lines_temp['x'] = ice_lines_temp[actual_columns[0]]
        else:
            ice_lines_temp['x'] = ice_lines_temp[actual_columns[0]]

        ice_plot_data_pts = pdp_calc_utils._sample_data(ice_lines=ice_lines_temp, frac_to_plot=frac_to_plot)
        _ice_plot_pts(ice_plot_data_pts=ice_plot_data_pts, ax=ax, plot_params=plot_params)

    std = ice_lines[feature_grids].std().values
    _pdp_std_plot(x=x, y=pdp_y, std=std, std_fill=std_fill, pdp_hl=pdp_hl, ax=ax, plot_params=plot_params)


def _pdp_std_plot(x, y, std, std_fill, pdp_hl, ax, plot_params):
    """
    PDP basic plot

    :param x: x axis values
    :param y: pdp values
    :param std: std values
    :param std_fill: whether to fill between std upper and lower
    :param pdp_hl: whether to highlight pdp line
    :param ax: axes to plot on
    :param plot_params: dictionary of plot config
    """

    upper = y + std
    lower = y - std

    pdp_color = '#1A4E5D'
    pdp_hl_color = '#FEDC00'
    pdp_linewidth = 2
    zero_color = '#E75438'
    zero_linewidth = 1.5
    fill_color = '#66C2D7'
    fill_alpha = 0.2
    markersize = 5

    if plot_params is not None:
        if 'pdp_color' in plot_params.keys():
            pdp_color = plot_params['pdp_color']
        if 'pdp_hl_color' in plot_params.keys():
            pdp_hl_color = plot_params['pdp_hl_color']
        if 'pdp_linewidth' in plot_params.keys():
            pdp_linewidth = plot_params['pdp_linewidth']
        if 'zero_color' in plot_params.keys():
            zero_color = plot_params['zero_color']
        if 'zero_linewidth' in plot_params.keys():
            zero_linewidth = plot_params['zero_linewidth']
        if 'fill_color' in plot_params.keys():
            fill_color = plot_params['fill_color']
        if 'fill_alpha' in plot_params.keys():
            fill_alpha = plot_params['fill_alpha']
        if 'markersize' in plot_params.keys():
            markersize = plot_params['markersize']

    if pdp_hl:
        ax.plot(x, y, color=pdp_hl_color, linewidth=pdp_linewidth * 3, alpha=0.8)

    ax.plot(x, y, color=pdp_color, linewidth=pdp_linewidth, marker='o', markersize=markersize)
    ax.plot(x, [0] * y, linestyle='--', linewidth=zero_linewidth, color=zero_color)

    if std_fill:
        ax.fill_between(x, upper, lower, alpha=fill_alpha, color=fill_color)

    ax.set_ylim(np.min([np.min(lower) * 2, 0]), np.max([np.max(upper) * 2, 0]))


def _ice_plot_pts(ice_plot_data_pts, ax, plot_params):
    """
    Plot the real data points

    :param ice_plot_data_pts: data points to plot
    :param ax: axes to plot on
    :param plot_params: dictionary of plot config
    """

    point_size = 50
    point_pos_color = '#5BB573'
    point_neg_color = '#E75438'

    if plot_params is not None:
        if 'point_size' in plot_params.keys():
            point_size = plot_params['point_size']
        if 'point_pos_color' in plot_params.keys():
            point_pos_color = plot_params['point_pos_color']
        if 'point_neg_color' in plot_params.keys():
            point_neg_color = plot_params['point_neg_color']

    ice_plot_data_pts['color'] = ice_plot_data_pts['actual_preds'].apply(lambda x: point_pos_color if x >= 0 else point_neg_color)
    ax.scatter(ice_plot_data_pts['x'], ice_plot_data_pts['actual_preds'], s=point_size, marker="+", linewidth=1,
               color=ice_plot_data_pts['color'])


def _ice_line_plot(x, ice_plot_data, feature_grids, ax, plot_params):
    """
    Plot the ice lines

    :param x: x axis values
    :param ice_plot_data: ice lines to plot
    :param ax: axes to plot on
    :param plot_params: dictionary of plot config
    """

    linewidth = np.max([1.0 / np.log10(ice_plot_data.shape[0]), 0.3])
    linealpha = np.max([1.0 / np.log10(ice_plot_data.shape[0]), 0.3])

    line_cmap = 'Blues'
    if plot_params is not None:
        if 'line_cmap' in plot_params.keys():
            line_cmap = plot_params['line_cmap']

    colors = plt.get_cmap(line_cmap)(np.linspace(0, 1, 20))[5:15]

    for i in range(len(ice_plot_data)):
        y = list(ice_plot_data[feature_grids].iloc[i].values)
        ax.plot(x, y, linewidth=linewidth, c=colors[i % 10], alpha=linealpha)


def _ice_cluster_plot(x, ice_lines, feature_grids, n_cluster_centers, cluster_method, ax, plot_params):
    """
    Cluster the ice lines and plot out the cluster centers

    :param x: x axis values
    :param ice_lines: ice lines
    :param n_cluster_centers: number of cluster centers
    :param cluster_method: cluster method
    :param ax: axes to plot on
    :param plot_params: dictionary of plot config
    """

    if cluster_method == 'approx':
        kmeans = MiniBatchKMeans(n_clusters=n_cluster_centers, random_state=0, verbose=0)
    else:
        kmeans = KMeans(n_clusters=n_cluster_centers, random_state=0, n_jobs=1)

    kmeans.fit(ice_lines[feature_grids])
    cluster_plot_data = pd.DataFrame(kmeans.cluster_centers_, columns=feature_grids)

    cluster_cmap = 'Blues'

    if plot_params is not None:
        if 'cluster_cmap' in plot_params.keys():
            cluster_cmap = plot_params['cluster_cmap']

    colors = plt.get_cmap(cluster_cmap)(np.linspace(0, 1, 20))[5:15]

    for i in range(len(cluster_plot_data)):
        y = list(cluster_plot_data[feature_grids].iloc[i].values)
        ax.plot(x, y, linewidth=1, c=colors[i % 10])


def _pdp_interact_plot_title(pdp_interact_out, feature_names, ax,
                             multi_flag, which_class, only_inter, plot_params):
    """
    Draw pdp interaction plot title

    :param pdp_interact_out: instance of pdp_interact_obj
    :param feature_name: name of the features
    :param ax: axes to plot on
    :param figsize: figure size
    :param multi_flag: whether it is a subplot of a multi-classes plot
    :param which_class: which class to plot
    :param only_inter: whether only draw interaction plot
    :param plot_params: values of plot parameters
    """

    font_family = 'Arial'
    title = 'Interaction PDP between %s and %s' % (feature_names[0], feature_names[1])

    title_fontsize = 14
    subtitle_fontsize = 12

    if type(pdp_interact_out) == dict:
        subtitle1 = 'Number of unique grid points of %s: %d' % (
        feature_names[0], len(pdp_interact_out['class_0'].feature_grids[0]))
        subtitle2 = 'Number of unique grid points of %s: %d' % (
        feature_names[1], len(pdp_interact_out['class_0'].feature_grids[1]))
    else:
        subtitle1 = 'Number of unique grid points of %s: %d' % (
        feature_names[0], len(pdp_interact_out.feature_grids[0]))
        subtitle2 = 'Number of unique grid points of %s: %d' % (
        feature_names[1], len(pdp_interact_out.feature_grids[1]))

    if plot_params is not None:
        if 'pdp_inter' in plot_params.keys():
            if 'font_family' in plot_params.keys():
                font_family = plot_params['font_family']
            if 'title' in plot_params.keys():
                title = plot_params['title']
            if 'title_fontsize' in plot_params.keys():
                title_fontsize = plot_params['title_fontsize']
            if 'subtitle_fontsize' in plot_params.keys():
                subtitle_fontsize = plot_params['subtitle_fontsize']

    ax.set_facecolor('white')
    if only_inter:
        ax.text(0, 0.8, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
        if multi_flag:
            ax.text(0, 0.62, "For Class %d" % which_class, va="top", ha="left", fontsize=title_fontsize,
                    fontname=font_family)
            ax.text(0, 0.45, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
            ax.text(0, 0.3, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
        else:
            ax.text(0, 0.55, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
            ax.text(0, 0.4, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
    else:
        ax.text(0, 0.6, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
        if multi_flag:
            ax.text(0, 0.53, "For Class %d" % which_class, va="top", ha="left", fontsize=title_fontsize,
                    fontname=font_family)
            ax.text(0, 0.4, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
            ax.text(0, 0.35, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
        else:
            ax.text(0, 0.4, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
            ax.text(0, 0.35, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family,
                    color='grey')
    ax.axis('off')


def _pdp_interact_plot(pdp_interact_out, feature_names, center, plot_org_pts, plot_lines, frac_to_plot, cluster,
                       n_cluster_centers, cluster_method, x_quantile, figsize, plot_params, multi_flag, which_class):
    """
    Plot interaction plot

    :param pdp_interact_out: instance of pdp_interact_obj
        a calculated pdp_interact_obj instance
    :param feature_names: list of feature names
    :param center: boolean, default=True
        whether to center the plot
    :param plot_org_pts: boolean, default=False
        whether to plot out the original points
    :param plot_lines: boolean, default=False
        whether to plot out the individual lines
    :param frac_to_plot: float or integer, default=1
        how many points or lines to plot, can be a integer or a float
    :param cluster: boolean, default=False
        whether to cluster the individual lines and only plot out the cluster centers
    :param n_cluster_centers: integer, default=None
        number of cluster centers
    :param cluster_method: string, default=None
        cluster method to use, default is KMeans, if 'approx' is passed, MiniBatchKMeans is used
    :param x_quantile: boolean, default=False
        whether to construct x axis ticks using quantiles
    :param figsize: figure size
    :param plot_params: dict, default=None
        values of plot parameters
    :param multi_flag: boolean, default=False
        whether it is a subplot of a multi-class plot
    :param which_class: integer, default=None
        must not be None under multi-class mode
    """

    if figsize is None:
        fig = plt.figure(figsize=(15, 15))
    else:
        fig = plt.figure(figsize=figsize)

    pdp_plot_params = None
    if plot_params is not None:
        if 'pdp' in plot_params.keys():
            pdp_plot_params = plot_params['pdp']

    gs = GridSpec(2, 2)
    ax0 = plt.subplot(gs[0, 0])

    _pdp_interact_plot_title(pdp_interact_out=pdp_interact_out, feature_names=feature_names, ax=ax0,
                             multi_flag=multi_flag, which_class=which_class, only_inter=False, plot_params=plot_params)

    ax1 = plt.subplot(gs[0, 1])
    _pdp_plot(pdp_isolate_out=pdp_interact_out.pdp_isolate_out1, feature_name=feature_names[0], center=center,
              plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster,
              n_cluster_centers=n_cluster_centers, cluster_method=cluster_method, x_quantile=x_quantile,
              ax=ax1, plot_params=pdp_plot_params)

    ax2 = plt.subplot(gs[1, 0])
    _pdp_plot(pdp_isolate_out=pdp_interact_out.pdp_isolate_out2, feature_name=feature_names[1], center=center,
              plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster,
              n_cluster_centers=n_cluster_centers, cluster_method=cluster_method, x_quantile=x_quantile, ax=ax2,
              plot_params=pdp_plot_params)

    ax3 = plt.subplot(gs[1, 1])
    _pdp_contour_plot(pdp_interact_out=pdp_interact_out, feature_names=feature_names, x_quantile=x_quantile,
                      ax=ax3, fig=fig, plot_params=plot_params)

    
class ColorBarLocator(object):
    def __init__(self, pax, pad=60, width=20):
        self.pax = pax
        self.pad = pad
        self.width = width

    def __call__(self, ax, renderer):
        x, y, w, h = self.pax.get_position().bounds
        fig = self.pax.get_figure()
        inv_trans = fig.transFigure.inverted()
        pad, _ = inv_trans.transform([self.pad, 0])
        width, _ = inv_trans.transform([self.width, 0])
        return [x, y - pad, w, width]


def _pdp_contour_plot(pdp_interact_out, feature_names, x_quantile, ax, fig, plot_params):
    """
    Plot PDP contour

    :param pdp_interact_out: instance of pdp_interact_obj
        a calculated pdp_interact_obj instance
    :param feature_names: list of feature names
    :param x_quantile: boolean, default=False
        whether to construct x axis ticks using quantiles
    :param ax: axes to plot on
    :param fig: plt figure
    :param plot_params: dict, default=None
        values of plot parameters
    """

    font_family = 'Arial'
    contour_color = 'white'
    contour_cmap = 'viridis'
    xticks_rotation = 0

    if plot_params is not None:
        if 'pdp_inter' in plot_params.keys():
            if 'contour_color' in plot_params['pdp_inter'].keys():
                contour_color = plot_params['pdp_inter']['contour_color']
            if 'contour_cmap' in plot_params['pdp_inter'].keys():
                contour_cmap = plot_params['pdp_inter']['contour_cmap']
            if 'font_family' in plot_params['pdp_inter'].keys():
                font_family = plot_params['pdp_inter']['font_family']
            if 'xticks_rotation' in plot_params.keys():
                xticks_rotation = plot_params['xticks_rotation']

    _axis_modify(font_family, ax)

    feature_types = pdp_interact_out.feature_types
    pdp = copy.deepcopy(pdp_interact_out.pdp)

    new_feature_names = []
    for i, feature_type in enumerate(feature_types):
        if feature_type == 'onehot':
            new_col = 'onehot_%d' % (i)
            pdp[new_col] = pdp.apply(lambda x: list(x[pdp_interact_out.features[i]]).index(1), axis=1)
            new_feature_names.append(new_col)
        else:
            new_feature_names.append(pdp_interact_out.features[i])

    if (feature_types[0] == 'numeric') and x_quantile:
        pdp[new_feature_names[0]] = pdp[new_feature_names[0]].apply(
            lambda x: list(pdp_interact_out.feature_grids[0]).index(x))

    if (feature_types[1] == 'numeric') and x_quantile:
        pdp[new_feature_names[1]] = pdp[new_feature_names[1]].apply(
            lambda x: list(pdp_interact_out.feature_grids[1]).index(x))

    X, Y = np.meshgrid(pdp[new_feature_names[0]].unique(), pdp[new_feature_names[1]].unique())
    Z = []
    for i in range(X.shape[0]):
        zs = []
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            z = pdp[(pdp[new_feature_names[0]] == x) & (pdp[new_feature_names[1]] == y)]['preds'].values[0]
            zs.append(z)
        Z.append(zs)
    Z = np.array(Z)

    if feature_types[0] == 'onehot':
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels(pdp_interact_out.pdp_isolate_out1.display_columns, rotation=xticks_rotation)
    elif feature_types[0] == 'binary':
        ax.set_xticks([0, 1])
        ax.set_xticklabels(pdp_interact_out.pdp_isolate_out1.display_columns, rotation=xticks_rotation)
    else:
        if x_quantile:
            ax.set_xticks(range(len(pdp_interact_out.feature_grids[0])))
            ax.set_xticklabels(pdp_interact_out.feature_grids[0], rotation=xticks_rotation)

    if feature_types[1] == 'onehot':
        ax.set_yticks(range(Y.shape[0]))
        ax.set_yticklabels(pdp_interact_out.pdp_isolate_out2.display_columns)
    elif feature_types[1] == 'binary':
        ax.set_yticks([0, 1])
        ax.set_yticklabels(pdp_interact_out.pdp_isolate_out2.display_columns)
    else:
        if x_quantile:
            ax.set_yticks(range(len(pdp_interact_out.feature_grids[1])))
            ax.set_yticklabels(pdp_interact_out.feature_grids[1])

    level = np.min([X.shape[0], X.shape[1]])
    c1 = ax.contourf(X, Y, Z, N=level, origin='lower', cmap=contour_cmap)
    c2 = ax.contour(c1, levels=c1.levels, colors=contour_color, origin='lower')
    ax.clabel(c2, contour_label_fontsize=9, inline=1)

    ax.set_xlabel(feature_names[0], fontsize=10)
    ax.set_ylabel(feature_names[1], fontsize=10)
    ax.get_yaxis().tick_right()

    if fig is not None:
        cax = fig.add_axes([0, 0, 0, 0], axes_locator=ColorBarLocator(ax))
        fig.colorbar(c1, cax=cax, orientation='horizontal')


