import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import copy

from .pdp_calc_utils import _sample_data
from .info_plot_utils import _axes_modify, _autolabel, _modify_legend_ax

from sklearn.cluster import MiniBatchKMeans, KMeans


def _pdp_plot_title(n_grids, feature_name, ax, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    title = plot_params.get('title', 'PDP for %s' % feature_name)
    subtitle = plot_params.get('subtitle', "Number of unique grid points: %d" % n_grids)
    title_fontsize = plot_params.get('title_fontsize', 15)
    subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)

    ax.set_facecolor('white')
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    ax.text(0, 0.5, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _draw_pdp_countplot(count_data, count_ax, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    count_data['count_norm'] = count_data['count'] * 1.0 / count_data['count'].sum()
    bar_plot_data = count_data['count_norm'].values
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(bar_plot_data))

    cmap = plot_params.get('line_cmap', 'Blues')
    count_ax.imshow(np.expand_dims(bar_plot_data, 0), aspect="auto", cmap=cmap, norm=norm)
    x = count_data['x'].values

    for idx in range(len(bar_plot_data)):
        text_color = "black"
        if bar_plot_data[idx] >= np.max(bar_plot_data) * 0.5:
            text_color = "white"
        count_ax.text(x[idx], 0, round(bar_plot_data[idx], 3), ha="center", va="center",
                      color=text_color, fontdict={'family': font_family})
    _modify_legend_ax(count_ax, font_family=font_family)


def _draw_pdp_distplot(hist_data, hist_ax, vmin, vmax, plot_params):
    font_family = plot_params.get('font_family', 'Arial')

    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(hist_data))
    cmap = plot_params.get('line_cmap', 'Blues')

    hist_ax.imshow(np.expand_dims(hist_data, 0), aspect="auto", cmap=cmap, norm=norm, extent=(vmin, vmax, 0, 0.5))
    _modify_legend_ax(hist_ax, font_family=font_family)


def _pdp_plot(pdp_isolate_out, feature_name, center, plot_lines, frac_to_plot, cluster, n_cluster_centers,
              cluster_method, x_quantile, show_percentile, pdp_ax, count_data, count_ax, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    # modify axes
    _axes_modify(font_family, pdp_ax)
    feature_type = pdp_isolate_out.feature_type
    feature_grids = pdp_isolate_out.feature_grids
    display_columns = pdp_isolate_out.display_columns
    percentile_info = pdp_isolate_out.percentile_info

    if feature_type == 'binary' or feature_type == 'onehot' or x_quantile:
        x = range(len(feature_grids))
        count_x = range(count_data['x'].min(), count_data['x'].max() + 1)
        pdp_ax.set_xticks(count_x)
        pdp_ax.set_xticklabels([])
        pdp_ax.set_xlim(count_x[0] - 0.5, count_x[-1] + 0.5)

        if x_quantile:
            display_columns_adj = display_columns
            if len(count_x) > len(display_columns):
                display_columns_adj = [''] + list(display_columns)
            pdp_ax.set_xticklabels(display_columns_adj, rotation=xticks_rotation)

            if show_percentile and len(percentile_info) > 0:
                percentile_ax = pdp_ax.twiny()
                percentile_ax.set_xticks(pdp_ax.get_xticks())
                percentile_ax.set_xbound(pdp_ax.get_xbound())
                percentile_info_adj = percentile_info
                if len(count_x) > len(percentile_info):
                    percentile_info_adj = [''] + list(percentile_info)
                percentile_ax.set_xticklabels(percentile_info_adj, rotation=xticks_rotation)
                percentile_ax.set_xlabel('percentile info')
                _axes_modify(font_family=font_family, ax=percentile_ax, top=True)
    else:
        # for numeric feature not x_quantile
        x = feature_grids
        # pdp_ax.set_xticklabels([])

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
        ice_lines[feature_grids[0]] = 0

    if cluster or plot_lines:
        std_fill = False
        pdp_hl = True
        if cluster:
            _ice_cluster_plot(x=x, ice_lines=ice_lines, feature_grids=feature_grids, n_cluster_centers=n_cluster_centers,
                              cluster_method=cluster_method, ax=pdp_ax, plot_params=plot_params)
        else:
            ice_plot_data = _sample_data(ice_lines=ice_lines, frac_to_plot=frac_to_plot)
            _ice_line_plot(x=x, ice_plot_data=ice_plot_data, feature_grids=feature_grids, ax=pdp_ax, plot_params=plot_params)

    std = ice_lines[feature_grids].std().values
    _pdp_std_plot(x=x, y=pdp_y, std=std, std_fill=std_fill, pdp_hl=pdp_hl, ax=pdp_ax, plot_params=plot_params)

    if not x_quantile and feature_type == 'numeric':
        hist_data = pdp_isolate_out.hist_data.copy()
        _draw_pdp_distplot(hist_data=hist_data, hist_ax=count_ax, vmin=np.min(x), vmax=np.max(x), plot_params=plot_params)
    else:
        _draw_pdp_countplot(count_data=count_data, count_ax=count_ax, plot_params=plot_params)
        count_ax.set_xticks(pdp_ax.get_xticks())

    count_ax.set_xbound(pdp_ax.get_xbound())

    if feature_type in ['binary', 'onehot']:
        count_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    else:
        if x_quantile:
            count_ax.set_xticklabels(count_data['display_column'].values)
    count_ax.set_xlabel(feature_name, fontsize=10)


def _pdp_std_plot(x, y, std, std_fill, pdp_hl, ax, plot_params):

    upper = y + std
    lower = y - std

    pdp_color = plot_params.get('pdp_color', '#1A4E5D')
    pdp_hl_color = plot_params.get('pdp_hl_color', '#FEDC00')
    pdp_linewidth = plot_params.get('pdp_linewidth', 2)
    zero_color = plot_params.get('zero_color', '#E75438')
    zero_linewidth = plot_params.get('zero_linewidth', 1.5)
    fill_color = plot_params.get('fill_color', '#66C2D7')
    fill_alpha = plot_params.get('fill_alpha', 0.2)
    markersize = plot_params.get('markersize', 5)

    if pdp_hl:
        ax.plot(x, y, color=pdp_hl_color, linewidth=pdp_linewidth * 3, alpha=0.8)

    ax.plot(x, y, color=pdp_color, linewidth=pdp_linewidth, marker='o', markersize=markersize)
    ax.plot(x, [0] * y, linestyle='--', linewidth=zero_linewidth, color=zero_color)

    if std_fill:
        ax.fill_between(x, upper, lower, alpha=fill_alpha, color=fill_color)

    ymin, ymax = np.min([np.min(lower) * 2, 0]), np.max([np.max(upper) * 2, 0])
    ax.set_ylim(ymin, ymax)


def _ice_line_plot(x, ice_plot_data, feature_grids, ax, plot_params):

    linewidth = np.max([1.0 / np.log10(ice_plot_data.shape[0]), 0.3])
    linealpha = np.max([1.0 / np.log10(ice_plot_data.shape[0]), 0.3])

    line_cmap = plot_params.get('line_cmap', 'Blues')
    colors = plt.get_cmap(line_cmap)(np.linspace(0, 1, 20))[5:15]

    for i in range(len(ice_plot_data)):
        y = list(ice_plot_data[feature_grids].iloc[i].values)
        ax.plot(x, y, linewidth=linewidth, c=colors[i % 10], alpha=linealpha)


def _ice_cluster_plot(x, ice_lines, feature_grids, n_cluster_centers, cluster_method, ax, plot_params):

    if cluster_method not in ['approx', 'accurate']:
        raise ValueError('cluster method: should be "approx" or "accurate".')
    if cluster_method == 'approx':
        kmeans = MiniBatchKMeans(n_clusters=n_cluster_centers, random_state=0, verbose=0)
    else:
        kmeans = KMeans(n_clusters=n_cluster_centers, random_state=0, n_jobs=1)

    kmeans.fit(ice_lines[feature_grids])
    cluster_plot_data = pd.DataFrame(kmeans.cluster_centers_, columns=feature_grids)

    cluster_cmap = plot_params.get('cluster_cmap', 'Blues')
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

    _axes_modify(font_family, ax)

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
            ax.set_xticks(range(len(pdp_interact_out.pdp_isolate_out1.display_columns)))
            ax.set_xticklabels(pdp_interact_out.pdp_isolate_out1.display_columns, rotation=xticks_rotation)

    if feature_types[1] == 'onehot':
        ax.set_yticks(range(Y.shape[0]))
        ax.set_yticklabels(pdp_interact_out.pdp_isolate_out2.display_columns)
    elif feature_types[1] == 'binary':
        ax.set_yticks([0, 1])
        ax.set_yticklabels(pdp_interact_out.pdp_isolate_out2.display_columns)
    else:
        if x_quantile:
            ax.set_yticks(range(len(pdp_interact_out.pdp_isolate_out2.display_columns)))
            ax.set_yticklabels(pdp_interact_out.pdp_isolate_out2.display_columns)

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


