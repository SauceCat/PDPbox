import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import copy

from .pdp_calc_utils import _sample_data
from .info_plot_utils import _axes_modify, _modify_legend_ax

from sklearn.cluster import MiniBatchKMeans, KMeans


def _pdp_plot_title(n_grids, feature_name, ax, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    title = plot_params.get('title', 'PDP for feature "%s"' % feature_name)
    subtitle = plot_params.get('subtitle', "Number of unique grid points: %d" % n_grids)
    title_fontsize = plot_params.get('title_fontsize', 15)
    subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)

    ax.set_facecolor('white')
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    ax.text(0, 0.5, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _draw_pdp_countplot(count_data, count_ax, pdp_ax, feature_type, display_columns, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    count_data['count_norm'] = count_data['count'] * 1.0 / count_data['count'].sum()
    count_plot_data = count_data['count_norm'].values
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(count_plot_data))
    cmap = plot_params.get('line_cmap', 'Blues')
    xticks_rotation = plot_params.get('xticks_rotation', 0)
    count_xticks_color = '#424242'
    count_xticks_size = 10

    _modify_legend_ax(count_ax, font_family=font_family)

    if feature_type == 'numeric':
        x = pdp_ax.get_xticks()[:-1] + 0.5
        count_ax.set_xticklabels(
            count_data['xticklabels'].values, color=count_xticks_color, fontsize=count_xticks_size, rotation=xticks_rotation)
    else:
        x = count_data['x'].values
        count_ax.set_xticklabels(
            display_columns, rotation=xticks_rotation, color=count_xticks_color, fontsize=count_xticks_size)

    count_ax.imshow(np.expand_dims(count_plot_data, 0), aspect="auto", cmap=cmap, norm=norm,
                    alpha=0.8, extent=(np.min(x)-0.5, np.max(x)+0.5, 0, 0.5))

    for idx in range(len(count_plot_data)):
        text_color = "black"
        if count_plot_data[idx] >= np.max(count_plot_data) * 0.5:
            text_color = "white"
        count_ax.text(x[idx], 0.25, round(count_plot_data[idx], 3), ha="center", va="center",
                      color=text_color, fontdict={'family': font_family})

    count_ax.set_xticks(x+0.5, minor=True)
    count_ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    count_ax.tick_params(which="minor", bottom=False, left=False)
    count_ax.set_title('distribution of data points',
                       fontdict={'family': font_family, 'color': count_xticks_color}, fontsize=count_xticks_size)

    count_ax.set_xticks(x)
    count_ax.set_xbound(pdp_ax.get_xbound())


def _draw_pdp_distplot(hist_data, hist_ax, vmin, vmax, plot_params):
    font_family = plot_params.get('font_family', 'Arial')

    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(hist_data))
    cmap = plot_params.get('line_cmap', 'Blues')

    hist_ax.imshow(np.expand_dims(hist_data, 0), aspect="auto", cmap=cmap, norm=norm,
                   extent=(vmin, vmax, 0, 0.5), alpha=0.8)
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
    percentile_xticklabels = list(percentile_info)

    if feature_type == 'binary' or feature_type == 'onehot' or x_quantile:
        # x for original pdp
        # anyway, pdp is started from x=0
        x = range(len(feature_grids))
        xticks = x
        xticklabels = list(display_columns)

        if count_ax is not None:
            if x_quantile:
                count_display_columns = count_data['xticklabels'].values
                # number of grids = number of bins + 1
                # need to include the final point: + 2
                count_x = range(count_data['x'].min(), count_data['x'].max() + 2)
                xticks = count_x
                if count_x[0] == -1:
                    xticklabels = [float(count_display_columns[0].split(',')[0].replace('[', ''))] + xticklabels
                    percentile_xticklabels = ['(0.0)'] + percentile_xticklabels
                if count_x[-1] == len(feature_grids):
                    xticklabels = xticklabels + [float(count_display_columns[-1].split(',')[1].replace(']', ''))]
                    percentile_xticklabels = percentile_xticklabels + ['(100.0)']
            else:
                xticklabels = []
            pdp_ax.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)

        pdp_ax.set_xticks(xticks)
        pdp_ax.set_xticklabels(xticklabels, rotation=xticks_rotation)
    else:
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

    if count_ax is not None:
        if not x_quantile and feature_type == 'numeric':
            # in this situation, x=feature_grids
            hist_data = pdp_isolate_out.hist_data.copy()
            _draw_pdp_distplot(hist_data=hist_data, hist_ax=count_ax, vmin=np.min(x), vmax=np.max(x),
                               plot_params=plot_params)
        else:
            _draw_pdp_countplot(count_data=count_data, count_ax=count_ax, pdp_ax=pdp_ax, feature_type=feature_type,
                                display_columns=display_columns, plot_params=plot_params)
        count_ax.set_xlabel(feature_name, fontsize=11, fontdict={'family': font_family})

    else:
        pdp_ax.set_xlabel(feature_name, fontsize=11, fontdict={'family': font_family})

    if show_percentile and len(percentile_info) > 0:
        percentile_pdp_ax = pdp_ax.twiny()
        percentile_pdp_ax.set_xticks(pdp_ax.get_xticks())
        percentile_pdp_ax.set_xbound(pdp_ax.get_xbound())
        percentile_pdp_ax.set_xticklabels(percentile_xticklabels, rotation=xticks_rotation)
        percentile_pdp_ax.set_xlabel('percentile info')
        _axes_modify(font_family=font_family, ax=percentile_pdp_ax, top=True)


def _pdp_std_plot(x, y, std, std_fill, pdp_hl, ax, plot_params):

    upper = y + std
    lower = y - std

    pdp_color = plot_params.get('pdp_color', '#1A4E5D')
    pdp_hl_color = plot_params.get('pdp_hl_color', '#FEDC00')
    pdp_linewidth = plot_params.get('pdp_linewidth', 1.5)
    zero_color = plot_params.get('zero_color', '#E75438')
    zero_linewidth = plot_params.get('zero_linewidth', 1)
    fill_color = plot_params.get('fill_color', '#66C2D7')
    fill_alpha = plot_params.get('fill_alpha', 0.2)
    markersize = plot_params.get('markersize', 3.5)

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


def _pdp_contour_plot(X, Y, pdp_mx, inter_ax, cmap, norm, inter_fill_alpha, fontsize, plot_params):

    contour_color = plot_params.get('contour_color', 'white')

    level = np.min([X.shape[0], X.shape[1]])
    c1 = inter_ax.contourf(X, Y, pdp_mx, N=level, origin='lower', cmap=cmap, norm=norm, alpha=inter_fill_alpha)
    c2 = inter_ax.contour(c1, levels=c1.levels, colors=contour_color, origin='lower')
    inter_ax.clabel(c2, contour_label_fontsize=fontsize, inline=1)


def _pdp_inter_grid(pdp_mx, inter_ax, cmap, norm, inter_fill_alpha, fontsize, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    inter_ax.imshow(pdp_mx, cmap=cmap, norm=norm, alpha=inter_fill_alpha, origin='lower')

    for r in range(pdp_mx.shape[0]):
        for c in range(pdp_mx.shape[1]):
            text_color = 'w'
            if pdp_mx[r, c] >= norm.vmin + (norm.vmax - norm.vmin) * 0.5:
                text_color = 'black'
            inter_ax.text(c, r, round(pdp_mx[r, c], 3), ha="center", va="center", color=text_color,
                          size=fontsize, fontdict={'family': font_family})

    # draw the white gaps
    inter_ax.set_xticks(np.arange(pdp_mx.shape[1] - 1) + 0.5, minor=True)
    inter_ax.set_yticks(np.arange(pdp_mx.shape[0] - 1) + 0.5, minor=True)
    inter_ax.grid(which="minor", color="w", linestyle='-', linewidth=1.5)


def _pdp_inter_one(pdp_interact_out, feature_names, plot_type, inter_ax, x_quantile, plot_params, norm, ticks=True):
    cmap = plot_params.get('cmap', 'viridis')
    inter_fill_alpha = plot_params.get('inter_fill_alpha', 0.8)
    fontsize = plot_params.get('inter_fontsize', 9)
    font_family = plot_params.get('font_family', 'Arial')

    # prepare pdp_mx
    pdp_mx_temp = copy.deepcopy(pdp_interact_out.pdp)
    for feature, feature_type, mark in zip(pdp_interact_out.features, pdp_interact_out.feature_types, ['x', 'y']):
        if feature_type in ['numeric', 'binary']:
            pdp_mx_temp[mark] = pdp_mx_temp[feature]
        else:
            pdp_mx_temp[mark] = pdp_mx_temp[feature].apply(lambda x : list(x).index(1), axis=1)
    pdp_mx_temp = pdp_mx_temp[['x', 'y', 'preds']].sort_values(by=['x', 'y'], ascending=True)

    pdp_inter = copy.deepcopy(pdp_mx_temp['preds'].values)
    n_grids_x, n_grids_y = len(pdp_interact_out.feature_grids[0]), len(pdp_interact_out.feature_grids[1])
    pdp_mx = pdp_inter.reshape((n_grids_x, n_grids_y)).T

    # if it is called by _pdp_inter_three, norm is not None
    if norm is None:
        pdp_min, pdp_max = np.min(pdp_inter), np.max(pdp_inter)
        norm = mpl.colors.Normalize(vmin=pdp_min, vmax=pdp_max)

    inter_params = {
        'pdp_mx': pdp_mx, 'inter_ax': inter_ax, 'cmap': cmap, 'norm': norm,
        'inter_fill_alpha': inter_fill_alpha, 'fontsize': fontsize, 'plot_params': plot_params
    }
    if plot_type == 'contour':
        if x_quantile:
            X, Y = np.meshgrid(range(pdp_mx.shape[1]), range(pdp_mx.shape[0]))
        else:
            X, Y = np.meshgrid(pdp_interact_out.feature_grids[0], pdp_interact_out.feature_grids[1])
        _pdp_contour_plot(X=X, Y=Y, **inter_params)
    elif plot_type == 'grid':
        _pdp_inter_grid(**inter_params)
    else:
        raise ValueError("plot_type: should be 'contour' or 'grid'")

    # if it is called by _pdp_inter_three, ticks is False
    if ticks:
        _axes_modify(font_family=font_family, ax=inter_ax, grid=True)

        if pdp_interact_out.feature_types[0] != 'numeric' or x_quantile:
            inter_ax.set_xticks(range(len(pdp_interact_out.feature_grids[0])))
            inter_ax.set_xticklabels(pdp_interact_out.feature_grids[0])

        if pdp_interact_out.feature_types[1] != 'numeric' or x_quantile:
            inter_ax.set_yticks(range(len(pdp_interact_out.feature_grids[1])))
            inter_ax.set_yticklabels(pdp_interact_out.feature_grids[1])

        inter_ax.set_xlabel(feature_names[0], fontsize=10, fontdict={'family': font_family})
        inter_ax.set_ylabel(feature_names[1], fontsize=10, fontdict={'family': font_family})
    inter_ax.tick_params(which="minor", bottom=False, left=False)


def _pdp_xy(pdp_values, vmean, pdp_ax, ticklabels, feature_name, cmap, norm, plot_type, plot_params, y=False):
    font_family = plot_params.get('font_family', 'Arial')
    fontsize = plot_params.get('inter_fontsize', 9)
    inter_fill_alpha = plot_params.get('inter_fill_alpha', 0.8)

    pdp_ax.imshow(np.expand_dims(pdp_values, int(y)), cmap=cmap, norm=norm, origin='lower', alpha=inter_fill_alpha)

    for idx in range(len(pdp_values)):
        text_color = 'w'
        if pdp_values[idx] >= vmean:
            text_color = 'black'

        text_params = {'s': round(pdp_values[idx], 3), 'ha': 'center', 'va': 'center', 'color': text_color,
                       'size': fontsize, 'fontdict': {'family': font_family}}
        if y:
            pdp_ax.text(x=0, y=idx, rotation='vertical', **text_params)
        else:
            pdp_ax.text(x=idx, y=0, **text_params)

    pdp_ax.set_frame_on(False)

    if y:
        pdp_ax.set_yticks(np.arange(len(pdp_values) - 1) + 0.5, minor=True)
        pdp_ax.set_yticks(range(len(ticklabels)))
        pdp_ax.set_yticklabels(ticklabels)
        pdp_ax.set_ylabel(feature_name, fontdict={'family': font_family, 'fontsize': 10})
        if plot_type == 'contour':
            pdp_ax.get_yaxis().set_label_position('right')
        pdp_ax.set_xticks([])
    else:
        pdp_ax.set_xticks(np.arange(len(pdp_values) - 1) + 0.5, minor=True)
        pdp_ax.set_xticks(range(len(ticklabels)))
        pdp_ax.get_xaxis().tick_top()
        pdp_ax.set_xticklabels(ticklabels)
        pdp_ax.set_xlabel(feature_name, fontdict={'family': font_family, 'fontsize': 10})
        if plot_type == 'grid':
            pdp_ax.get_xaxis().set_label_position('top')
        pdp_ax.set_yticks([])

    pdp_ax.grid(which="minor", color="w", linestyle='-', linewidth=1.5)
    pdp_ax.tick_params(which="minor", top=False, left=False)
    pdp_ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='#424242', colors='#9E9E9E')


def _pdp_inter_three(pdp_interact_out, feature_names, plot_type, chart_grids, x_quantile, fig, plot_params):
    cmap = plot_params.get('cmap', 'viridis')

    inter_ax = plt.subplot(chart_grids[3])
    fig.add_subplot(inter_ax)
    pdp_x_ax = plt.subplot(chart_grids[1], sharex=inter_ax)
    fig.add_subplot(pdp_x_ax)
    pdp_y_ax = plt.subplot(chart_grids[2], sharey=inter_ax)
    fig.add_subplot(pdp_y_ax)

    pdp_x = copy.deepcopy(pdp_interact_out.pdp_isolate_out1.pdp)
    pdp_y = copy.deepcopy(pdp_interact_out.pdp_isolate_out2.pdp)
    pdp_inter = copy.deepcopy(pdp_interact_out.pdp['preds'].values)
    pdp_values = np.concatenate((pdp_x, pdp_y, pdp_inter))
    pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)

    norm = mpl.colors.Normalize(vmin=pdp_min, vmax=pdp_max)
    vmean = norm.vmin + (norm.vmax - norm.vmin) * 0.5
    feature_grids = pdp_interact_out.feature_grids

    pdp_xy_params = {'cmap': cmap, 'norm': norm, 'vmean': vmean, 'plot_params': plot_params, 'plot_type': plot_type}
    _pdp_xy(pdp_values=pdp_x, pdp_ax=pdp_x_ax, ticklabels=feature_grids[0], feature_name=feature_names[0], y=False, **pdp_xy_params)
    _pdp_xy(pdp_values=pdp_y, pdp_ax=pdp_y_ax, ticklabels=feature_grids[1], feature_name=feature_names[1], y=True, **pdp_xy_params)

    _pdp_inter_one(pdp_interact_out=pdp_interact_out, feature_names=feature_names, plot_type=plot_type,
                   inter_ax=inter_ax, x_quantile=x_quantile, plot_params=plot_params, norm=norm, ticks=False)

    inter_ax.set_frame_on(False)
    plt.setp(inter_ax.get_xticklabels(), visible=False)
    plt.setp(inter_ax.get_yticklabels(), visible=False)
    inter_ax.tick_params(which="minor", bottom=False, left=False)
    inter_ax.tick_params(which="major", bottom=False, left=False)

    return {
        'pdp_x_ax': pdp_x_ax,
        'pdp_y_ax': pdp_y_ax,
        'pdp_inter_ax': inter_ax
    }





