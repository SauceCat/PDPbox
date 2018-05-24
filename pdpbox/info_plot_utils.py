
from __future__ import absolute_import

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

from .pdp_plot_utils import _axes_modify
from .pdp_calc_utils import (_get_grids, _find_bucket, _make_bucket_column_names, _find_onehot_actual,
                             _make_bucket_column_names_percentile)


def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                    grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
    display_columns = bound_ups = bound_lows = []
    percentile_columns = percentile_bound_lows = percentile_bound_ups = []
    data_x = data.copy()

    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        display_columns = ['%s_0' % feature, '%s_1' % feature]
        data_x['x'] = data_x[feature]
    if feature_type == 'numeric':
        percentile_info = None
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(
                x=data_x[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))

        if not show_outliers:
            data_x = data_x[(data_x[feature] >= feature_grids[0])
                            & (data_x[feature] <= feature_grids[-1])].reset_index(drop=True)

        # map feature value into value buckets
        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids, endpoint=endpoint))
        uni_xs = sorted(data_x['x'].unique())

        # create bucket names
        display_columns, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=feature_grids, endpoint=endpoint)
        display_columns = np.array(display_columns)[range(uni_xs[0], uni_xs[-1]+1)]
        bound_lows = np.array(bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
        bound_ups = np.array(bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]

        # create percentile bucket names
        if show_percentile and grid_type == 'percentile':
            percentile_columns, percentile_bound_lows, percentile_bound_ups = \
                _make_bucket_column_names_percentile(percentile_info=percentile_info, endpoint=endpoint)
            percentile_columns = np.array(percentile_columns)[range(uni_xs[0], uni_xs[-1]+1)]
            percentile_bound_lows = np.array(percentile_bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
            percentile_bound_ups = np.array(percentile_bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]

        # adjust results
        data_x['x'] = data_x['x'] - data_x['x'].min()

    if feature_type == 'onehot':
        feature_grids = display_columns = np.array(feature)
        data_x['x'] = data_x[feature].apply(lambda x: _find_onehot_actual(x=x), axis=1)
        data_x = data_x[~data_x['x'].isnull()].reset_index(drop=True)

    data_x['x'] = data_x['x'].map(int)
    results = {
        'data': data_x,
        'value_display': (list(display_columns), list(bound_lows), list(bound_ups)),
        'percentile_display': (list(percentile_columns), list(percentile_bound_lows), list(percentile_bound_ups))
    }

    return results


def _autolabel(rects, ax, bar_color):
    for rect in rects:
        height = rect.get_height()
        bbox_props = {'facecolor': 'white', 'edgecolor': bar_color, 'boxstyle': "square,pad=0.5"}
        ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=bar_color)


def _draw_barplot(bar_data, bar_ax, display_columns, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    bar_color = plot_params.get('bar_color', '#5BB573')
    bar_width = plot_params.get('bar_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))

    # add value label for bar plot
    rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=bar_width, color=bar_color, alpha=0.5)
    _autolabel(rects=rects, ax=bar_ax, bar_color=bar_color)
    _axes_modify(font_family=font_family, ax=bar_ax)


def _draw_lineplot(line_data, line_ax, line_color, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    line_width = plot_params.get('line_width', 1)

    line_ax.plot(line_data['x'], line_data['y'], linewidth=line_width, c=line_color, marker='o')
    for idx in line_data.index.values:
        bbox_props = {'facecolor': line_color, 'edgecolor': 'none', 'boxstyle': "square,pad=0.5"}
        line_ax.text(line_data.loc[idx, 'x'], line_data.loc[idx, 'y'],
                     '%.3f' % line_data.loc[idx, 'y'],
                     ha="center", va="top", size=10, bbox=bbox_props, color='#ffffff')

    _axes_modify(font_family=font_family, ax=line_ax)


def _draw_bar_line(bar_data, bar_ax, line_data, line_ax, line_color,
                   feature_name, display_columns, percentile_columns, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)

    # bar plot
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel('count')

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)

    # display percentile
    if len(percentile_columns) > 0:
        percentile_ax = bar_ax.twiny()
        percentile_ax.set_xticks(bar_ax.get_xticks())
        percentile_ax.set_xbound(bar_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)

    _draw_lineplot(line_data=line_data, line_ax=line_ax, line_color=line_color, plot_params=plot_params)
    line_ax.get_yaxis().tick_right()
    line_ax.grid(False)
    line_ax.set_ylabel('target_avg')


def _info_plot_title(title, subtitle, ax, plot_params):
    font_family = plot_params.get('font_family', 'Arial')
    title_fontsize = plot_params.get('title_fontsize', 15)
    subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)

    ax.set_facecolor('white')
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    ax.text(0, 0.5, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _target_plot(feature_name, display_columns, percentile_columns, target, bar_data,
                 target_lines, figsize, ncols, plot_params):
    # set up graph parameters
    width, height = 15, 9
    nrows = 1

    if len(target) > 1:
        nrows = int(np.ceil(len(target) * 1.0 / ncols))
        ncols = np.min([len(target), ncols])
        width = np.min([7.5 * len(target), 15])
        height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Arial')
    line_color = plot_params.get('line_color', '#1A4E5D')
    line_colors_cmap = plot_params.get('line_colors_cmap', 'tab20')
    line_colors = plot_params.get('line_colors', plt.get_cmap(line_colors_cmap)(
        range(np.min([20, len(target)]))))
    title = plot_params.get('title', 'Target plot for feature "%s"' % feature_name)
    subtitle = plot_params.get('subtitle', 'Average target value through different feature values.')

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height-2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    _info_plot_title(title=title, subtitle=subtitle, ax=title_ax, plot_params=plot_params)

    if len(target) == 1:
        bar_ax = plt.subplot(outer_grid[1])
        fig.add_subplot(bar_ax)
        line_ax = bar_ax.twinx()

        line_data = target_lines[0].rename(columns={target[0]: 'y'}).sort_values('x', ascending=True)
        _draw_bar_line(bar_data=bar_data, bar_ax=bar_ax, line_data=line_data, line_ax=line_ax, line_color=line_color,
                       feature_name=feature_name, display_columns=display_columns,
                       percentile_columns=percentile_columns, plot_params=plot_params)

    else:
        inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.35)
        plot_axes = []
        for inner_idx in range(len(target)):
            ax = plt.subplot(inner_grid[inner_idx])
            plot_axes.append(ax)
            fig.add_subplot(ax)

        bar_ax = []
        line_ax = []

        # get max average target value
        ys = []
        for target_idx in range(len(target)):
            ys += list(target_lines[target_idx][target[target_idx]].values)
        y_max = np.max(ys)

        for target_idx in range(len(target)):
            inner_line_color = line_colors[target_idx % len(line_colors)]
            inner_bar_ax = plot_axes[target_idx]
            inner_line_ax = inner_bar_ax.twinx()

            line_data = target_lines[target_idx].rename(
                columns={target[target_idx]: 'y'}).sort_values('x', ascending=True)

            _draw_bar_line(bar_data=bar_data, bar_ax=inner_bar_ax, line_data=line_data, line_ax=inner_line_ax,
                           line_color=inner_line_color, feature_name=feature_name, display_columns=display_columns,
                           percentile_columns=percentile_columns, plot_params=plot_params)

            subplot_title = target[target_idx]
            if len(percentile_columns) > 0:
                subplot_title += '\n\n'
            plot_axes[target_idx].set_title(subplot_title, fontdict={'fontsize': 13, 'fontname': font_family})

            inner_line_ax.set_ylim(0., y_max)
            bar_ax.append(inner_bar_ax)
            line_ax.append(inner_line_ax)

            if target_idx % ncols != 0:
                inner_bar_ax.set_yticklabels([])

            if (target_idx % ncols + 1 != ncols) and target_idx != len(plot_axes) - 1:
                inner_line_ax.set_yticklabels([])

        if len(plot_axes) > len(target):
            for idx in range(len(target), len(plot_axes)):
                plot_axes[idx].axis('off')

    axes = {'title_ax': title_ax, 'bar_ax': bar_ax, 'line_ax': line_ax}
    return fig, axes


def _draw_boxplot(box_data, box_line_data, box_ax, display_columns, box_color, plot_params):
    font_family = plot_params.get('font_family', 'Arial')
    box_line_width = plot_params.get('box_line_width', 1.5)
    box_width = plot_params.get('box_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))

    xs = sorted(box_data['x'].unique())
    ys = []
    for x in xs:
        ys.append(box_data[box_data['x'] == x]['y'].values)

    boxprops = dict(linewidth=box_line_width, color=box_color)
    medianprops = dict(linewidth=0)
    whiskerprops = dict(linewidth=box_line_width, color=box_color)
    capprops = dict(linewidth=box_line_width, color=box_color)

    box_ax.boxplot(ys, positions=xs, showfliers=False, widths=box_width,
                   whiskerprops=whiskerprops, capprops=capprops, boxprops=boxprops, medianprops=medianprops)

    _axes_modify(font_family=font_family, ax=box_ax)

    box_ax.plot(box_line_data['x'], box_line_data['y'], linewidth=1, c=box_color, linestyle='--')
    for idx in box_line_data.index.values:
        bbox_props = {'facecolor': 'white', 'edgecolor': box_color,
                      'boxstyle': "square,pad=0.5", 'lw': 1}
        box_ax.text(box_line_data.loc[idx, 'x'], box_line_data.loc[idx, 'y'], '%.3f' % box_line_data.loc[idx, 'y'],
                    ha="center", va="top", size=10, bbox=bbox_props, color=box_color)


def _draw_box_bar(bar_data, bar_ax, box_data, box_line_data, box_color, box_ax,
                  feature_name, display_columns, percentile_columns, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    _draw_boxplot(box_data=box_data, box_line_data=box_line_data, box_ax=box_ax,
                  display_columns=display_columns, box_color=box_color, plot_params=plot_params)
    box_ax.set_ylabel('actual_prediction')
    box_ax.set_xticklabels([])

    _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)

    # bar plot
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel('count')

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)

    plt.setp(box_ax.get_xticklabels(), visible=False)

    # display percentile
    if len(percentile_columns) > 0:
        percentile_ax = box_ax.twiny()
        percentile_ax.set_xticks(box_ax.get_xticks())
        percentile_ax.set_xbound(box_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)


def _actual_plot(plot_data, bar_data, box_lines, actual_prediction_columns, feature_name,
                 display_columns, percentile_columns, figsize, ncols, plot_params):
    # set up graph parameters
    width, height = 15, 9
    nrows = 1

    if plot_params is None:
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Arial')
    box_color = plot_params.get('box_color', '#3288bd')
    box_colors_cmap = plot_params.get('box_colors_cmap', 'tab20')
    box_colors = plot_params.get('box_colors', plt.get_cmap(box_colors_cmap)(
        range(np.min([20, len(actual_prediction_columns)]))))
    title = plot_params.get('title', 'Actual predictions plot for %s' % feature_name)
    subtitle = plot_params.get('subtitle', 'Distribution of actual prediction through different feature values.')

    if len(actual_prediction_columns) > 1:
        nrows = int(np.ceil(len(actual_prediction_columns) * 1.0 / ncols))
        ncols = np.min([len(actual_prediction_columns), ncols])
        width = np.min([7.5 * len(actual_prediction_columns), 15])
        height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height-2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    _info_plot_title(title=title, subtitle=subtitle, ax=title_ax, plot_params=plot_params)

    if len(actual_prediction_columns) == 1:
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1])

        box_ax = plt.subplot(inner_grid[0])
        bar_ax = plt.subplot(inner_grid[1], sharex=box_ax)
        fig.add_subplot(box_ax)
        fig.add_subplot(bar_ax)

        box_data = plot_data[['x', actual_prediction_columns[0]]].rename(columns={actual_prediction_columns[0]: 'y'})
        box_line_data = box_lines[0].rename(columns={actual_prediction_columns[0] + '_q2': 'y'})
        _draw_box_bar(bar_data=bar_data, bar_ax=bar_ax, box_data=box_data, box_line_data=box_line_data,
                      box_color=box_color, box_ax=box_ax, feature_name=feature_name, display_columns=display_columns,
                      percentile_columns=percentile_columns, plot_params=plot_params)
    else:
        inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.35)

        box_ax = []
        bar_ax = []

        # get max average target value
        ys = []
        for idx in range(len(box_lines)):
            ys += list(box_lines[idx][actual_prediction_columns[idx] + '_q2'].values)
        y_max = np.max(ys)

        for idx in range(len(actual_prediction_columns)):
            box_color = box_colors[idx % len(box_colors)]

            inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_grid[idx], wspace=0, hspace=0.2)
            inner_box_ax = plt.subplot(inner[0])
            inner_bar_ax = plt.subplot(inner[1], sharex=inner_box_ax)
            fig.add_subplot(inner_box_ax)
            fig.add_subplot(inner_bar_ax)

            inner_box_data = plot_data[['x', actual_prediction_columns[idx]]].rename(
                columns={actual_prediction_columns[idx]: 'y'})
            inner_box_line_data = box_lines[idx].rename(columns={actual_prediction_columns[idx] + '_q2': 'y'})
            _draw_box_bar(bar_data=bar_data, bar_ax=inner_bar_ax, box_data=inner_box_data,
                          box_line_data=inner_box_line_data, box_color=box_color,
                          box_ax=inner_box_ax, feature_name=feature_name, display_columns=display_columns,
                          percentile_columns=percentile_columns, plot_params=plot_params)

            subplot_title = 'target_%s' % actual_prediction_columns[idx].split('_')[-1]
            if len(percentile_columns) > 0:
                subplot_title += '\n\n'
            inner_box_ax.set_title(subplot_title, fontdict={'fontsize': 13, 'fontname': font_family})
            inner_box_ax.set_ylim(0., y_max)

            if idx % ncols != 0:
                inner_bar_ax.set_yticklabels([])
                inner_box_ax.set_yticklabels([])

            box_ax.append(inner_box_ax)
            bar_ax.append(inner_bar_ax)

    axes = {'title_ax': title_ax, 'box_ax': box_ax, 'bar_ax': bar_ax}
    return fig, axes


def _plot_interact(plot_data, y, plot_ax, feature_names, display_columns, percentile_columns,
                   marker_sizes, cmap, line_width, xticks_rotation, font_family):
    plot_ax.set_xticks(range(len(display_columns[0])))
    plot_ax.set_xticklabels(display_columns[0], rotation=xticks_rotation)
    plot_ax.set_yticks(range(len(display_columns[1])))
    plot_ax.set_yticklabels(display_columns[1])
    plot_ax.set_xlim(-0.5, len(display_columns[0]) - 0.5)
    plot_ax.set_ylim(-0.5, len(display_columns[1]) - 0.5)

    percentile_ax = None
    percentile_ay = None
    # display percentile
    if len(percentile_columns[0]) > 0:
        percentile_ax = plot_ax.twiny()
        percentile_ax.set_xticks(plot_ax.get_xticks())
        percentile_ax.set_xbound(plot_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns[0], rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)

    # display percentile
    if len(percentile_columns[1]) > 0:
        percentile_ay = plot_ax.twinx()
        percentile_ay.set_yticks(plot_ax.get_yticks())
        percentile_ay.set_ybound(plot_ax.get_ybound())
        percentile_ay.set_yticklabels(percentile_columns[1])
        percentile_ay.set_ylabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ay, right=True)

    value_min, value_max = plot_data[y].min(), plot_data[y].max()
    colors = [plt.get_cmap(cmap)(float(v - value_min) / (value_max - value_min))
              for v in plot_data[y].values]

    plot_ax.scatter(plot_data['x1'].values, plot_data['x2'].values,
                    s=marker_sizes, c=colors, linewidth=line_width, edgecolors=plt.get_cmap(cmap)(1.))
    plot_ax.set_xlabel(feature_names[0])
    plot_ax.set_ylabel(feature_names[1])

    _axes_modify(font_family=font_family, ax=plot_ax)
    return value_min, value_max, percentile_ax, percentile_ay


def _modify_legend_ax(ax, font_family):
    for d in ['top', 'bottom', 'right', 'left']:
        ax.spines[d].set_visible(False)

    for tick in ax.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_family)

    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_legend_colorbar(value_min, value_max, colorbar_ax, cmap, font_family, height="30%", width="100%"):
    norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))

    # color bar
    cax = inset_axes(colorbar_ax, height=height, width=width, loc=10)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=norm, orientation='horizontal')
    cb.outline.set_linewidth(0)
    cb.set_ticks([])

    width_float = float(width.replace('%', '')) / 100
    cb.ax.text((1. - width_float) / 2, 0.5, round(value_min, 3), fontdict={'color': '#424242', 'fontsize': 10},
               transform=colorbar_ax.transAxes, horizontalalignment='left', verticalalignment='center')

    cb.ax.text(1. - (1. - width_float) / 2, 0.5, round(value_max, 3), fontdict={'color': '#ffffff', 'fontsize': 10},
               transform=colorbar_ax.transAxes, horizontalalignment='right', verticalalignment='center')
    _modify_legend_ax(colorbar_ax, font_family)


def _plot_legend_circles(count_min, count_max, circle_ax, cmap, font_family):
    # circle size
    circle_ax.scatter([0.75, 2], [1] * 2, s=[500, 1500],
                      edgecolors=plt.get_cmap(cmap)(1.), color='white')
    circle_ax.set_xlim(0., 3)

    circle_ax.text(0.75, 1., count_min, fontdict={'color': '#424242', 'fontsize': 10},
                   horizontalalignment='center', verticalalignment='center')
    circle_ax.text(1.275, 1., '-->', fontdict={'color': '#424242', 'fontsize': 10, 'weight': 'bold'},
                   horizontalalignment='center', verticalalignment='center')
    circle_ax.text(2, 1., count_max, fontdict={'color': '#424242', 'fontsize': 10},
                   horizontalalignment='center', verticalalignment='center')
    _modify_legend_ax(circle_ax, font_family)


def _info_plot_interact(feature_names, display_columns, percentile_columns, ys,
                        plot_data, title, subtitle, figsize, ncols, plot_params):

    width, height = 15, 9
    nrows = 1

    if len(ys) > 1:
        nrows = int(np.ceil(len(ys) * 1.0 / ncols))
        ncols = np.min([len(ys), ncols])
        width = np.min([7.5 * len(ys), 15])
        height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    font_family = plot_params.get('font_family', 'Arial')
    cmap = plot_params.get('cmap', 'Blues')
    cmaps = plot_params.get('cmaps', ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys'])
    line_width = plot_params.get('line_width', 1)
    xticks_rotation = plot_params.get('xticks_rotation', 0)
    marker_size_min = plot_params.get('marker_size_min', 50)
    marker_size_max = plot_params.get('marker_size_max', 1500)

    # plot title
    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height-2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    _info_plot_title(title=title, subtitle=subtitle, ax=title_ax, plot_params=plot_params)

    # draw value plots and legend
    count_min, count_max = plot_data['fake_count'].min(), plot_data['fake_count'].max()
    marker_sizes = []
    for count in plot_data['fake_count'].values:
        size = float(count - count_min) / (count_max - count_min) * (marker_size_max - marker_size_min) + marker_size_min
        marker_sizes.append(size)

    if len(ys) == 1:
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1], height_ratios=[height-3, 1])
        value_ax = plt.subplot(inner_grid[0])
        fig.add_subplot(value_ax)
        value_min, value_max, percentile_ax, percentile_ay = _plot_interact(
            plot_data=plot_data, y=ys[0], plot_ax=value_ax, feature_names=feature_names,
            display_columns=display_columns, percentile_columns=percentile_columns,
            marker_sizes=marker_sizes, cmap=cmap, line_width=line_width, xticks_rotation=xticks_rotation,
            font_family=font_family)

        # draw legend
        # _, legend_ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(legend_width, 1))
        legend_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=inner_grid[1], wspace=0)
        legend_ax = [plt.subplot(legend_grid[0]), plt.subplot(legend_grid[1])]
        fig.add_subplot(legend_ax[0])
        fig.add_subplot(legend_ax[1])
        _plot_legend_colorbar(value_min=value_min, value_max=value_max, colorbar_ax=legend_ax[0],
                              cmap=cmap, font_family=font_family)
        _plot_legend_circles(count_min=count_min, count_max=count_max, circle_ax=legend_ax[1],
                             cmap=cmap, font_family=font_family)

    else:
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1], height_ratios=[1, height - 3])
        legend_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=inner_grid[0], wspace=0)
        legend_ax = plt.subplot(legend_grid[0])
        fig.add_subplot(legend_ax)

        _plot_legend_circles(count_min=count_min, count_max=count_max, circle_ax=legend_ax,
                             cmap=cmap, font_family=font_family)

        value_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=inner_grid[1], wspace=0.2, hspace=0.35)
        value_ax = []
        for y_idx in range(len(ys)):
            ax = plt.subplot(value_grid[y_idx])
            value_ax.append(ax)
            fig.add_subplot(ax)

        for idx in range(len(ys)):

            cmap_idx = cmaps[idx % len(cmaps)]

            value_min, value_max, percentile_ax, percentile_ay = _plot_interact(
                plot_data=plot_data, y=ys[idx], plot_ax=value_ax[idx], feature_names=feature_names,
                display_columns=display_columns, percentile_columns=percentile_columns,
                marker_sizes=marker_sizes, cmap=cmap_idx, line_width=line_width,
                xticks_rotation=xticks_rotation, font_family=font_family)

            colorbar_ax = make_axes_locatable(value_ax[idx]).append_axes('bottom', size='5%', pad=0.5)
            _plot_legend_colorbar(value_min=value_min, value_max=value_max, colorbar_ax=colorbar_ax,
                                  cmap=cmap_idx, font_family=font_family, height="90%", width="80%")

            subplot_title = ys[idx]
            if len(percentile_columns[1]) > 0:
                subplot_title += '\n\n\n'
            value_ax[idx].set_title(subplot_title, fontdict={'fontsize': 12, 'fontname': font_family})

            if idx % ncols != 0:
                value_ax[idx].set_yticklabels([])

            if (idx % ncols + 1 != ncols) and idx != len(value_ax) - 1:
                if percentile_ay is not None:
                    percentile_ay.set_yticklabels([])

        if len(value_ax) > len(ys):
            for idx in range(len(ys), len(value_ax)):
                value_ax[idx].axis('off')

    axes = {'title_ax': title_ax, 'value_ax': value_ax, 'legend_ax': legend_ax}
    return fig, axes


def _prepare_info_plot_data(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                            grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
    prepared_results = _prepare_data_x(
        feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points, grid_type=grid_type,
        percentile_range=percentile_range, grid_range=grid_range, cust_grid_points=cust_grid_points,
        show_percentile=show_percentile, show_outliers=show_outliers, endpoint=endpoint)
    data_x = prepared_results['data']
    display_columns, bound_lows, bound_ups = prepared_results['value_display']
    percentile_columns, percentile_bound_lows, percentile_bound_ups = prepared_results['percentile_display']

    data_x['fake_count'] = 1
    bar_data = data_x.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)
    summary_df = pd.DataFrame(np.arange(data_x['x'].min(), data_x['x'].max() + 1), columns=['x'])
    summary_df = summary_df.merge(bar_data.rename(columns={'fake_count': 'count'}), on='x', how='left').fillna(0)

    summary_df['display_column'] = summary_df['x'].apply(lambda x: display_columns[int(x)])
    info_cols = ['x', 'display_column']
    if feature_type == 'numeric':
        summary_df['value_lower'] = summary_df['x'].apply(lambda x: bound_lows[int(x)])
        summary_df['value_upper'] = summary_df['x'].apply(lambda x: bound_ups[int(x)])
        info_cols += ['value_lower', 'value_upper']

    if len(percentile_columns) != 0:
        summary_df['percentile_column'] = summary_df['x'].apply(lambda x: percentile_columns[int(x)])
        summary_df['percentile_lower'] = summary_df['x'].apply(lambda x: percentile_bound_lows[int(x)])
        summary_df['percentile_upper'] = summary_df['x'].apply(lambda x: percentile_bound_ups[int(x)])
        info_cols += ['percentile_column', 'percentile_lower', 'percentile_upper']

    return data_x, bar_data, summary_df, info_cols, display_columns, percentile_columns


def _prepare_info_plot_interact_data(data_input, features, feature_types, num_grid_points, grid_types,
                                     percentile_ranges, grid_ranges, cust_grid_points, show_percentile,
                                     show_outliers, endpoint, agg_dict):
    prepared_results = []
    for i in range(2):
        prepared_result = _prepare_data_x(
            feature=features[i], feature_type=feature_types[i], data=data_input,
            num_grid_points=num_grid_points[i], grid_type=grid_types[i], percentile_range=percentile_ranges[i],
            grid_range=grid_ranges[i], cust_grid_points=cust_grid_points[i],
            show_percentile=show_percentile, show_outliers=show_outliers[i], endpoint=endpoint)
        prepared_results.append(prepared_result)
        if i == 0:
            data_input = prepared_result['data'].rename(columns={'x': 'x1'})

    data_x = prepared_results[1]['data'].rename(columns={'x': 'x2'})
    data_x['fake_count'] = 1
    plot_data = data_x.groupby(['x1', 'x2'], as_index=False).agg(agg_dict)

    return data_x, plot_data, prepared_results


def _prepare_info_plot_interact_summary(data_x, plot_data, prepared_results, feature_types):
    x1_values = []
    x2_values = []
    for x1_value in range(data_x['x1'].min(), data_x['x1'].max() + 1):
        for x2_value in range(data_x['x2'].min(), data_x['x2'].max() + 1):
            x1_values.append(x1_value)
            x2_values.append(x2_value)
    summary_df = pd.DataFrame()
    summary_df['x1'] = x1_values
    summary_df['x2'] = x2_values
    summary_df = summary_df.merge(plot_data.rename(columns={'fake_count': 'count'}),
                                  on=['x1', 'x2'], how='left').fillna(0)

    info_cols = ['x1', 'x2', 'display_column_1', 'display_column_2']
    display_columns = []
    percentile_columns = []
    for i in range(2):
        display_columns_i, bound_lows_i, bound_ups_i = prepared_results[i]['value_display']
        percentile_columns_i, percentile_bound_lows_i, percentile_bound_ups_i = prepared_results[i]['percentile_display']
        display_columns.append(display_columns_i)
        percentile_columns.append(percentile_columns_i)

        summary_df['display_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: display_columns_i[int(x)])
        if feature_types[i] == 'numeric':
            summary_df['value_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: bound_lows_i[int(x)])
            summary_df['value_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: bound_ups_i[int(x)])
            info_cols += ['value_lower_%d' % (i + 1), 'value_upper_%d' % (i + 1)]

        if len(percentile_columns_i) != 0:
            summary_df['percentile_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_columns_i[int(x)])
            summary_df['percentile_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_bound_lows_i[int(x)])
            summary_df['percentile_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_bound_ups_i[int(x)])
            info_cols += ['percentile_column_%d' % (i + 1), 'percentile_lower_%d' % (i + 1),
                          'percentile_upper_%d' % (i + 1)]

    return summary_df, info_cols, display_columns, percentile_columns

