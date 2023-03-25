import numpy as np
import pandas as pd
import psutil

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_rgba(color, opacity=1.0):
    color = [str(int(v * 255)) for v in color[:3]] + [str(opacity)]
    return "rgba({})".format(",".join(color))


def _check_percentile_range(percentile_range):
    """Make sure percentile range is valid"""
    if percentile_range is not None:
        name = "percentile_range"
        assert isinstance(percentile_range, tuple), f"{name} should be a tuple"
        assert len(percentile_range) == 2, f"{name} should contain 2 elements"
        assert all(
            0 <= v <= 100 for v in percentile_range
        ), f"{name} should be between 0 and 100"
        assert percentile_range[0] < percentile_range[1], f"{name} should be in order"


def _check_col(col, df, is_target=True):
    """Validate column and return type"""
    cols = set(df.columns.values)
    name = "target" if is_target else "feature"
    if isinstance(col, list):
        assert (
            len(col) > 1
        ), f"a list of {name} should contain more than 1 element, \
        please provide the full list or just use the single element"
        assert set(col) < cols, f"{name} does not exist: {set(col) - cols}"
        assert set(np.unique(df[col].values)) == {
            0,
            1,
        }, f"one-hot encoded {name} should only contain 0 or 1"
        if not is_target:
            assert set(df[col].sum(axis=1).unique()) == {
                1
            }, f"{name} should be one-hot encoded"
        col_type = "multi-class" if is_target else "onehot"
    else:
        assert col in cols, f"{name} does not exist: {col}"
        if set(np.unique(df[col].values)) == {0, 1}:
            col_type = "binary"
        else:
            col_type = "regression" if is_target else "numeric"

    return col_type


def _check_feature(feature, df):
    """Make sure feature exists and infer feature type

    Feature types
    -------------
    1. binary
    2. onehot
    3. numeric
    """

    return _check_col(feature, df, False)


def _check_target(target, df):
    """Check and return target type

    target types
    ------------
    1. binary
    2. multi-class
    3. regression
    """

    return _check_col(target, df)


def _check_dataset(df):
    """Make sure input dataset is pandas DataFrame"""
    assert isinstance(df, pd.DataFrame), "only accept pandas DataFrame"


def _make_list(x):
    """Make list when it is necessary"""
    if isinstance(x, list):
        return x
    return [x]


def _expand_default(x, default):
    """Create a list of default values"""
    if x is None:
        return [default] * 2
    return x


def _check_model(model, n_classes, pred_func):
    """Check model input, return class information and predict function"""

    n_classes_ = None
    if hasattr(model, "n_classes_"):
        n_classes_ = model.n_classes_

    if n_classes_ is None:
        assert (
            n_classes is not None
        ), "n_classes is required when it can't be accessed through model.n_classes_."
    else:
        n_classes = n_classes_

    pred_func_ = None
    from_model = True
    if hasattr(model, "predict_proba"):
        pred_func_ = model.predict_proba
    elif hasattr(model, "predict"):
        pred_func_ = model.predict

    if pred_func is None:
        assert (
            pred_func_ is not None
        ), "pred_func is required when model.predict_proba or model.predict doesn't exist."
        pred_func = pred_func_
        print("obtain pred_func from the provided model.")
    else:
        print("using provided pred_func.")
        from_model = False

    return n_classes, pred_func, from_model


def _check_grid_type(grid_type):
    """Make sure grid type is percentile or equal"""
    assert grid_type in {
        "percentile",
        "equal",
    }, "grid_type should be either 'percentile' or 'equal'"


def _check_classes(classes_list, n_classes):
    """Makre sure classes list is valid

    Notes
    -----
    class index starts from 0

    """
    if len(classes_list) > 0 and n_classes > 2:
        if np.min(classes_list) < 0:
            raise ValueError("class index should be >= 0.")
        if np.max(classes_list) > n_classes - 1:
            raise ValueError("class index should be < n_classes.")


def _check_memory_limit(memory_limit):
    """Make sure memory limit is between 0 and 1"""
    if memory_limit <= 0 or memory_limit >= 1:
        raise ValueError("memory_limit: should be (0, 1)")


def _check_frac_to_plot(frac_to_plot):
    """Make sure frac_to_plot is between 0 and 1 if it is float"""
    if type(frac_to_plot) == float:
        if (frac_to_plot <= 0.0) or (frac_to_plot > 1.0):
            raise ValueError("frac_to_plot: should in range(0, 1) when it is a float")
    elif type(frac_to_plot) == int:
        if frac_to_plot <= 0:
            raise ValueError("frac_to_plot: should be larger than 0.")
    else:
        raise ValueError("frac_to_plot: should be float or integer")


def _plot_title(axes, plot_style):
    """Add plot title."""
    title_params = {
        "fontname": plot_style.font_family,
        "x": 0,
        "va": "top",
        "ha": "left",
    }
    axes.set_facecolor("white")
    title_style, subtitle_style = (
        plot_style.title["title"],
        plot_style.title["subtitle"],
    )
    axes.text(
        y=0.7, s=title_style["text"], fontsize=title_style["font_size"], **title_params
    )
    axes.text(
        y=0.5,
        s=subtitle_style["text"],
        fontsize=subtitle_style["font_size"],
        color=subtitle_style["color"],
        **title_params,
    )
    axes.axis("off")


def _calc_memory_usage(df, total_units, n_jobs, memory_limit):
    """Calculate n_jobs to use"""
    unit_memory = df.memory_usage(deep=True).sum()
    free_memory = psutil.virtual_memory()[1] * memory_limit
    num_units = int(np.floor(free_memory / unit_memory))
    true_n_jobs = np.min([num_units, n_jobs, total_units])
    if true_n_jobs < 1:
        true_n_jobs = 1

    return true_n_jobs


def _calc_n_jobs(df, feature_grids, memory_limit, n_jobs):
    _check_memory_limit(memory_limit)
    true_n_jobs = _calc_memory_usage(
        df,
        len(feature_grids),
        n_jobs,
        memory_limit,
    )
    return true_n_jobs


def _axes_modify(axes, plot_style, top=False, right=False, grid=False):
    """Modify matplotlib Axes

    Parameters
    ----------
    top: bool, default=False
        xticks location=top
    right: bool, default=False
        yticks, location=right
    grid: bool, default=False
        whether it is for grid plot
    """
    axes.set_facecolor("white")
    axes.tick_params(**plot_style.tick["tick_params"])
    for ticks in [axes.get_xticklabels(), axes.get_yticklabels()]:
        for tick in ticks:
            tick.set_fontname(plot_style.font_family)
    axes.set_frame_on(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()

    if top:
        axes.get_xaxis().tick_top()
    if right:
        axes.get_yaxis().tick_right()
    if not grid:
        axes.grid(True, "major", "both", ls="--", lw=0.5, c="k", alpha=0.3)


def _modify_legend_axes(axes, font_family):
    """Modify legend like Axes"""
    axes.set_frame_on(False)

    for tick in axes.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in axes.get_yticklabels():
        tick.set_fontname(font_family)

    axes.set_facecolor("white")
    axes.set_xticks([])
    axes.set_yticks([])


def _get_grids(values, num_grid_points, grid_type, percentile_range, grid_range):
    """Calculate grid points for numeric feature"""
    if grid_type == "percentile":
        # grid points are calculated based on percentile in unique level
        # thus the final number of grid points might be smaller than num_grid_points
        start, end = 0, 100
        if percentile_range is not None:
            start, end = percentile_range

        percentiles = np.linspace(start=start, stop=end, num=num_grid_points)
        grids_df = pd.DataFrame(
            {
                "percentiles": [round(x, 2) for x in percentiles],
                "grids": np.percentile(values, percentiles),
            }
        )

        # sometimes different percentiles correspond to the same value
        grids_df = (
            grids_df.groupby(["grids"], as_index=False)
            .agg({"percentiles": lambda x: tuple(x)})
            .sort_values("grids", ascending=True)
        )
        grids, percentiles = (
            grids_df["grids"].values,
            grids_df["percentiles"].values,
        )
    else:
        if grid_range is not None:
            values = grid_range
        grids = np.linspace(np.min(values), np.max(values), num_grid_points)
        percentiles = None

    return grids, percentiles


def _get_grid_combos(feature_grids, feature_types):
    """Calculate grid combinations of two grid lists"""

    # create grid combination
    grids1, grids2 = feature_grids
    if feature_types[0] == "onehot":
        grids1 = np.eye(len(grids1)).astype(int).tolist()
    if feature_types[1] == "onehot":
        grids2 = np.eye(len(grids2)).astype(int).tolist()

    grid_combos = []
    for g1 in grids1:
        for g2 in grids2:
            grid_combos.append(_make_list(g1) + _make_list(g2))

    return np.array(grid_combos)


def _find_onehot_actual(x):
    """Map one-hot value to one-hot name"""
    try:
        value = list(x).index(1)
    except:
        value = np.nan
    return value


def _find_bucket(x, grids, endpoint):
    """Find bucket that x falls in"""

    # grids:   ...1...2...3...4...
    # buckets:  1 . 2 . 3 . 4 . 5
    # number of buckets: len(grids) + 1
    # index of ranges: 0 ~ len(grids)

    num_buckets = len(grids) + 1
    if x < grids[0]:
        bucket = 1
    elif x >= grids[-1]:
        if endpoint and x == grids[-1]:
            bucket = num_buckets - 1
        else:
            bucket = num_buckets
    else:
        # 2 ~ (num_buckets - 1)
        for i in range(2, num_buckets):
            if grids[i - 2] <= x < grids[i - 1]:
                bucket = i
    # index from 0
    return bucket - 1


def _get_string(x):
    if int(x) == x:
        x = int(x)
    elif round(x, 1) == x:
        x = round(x, 1)
    else:
        x = round(x, 2)
    return str(x)


def _make_bucket_column_names(grids, endpoint, ranges):
    """Create bucket names based on grids"""
    names = []
    lowers = [np.nan]
    uppers = [grids[0]]

    grids_str = []
    for g in grids:
        grids_str.append(_get_string(g))

    # number of inner buckets: len(grids_str) - 1
    for i in range(len(grids_str) - 1):
        lower, upper = i, i + 1
        name = f"[{grids_str[lower]}, {grids_str[upper]})"
        lowers.append(grids[lower])
        uppers.append(grids[upper])

        if (i == len(grids_str) - 2) and endpoint:
            name = name.replace(")", "]")
        names.append(name)

    end_sign = ">" if endpoint else ">="
    names = [f"< {grids_str[0]}"] + names + [f"{end_sign} {grids_str[-1]}"]
    lowers.append(grids[-1])
    uppers.append(np.nan)
    return [np.array(lst)[ranges] for lst in [names, lowers, uppers]]


def _make_bucket_column_names_percentile(percentiles, endpoint, ranges):
    """Create bucket names based on percentile info"""
    total = len(percentiles)
    names, p_numerics = [], []

    # p is a tuple
    for i, p in enumerate(percentiles):
        p_array = np.array(p).astype(np.float64)
        p_numerics.append(np.min(p_array))

    lowers = [0]
    uppers = [p_numerics[0]]
    for i in range(total - 1):
        # for each grid point, percentile information is in tuple format
        # (percentile1, percentile2, ...)
        # some grid points belong to multiple percentiles
        lower, upper = p_numerics[i], p_numerics[i + 1]
        lower_str, upper_str = _get_string(lower), _get_string(upper)
        name = f"[{lower_str}, {upper_str})"
        lowers.append(lower)
        uppers.append(upper)

        if i == total - 2 and endpoint:
            name = name.replace(")", "]")
        names.append(name)

    lower, upper = p_numerics[0], p_numerics[-1]
    lower_str, upper_str = _get_string(lower), _get_string(upper)
    end_sign = ">" if endpoint else ">="
    names = [f"< {lower_str}"] + names + [f"{end_sign} {upper_str}"]
    lowers.append(upper)
    uppers.append(100)
    return [np.array(lst)[ranges] for lst in [names, lowers, uppers]]


def _calc_figsize(num_charts, ncols, title_height, unit_figsize):
    """Calculate figure size"""
    if num_charts > 1:
        nrows = int(np.ceil(num_charts * 1.0 / ncols))
        ncols = np.min([num_charts, ncols])
        width = np.min([unit_figsize[0] * ncols, 15])
        height = np.min([width * 1.0 / ncols, unit_figsize[1]]) * nrows + title_height
    else:
        width, height, nrows, ncols = (
            unit_figsize[0],
            unit_figsize[1] + title_height,
            1,
            1,
        )

    return width, height, nrows, ncols


def _q1(x):
    return x.quantile(0.25)


def _q2(x):
    return x.quantile(0.5)


def _q3(x):
    return x.quantile(0.75)


def _check_plot_params(df, feature, grid_type, percentile_range):
    _check_dataset(df)
    feature_type = _check_feature(feature, df)
    _check_grid_type(grid_type)
    _check_percentile_range(percentile_range)

    return feature_type


def _check_interact_plot_params(
    df,
    features,
    grid_types,
    percentile_ranges,
    grid_ranges,
    num_grid_points,
    cust_grid_points,
):
    _check_dataset(df)
    num_grid_points = _expand_default(num_grid_points, 10)

    grid_types = _expand_default(grid_types, "percentile")
    [_check_grid_type(v) for v in grid_types]

    percentile_ranges = _expand_default(percentile_ranges, None)
    [_check_percentile_range(v) for v in percentile_ranges]

    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)
    feature_types = [_check_feature(f, df) for f in features]

    return {
        "num_grid_points": num_grid_points,
        "grid_types": grid_types,
        "percentile_ranges": percentile_ranges,
        "grid_ranges": grid_ranges,
        "cust_grid_points": cust_grid_points,
        "feature_types": feature_types,
    }


def _get_grids_and_cols(
    feature,
    feature_type,
    data,
    num_grid_points,
    grid_type,
    percentile_range,
    grid_range,
    cust_grid_points,
):
    percentiles = None

    if feature_type == "binary":
        grids = np.array([0, 1])
        cols = [f"{feature}_{g}" for g in grids]
    elif feature_type == "onehot":
        grids = np.array(feature)
        cols = grids[:]
    else:
        if cust_grid_points is None:
            grids, percentiles = _get_grids(
                data[feature].values,
                num_grid_points,
                grid_type,
                percentile_range,
                grid_range,
            )
        else:
            grids = np.array(sorted(np.unique(cust_grid_points)))
        cols = [_get_string(v) for v in grids]

    return grids, cols, percentiles


def _calc_preds_each(model, X, pred_func, from_model, predict_kwds):
    if from_model:
        preds = pred_func(X, **predict_kwds)
    else:
        preds = pred_func(model, X, **predict_kwds)
    return preds


def _calc_preds(model, X, pred_func, from_model, predict_kwds, chunk_size=-1):
    total = len(X)
    if chunk_size > 0 and chunk_size >= total:
        chunk_size = -1

    if chunk_size == -1:
        preds = _calc_preds_each(model, X, pred_func, from_model, predict_kwds)
    else:
        preds = []
        for i in range(0, total, chunk_size):
            preds.append(
                _calc_preds_each(
                    model, X[i : i + chunk_size], pred_func, from_model, predict_kwds
                )
            )
        preds = np.concatenate(preds)

    return preds


def _prepare_plot_params(
    plot_params,
    ncols,
    display_columns,
    percentile_columns,
    figsize,
    dpi,
    template,
    engine,
):
    if plot_params is None:
        plot_params = {}
    if figsize is not None:
        plot_params["figsize"] = figsize
    if template is not None:
        plot_params["template"] = template

    plot_params.update(
        {
            "ncols": ncols,
            "display_columns": display_columns,
            "percentile_columns": percentile_columns,
            "dpi": dpi,
            "engine": engine,
        }
    )
    return plot_params


def _make_subplots(plot_style):
    fig = plt.figure(figsize=plot_style.figsize, dpi=plot_style.dpi)
    title_ratio = 2
    outer_grid = GridSpec(
        nrows=2,
        ncols=1,
        wspace=0.0,
        hspace=0.0,
        height_ratios=[title_ratio, plot_style.figsize[1] - title_ratio],
    )
    title_axes = plt.subplot(outer_grid[0])
    fig.add_subplot(title_axes)
    _plot_title(title_axes, plot_style)

    inner_grid = GridSpecFromSubplotSpec(
        plot_style.nrows,
        plot_style.ncols,
        subplot_spec=outer_grid[1],
        wspace=0.3,
        hspace=0.2,
    )

    return fig, inner_grid, title_axes


def _make_subplots_plotly(plot_args, plot_style):
    fig = make_subplots(**plot_args)
    fig.update_layout(
        width=plot_style.figsize[0],
        height=plot_style.figsize[1],
        template=plot_style.template,
        showlegend=False,
        title=go.layout.Title(
            text=f"{plot_style.title['title']['text']} <br><sup>{plot_style.title['subtitle']['text']}</sup>",
            xref="paper",
            x=0,
        ),
    )

    return fig


def _display_percentile(axes, plot_style):
    percentile_columns = plot_style.percentile_columns
    if len(percentile_columns) > 0 and plot_style.show_percentile:
        per_axes = axes.twiny()
        per_axes.set_xticks(axes.get_xticks())
        per_axes.set_xbound(axes.get_xbound())
        per_axes.set_xticklabels(
            percentile_columns, rotation=plot_style.tick["xticks_rotation"]
        )
        per_axes.set_xlabel("percentile buckets", fontdict=plot_style.label["fontdict"])
        _axes_modify(per_axes, plot_style, top=True)


def _get_ticks_plotly(feat_name, plot_style):
    ticktext = plot_style.display_columns.copy()
    if len(plot_style.percentile_columns) > 0:
        for j, p in enumerate(plot_style.percentile_columns):
            ticktext[j] += f"<br><sup><b>{p}</b></sup>"
        title_text = f"<b>{feat_name}</b> (value+percentile)"
    else:
        title_text = f"<b>{feat_name}</b> (value)"

    return title_text, ticktext
