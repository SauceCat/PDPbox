import numpy as np
import pandas as pd
import psutil


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


def _check_model(model):
    """Check model input, return class information and predict function"""
    try:
        n_classes = len(model.classes_)
        predict = model.predict_proba
    except:
        n_classes = 0
        predict = model.predict

    return n_classes, predict


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


def _modify_legend_ax(ax, font_family):
    """Modify legend like Axes"""
    ax.set_frame_on(False)

    for tick in ax.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_family)

    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])


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


def _sample_data(ice_lines, frac_to_plot):
    """Get sample ice lines to plot

    Notes
    -----
    If frac_to_plot==1, will plot all lines instead of sampling one line

    """

    if frac_to_plot < 1.0:
        ice_plot_data = ice_lines.sample(int(ice_lines.shape[0] * frac_to_plot))
    elif frac_to_plot > 1:
        ice_plot_data = ice_lines.sample(frac_to_plot)
    else:
        ice_plot_data = ice_lines.copy()

    ice_plot_data = ice_plot_data.reset_index(drop=True)
    return ice_plot_data


def _find_onehot_actual(x):
    """Map one-hot value to one-hot name"""
    try:
        value = list(x).index(1)
    except:
        value = np.nan
    return value


def _find_bucket(x, grids, endpoint):
    """Find bucket that x falls in"""

    # grids: ...1...2...3...4...
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
