import numpy as np
import pandas as pd
import psutil

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FeatureInfo:
    def __init__(
        self,
        feature,
        feature_name,
        df,
        cust_grid_points=None,
        grid_type="percentile",
        num_grid_points=10,
        percentile_range=None,
        grid_range=None,
        show_outliers=False,
        endpoint=True,
    ):
        self.col_name = feature
        self.name = feature_name
        self.type = _check_col(self.col_name, df, is_target=False)
        self.cust_grid_points = cust_grid_points
        self.grid_type = grid_type
        self.num_grid_points = num_grid_points
        self.percentile_range = percentile_range
        self.grid_range = grid_range
        self.show_outliers = show_outliers
        self.endpoint = endpoint

        self._check_grid_type()
        self._check_range()

        if all(
            v is None
            for v in [self.percentile_range, self.grid_range, self.cust_grid_points]
        ):
            self.show_outliers = False

    def _check_grid_type(self):
        assert self.grid_type in {
            "percentile",
            "equal",
        }, "grid_type should be either 'percentile' or 'equal'"

    def _check_range(self):
        for name, range_value in zip(
            ["grid_range", "percentile_range"], [self.grid_range, self.percentile_range]
        ):
            if range_value is not None:
                assert isinstance(range_value, tuple), f"{name} should be a tuple"
                assert len(range_value) == 2, f"{name} should contain 2 elements"
                assert range_value[0] < range_value[1], f"{name} should be in order"
                if name == "percentile_range":
                    assert all(
                        0 <= v <= 100 for v in range_value
                    ), f"{name} should be between 0 and 100"

    def prepare(self, df):
        self._get_grids(df)
        df = self._map_values_to_buckets(df).reset_index(drop=True)
        df["count"] = 1
        count_df = (
            df.groupby("x", as_index=False).agg({"count": "count"}).sort_values("x")
        )

        summary_df = pd.DataFrame(
            np.arange(df["x"].min(), df["x"].max() + 1), columns=["x"]
        )
        summary_df = summary_df.merge(count_df, on="x", how="left").fillna(0)
        summary_df["x"] = summary_df["x"].astype(int)
        summary_df["value"] = summary_df["x"].apply(lambda x: self.display_columns[x])

        info_cols = ["x", "value"]
        if len(self.percentile_columns):
            summary_df["percentile"] = summary_df["x"].apply(
                lambda x: self.percentile_columns[x]
            )
            info_cols.append("percentile")

        summary_df = summary_df[info_cols + ["count"]]

        return df, count_df, summary_df

    def _get_grids(self, df):
        self.percentiles = None
        if self.type == "binary":
            self.grids = np.array([0, 1])
            self.num_bins = 2
        elif self.type == "onehot":
            self.grids = np.array(self.col_name)
            self.num_bins = len(self.grids)
        else:
            if self.cust_grid_points is None:
                self.grids, self.percentiles = self._get_numeric_grids(
                    df[self.col_name].values,
                )
            else:
                self.grids = np.array(sorted(np.unique(self.cust_grid_points)))
            self.num_bins = len(self.grids) - 1

    def _get_numeric_grids(self, values):
        if self.grid_type == "percentile":
            start, end = self.percentile_range or (0, 100)
            percentiles = np.linspace(start=start, stop=end, num=self.num_grid_points)
            grids_df = (
                pd.DataFrame(
                    {
                        "percentiles": [round(x, 2) for x in percentiles],
                        "grids": np.percentile(values, percentiles),
                    }
                )
                .groupby(["grids"], as_index=False)
                .agg({"percentiles": lambda x: list(x)})
                .sort_values("grids", ascending=True)
            )
            return grids_df["grids"].values, grids_df["percentiles"].values
        else:
            min_value, max_value = self.grid_range or (np.min(values), np.max(values))
            return np.linspace(min_value, max_value, self.num_grid_points), None

    def _map_values_to_buckets(self, df):
        col_name, grids = self.col_name, self.grids
        percentile_columns = []

        if self.type == "binary":
            df["x"] = df[col_name]
            display_columns = [f"{col_name}_{v}" for v in grids]
        elif self.type == "onehot":
            df["x"] = np.argmax(df[col_name].values, axis=1)
            df = df[~df["x"].isnull()].reset_index(drop=True)
            display_columns = list(grids)
        else:
            # map feature value into value buckets
            cut_result = pd.cut(
                df[self.col_name].values, bins=self.grids, right=False, precision=2
            )
            df["x"] = cut_result.codes
            display_columns = [str(v) for v in cut_result.categories]
            # 1 ... 2 ... 3, index from 0
            x_max = len(self.grids) - 2

            if self.endpoint:
                df["x"] = df.apply(
                    lambda row: x_max
                    if row[self.col_name] == self.grids[-1]
                    else row["x"],
                    axis=1,
                )
                display_columns[-1] = display_columns[-1].replace(")", "]")

            if self.grid_type == "percentile":
                for i, col in enumerate(display_columns):
                    per_vals = self.percentiles[i] + self.percentiles[i + 1]
                    per_min, per_max = str(min(per_vals)), str(max(per_vals))
                    percentile_columns.append(
                        col[0] + ", ".join([per_min, per_max]) + col[-1]
                    )

            if self.show_outliers:

                def _assign_x(row):
                    if row["x"] != -1:
                        return row["x"]
                    if row[col_name] < grids[0]:
                        return -1
                    # if self.endpoint, grids[-1] is already assigned with x_max
                    if row[col_name] >= grids[-1]:
                        return x_max + 1

                df["x"] = df.apply(lambda row: _assign_x(row), axis=1)
                if df["x"].min() == -1:
                    display_columns = ["<" + str(self.grids[0])] + display_columns
                    percentile_columns = [
                        "<" + str(min(self.percentiles[0]))
                    ] + percentile_columns
                if df["x"].max() == x_max + 1:
                    display_columns += [
                        (">" if self.endpoint else ">=") + str(self.grids[-1])
                    ]
                    percentile_columns += [
                        (">" if self.endpoint else ">=")
                        + str(max(self.percentiles[-1]))
                    ]
            else:
                df = df[df["x"] != -1]

            # offset results
            df["x"] -= df["x"].min()

        df["x"] = df["x"].map(int)
        self.display_columns = display_columns
        self.percentile_columns = percentile_columns

        return df


def _to_rgba(color, opacity=1.0):
    color = [str(int(v * 255)) for v in color[:3]] + [str(opacity)]
    return "rgba({})".format(",".join(color))


def _check_col(col, df, is_target=True):
    """Validate column and return type"""

    def _validate_cols(cols, df_cols, name):
        missing_cols = cols - df_cols
        if missing_cols:
            raise ValueError(f"{name} does not exist: {missing_cols}")
        return True

    df_cols = set(df.columns)
    name = "target" if is_target else "feature"

    if isinstance(col, list):
        if len(col) <= 1:
            raise ValueError(
                f"a list of {name} should contain more than 1 element, "
                f"please provide the full list or just use the single element"
            )

        _validate_cols(set(col), df_cols, name)

        unique_values = set(np.unique(df[col].values))
        if unique_values != {0, 1}:
            raise ValueError(f"one-hot encoded {name} should only contain 0 or 1")

        if is_target:
            col_type = "multi-class"
        else:
            if set(df[col].sum(axis=1).unique()) != {1}:
                raise ValueError(f"{name} should be one-hot encoded")
            col_type = "onehot"
    else:
        _validate_cols({col}, df_cols, name)

        unique_values = set(np.unique(df[col].values))
        if unique_values == {0, 1}:
            col_type = "binary"
        else:
            col_type = "regression" if is_target else "numeric"

    return col_type


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


def _expand_values(name, value, num):
    if isinstance(value, (list, tuple)):
        assert len(value) == num, f"{name}: length should be {num}."
    return [value] * num


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


def _calc_memory_usage(df, total_units, n_jobs, memory_limit):
    """Calculate n_jobs to use"""
    unit_memory = df.memory_usage(deep=True).sum()
    free_memory = psutil.virtual_memory()[1] * memory_limit
    num_units = int(np.floor(free_memory / unit_memory))
    true_n_jobs = np.min([num_units, n_jobs, total_units])
    if true_n_jobs < 1:
        true_n_jobs = 1

    return true_n_jobs


def _calc_n_jobs(df, n_grids, memory_limit, n_jobs):
    _check_memory_limit(memory_limit)
    true_n_jobs = _calc_memory_usage(
        df,
        n_grids,
        n_jobs,
        memory_limit,
    )
    return true_n_jobs


def _get_grid_combos(feature_grids, feature_types):
    """Calculate grid combinations of two grid lists"""

    # create grid combination
    grids1, grids2 = feature_grids
    feat_type1, feat_type2 = feature_types
    if feat_type1 == "onehot":
        grids1 = np.eye(len(grids1)).astype(int).tolist()
    if feat_type2 == "onehot":
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


def _expand_params_for_interact(params):
    for name, value in params.items():
        if isinstance(value, list):
            assert len(value) == 2, f"{name}: length should be 2."
        else:
            params[name] = _expand_values(name, value, 2)
    return params


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
