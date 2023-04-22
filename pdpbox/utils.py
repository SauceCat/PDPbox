import numpy as np
import pandas as pd
import psutil


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
        for name, range_value in [
            ("grid_range", self.grid_range),
            ("percentile_range", self.percentile_range),
        ]:
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
        count_df["count_norm"] = count_df["count"] / count_df["count"].sum()

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

            if self.cust_grid_points is None and self.grid_type == "percentile":
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


def _check_dataset(df, features=None):
    """Make sure input dataset is pandas DataFrame"""
    assert isinstance(df, pd.DataFrame), "only accept pandas DataFrame"
    if features is not None:
        features = set(features)
        cols = set(df.columns.values)
        if len(features - cols):
            raise ValueError(
                f"df doesn't contain all model features, missing: {features - cols}"
            )


def _make_list(x):
    """Make list when it is necessary"""
    if isinstance(x, list):
        return x
    return [x]


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


def _check_classes(which_classes, n_classes):
    """Makre sure classes list is valid"""
    if which_classes is None or len(which_classes) == 0:
        which_classes = list(np.arange(n_classes))
    else:
        if n_classes > 0:
            if np.min(which_classes) < 0:
                raise ValueError("class index should be >= 0.")
            if np.max(which_classes) > n_classes - 1:
                raise ValueError("class index should be < n_classes.")
    if n_classes <= 2:
        which_classes = [0]
    return which_classes


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


def _get_string(x):
    if int(x) == x:
        x = int(x)
    elif round(x, 1) == x:
        x = round(x, 1)
    else:
        x = round(x, 2)
    return str(x)


def _q1(x):
    return x.quantile(0.25)


def _q2(x):
    return x.quantile(0.5)


def _q3(x):
    return x.quantile(0.75)


def _expand_values(name, value, num):
    if isinstance(value, (list, tuple)):
        assert len(value) == num, f"{name}: length should be {num}."
    return [value] * num


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
