import numpy as np
import pandas as pd
import psutil


class FeatureInfo:
    """
    A class to store information about a feature, preprocess the data, and
    prepare summary statistics.

    Attributes
    ----------
    col_name : str or list of str
        The column name(s) of the feature in the input DataFrame.
    name : str
        The custom name of the feature.
    type : str
        The type of the feature, as determined by the _check_col function.
    cust_grid_points : array-like or list of arrays
        Custom grid points for the feature. For interact plot, it can also be a
        list of two arrays, indicating the grid points for each feature.
    grid_type : {'percentile', 'equal'}
        The grid type. Only applicable for numeric feature.
    num_grid_points : int or list of int
        The number of grid points to use. Only applicable for numeric feature. For
        interact plot, it can also be a list of two integers, indicating the number
        of grid points for each feature.
    percentile_range : tuple
        A tuple of two values indicating the range of percentiles to use. Only
        applicable for numeric feature and when `grid_type` is 'percentile'. If it
        is None, will use all samples.
    grid_range : tuple
        A tuple of two values indicating the range of grid values to use. Only
        applicable for numeric feature. If it is None, will use all samples.
    show_outliers : bool or list of bool
        Whether to show outliers in the plot. Only applicable for numeric feature.
        For interact plot, it can also be a list of two booleans, indicating
        whether to show outliers for each feature.
    endpoint : bool
        Whether to include the endpoint of the range.

    Methods
    -------
    prepare(**kwargs)
        Prepares the input DataFrame and calculates summary statistics.
    """

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
        """
        Initializes the `FeatureInfo` class.

        Parameters
        ----------
        feature : str or list of str
            The column name(s) of the chosen feature. It is a list of str when the
            chosen feature is one-hot encoded.
        feature_name : str
            A custom name for the chosen feature.
        df : pd.DataFrame
            A DataFrame that at least contains the feature(s).
        cust_grid_points : array-like or list of arrays, optional
            Custom grid points for the feature. For interact plot, it can also be a
            list of two arrays, indicating the grid points for each feature. Default is
            None.
        grid_type : {'percentile', 'equal'}, optional
            The grid type. Only applicable for numeric feature. Default is percentile.
        num_grid_points : int or list of int, optional
            The number of grid points to use. Only applicable for numeric feature. For
            interact plot, it can also be a list of two integers, indicating the number
            of grid points for each feature. Default is 10.
        percentile_range : tuple, optional
            A tuple of two values indicating the range of percentiles to use. Only
            applicable for numeric feature and when `grid_type` is 'percentile'. If it
            is None, will use all samples. Default is None.
        grid_range : tuple, optional
            A tuple of two values indicating the range of grid values to use. Only
            applicable for numeric feature. If it is None, will use all samples.
            Default is None.
        show_outliers : bool or list of bool, optional
            Whether to show outliers in the plot. Only applicable for numeric feature.
            For interact plot, it can also be a list of two booleans, indicating
            whether to show outliers for each feature. Default is False.
        endpoint : bool, optional
            Whether to include the endpoint of the range. Default is True.
        """
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
        """
        Validates the `grid_type` attribute and sets it to 'custom' if `cust_grid_points` are provided.
        """
        assert self.grid_type in {
            "percentile",
            "equal",
        }, "grid_type should be either 'percentile' or 'equal'"
        if self.cust_grid_points is not None:
            self.grid_type = "custom"

    def _check_range(self):
        """
        Validates the `grid_range` and `percentile_range` attributes, ensuring
        they are tuples with two elements in ascending order.
        If `percentile_range` is used, it also checks that both values are between 0 and 100.
        """
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

    def _get_grids(self, df):
        """
        Determines the appropriate grid points for the feature based on its
        type and other class attributes based on the input DataFrame.
        """
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
        """
        Calculates the grid points for numeric features based on the `grid_type`
        attribute and other class attributes. Return the grid points and percentiles
        corresponding to each grid point (or None if not applicable).
        """
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
                .agg({"percentiles": list})
                .sort_values("grids", ascending=True)
            )
            return grids_df["grids"].values, grids_df["percentiles"].values
        else:
            min_value, max_value = self.grid_range or (np.min(values), np.max(values))
            return np.linspace(min_value, max_value, self.num_grid_points), None

    def _map_values_to_buckets(self, df):
        """
        Maps the values of the feature column to the appropriate bucket based
        on the grids attribute. Returns the input DataFrame with an additional
        column 'x' representing the bucket index for each value.
        """
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

                df["x"] = df.apply(_assign_x, axis=1)
                if df["x"].min() == -1:
                    display_columns = ["<" + str(self.grids[0])] + display_columns

                    if self.grid_type == "percentile":
                        percentile_columns = [
                            "<" + str(min(self.percentiles[0]))
                        ] + percentile_columns

                if df["x"].max() == x_max + 1:
                    display_columns += [
                        (">" if self.endpoint else ">=") + str(self.grids[-1])
                    ]
                    if self.grid_type == "percentile":
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

    def prepare(self, df):
        """
        Process the input DataFrame, calculates summary statistics, returns
        3 DataFrames: the processed input DataFrame, a DataFrame containing the
        count of each bucket, and a DataFrame containing the summary statistics
        for each bucket.
        """
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


def _to_rgba(color, opacity=1.0):
    """
    Convert a color from tuple representation to an RGBA string representation.

    Parameters
    ----------
    color : tuple
        A tuple of 3 values representing RGB color in the range [0, 1].
    opacity : float, optional
        Opacity of the color in the range [0, 1]. Defaults to 1.0.

    Returns
    -------
    str
        The RGBA string representation of the input color.
    """
    return f"rgba({','.join(str(int(v * 255)) for v in color[:3])},{opacity})"


def _check_col(col, df, is_target=True):
    """
    Validate column and return its type.

    Parameters
    ----------
    col : str or list of str
        The column name(s) in the input DataFrame.
    df : pandas.DataFrame
        The input DataFrame containing the columns to validate.
    is_target : bool, default=True
        If True, the column is treated as a target column. Otherwise, it's treated as a feature column.

    Returns
    -------
    str
        The determined type of the column.
        For target columns, possible values are: "binary", "multi-class", "regression".
        For feature columns, possible values are: "binary", "onehot", "numeric".

    Raises
    ------
    ValueError
        If the provided column(s) is not found in the input DataFrame or does not meet the expected format.
    """

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
    return _check_col(target, df)


def _check_dataset(df, features=None):
    """
    Validate input dataset and ensure it is a pandas DataFrame with the specified features.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset to be validated.
    features : list, optional
        A list of feature column names that should be present in the input dataset, by default None.

    Raises
    ------
    AssertionError
        If the input dataset is not a pandas DataFrame.
    ValueError
        If the input dataset does not contain all the specified features.
    """
    if not isinstance(df, pd.DataFrame):
        raise AssertionError("only accept pandas DataFrame")

    if features is not None:
        features = set(features)
        cols = set(df.columns.values)
        missing_features = features - cols
        if missing_features:
            raise ValueError(
                f"df doesn't contain all model features, missing: {missing_features}"
            )


def _make_list(x):
    """
    Make list when it is necessary.

    Parameters
    ----------
    x : object
        The input object to be converted into a list.

    Returns
    -------
    list
        If the input is already a list, it is returned unchanged.
        If the input is not a list, a new list containing the input object is returned.
    """
    if isinstance(x, list):
        return x
    return [x]


def _check_model(model, n_classes, pred_func):
    """
    Check input model, return class information and predict function.

    Parameters
    ----------
    model : object
        The input model object.
        It should have either `predict_proba` or `predict` method if pred_func is not provided.
        It should have `n_classes_` attribute if n_classes is not provided or it's a regression model.
    n_classes : int, optional
        The number of classes in the target variable. If not provided, it will try to obtain the value from the model.
    pred_func : callable, optional
        The prediction function to be used. If not provided, it will try to obtain the function from the model.

    Returns
    -------
    int
        The number of classes in the target variable.
    callable
        The prediction function to be used.
    bool
        Flag indicating if the prediction function was obtained from the model or was provided as input.

    Raises
    ------
    AssertionError
        If `n_classes` is not provided and cannot be accessed through `model.n_classes_`,
        or if `pred_func` is not provided and neither `model.predict_proba` nor `model.predict` exist.
    """

    n_classes_ = None
    if hasattr(model, "n_classes_"):
        n_classes_ = model.n_classes_

    if n_classes_ is None:
        assert (
            n_classes is not None
        ), "n_classes is required when it can't be accessed through model.n_classes_."
    else:
        n_classes = n_classes_

    if pred_func is None:
        pred_func_ = None
        from_model = True
        if hasattr(model, "predict_proba"):
            pred_func_ = model.predict_proba
        elif hasattr(model, "predict"):
            pred_func_ = model.predict
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
    """
    Make sure classes list is valid.

    Parameters
    ----------
    which_classes : list, optional
        List of class indices to check. Defaults to None.
    n_classes : int
        Total number of classes.

    Returns
    -------
    list
        List of valid class indices.

    Raises
    ------
    ValueError
        If class index is less than 0 or greater than or equal to the number of classes.
    """
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
    """
    Make sure memory limit is between 0 and 1.

    Parameters
    ----------
    memory_limit : float
        Memory limit value to be checked.

    Raises
    ------
    ValueError
        If `memory_limit` is not between 0 and 1 (exclusive).
    """
    if memory_limit <= 0 or memory_limit >= 1:
        raise ValueError("memory_limit: should be (0, 1)")


def _check_frac_to_plot(frac_to_plot):
    """
    Make sure `frac_to_plot` is between 0 and 1 if it is float, and greater than 0 if it is int.

    Parameters
    ----------
    frac_to_plot : float or int
        Fraction or number of instances to be plotted.

    Raises
    ------
    ValueError
        If `frac_to_plot` is not within the valid range for the given type (float or int).
    """

    if type(frac_to_plot) == float:
        if (frac_to_plot <= 0.0) or (frac_to_plot > 1.0):
            raise ValueError("frac_to_plot: should in range(0, 1) when it is a float")
    elif type(frac_to_plot) == int:
        if frac_to_plot <= 0:
            raise ValueError("frac_to_plot: should be larger than 0.")
    else:
        raise ValueError("frac_to_plot: should be float or integer")


def _calc_n_jobs(df, n_grids, memory_limit, n_jobs):
    """
    Calculate the number of jobs to be executed in parallel based on memory constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_grids : int
        Number of grid points.
    memory_limit : float
        Fraction of available memory to use.
    n_jobs : int
        Number of jobs specified.

    Returns
    -------
    int
        The calculated number of jobs to be executed in parallel.
    """
    _check_memory_limit(memory_limit)
    # df memory in bytes
    unit_memory = df.memory_usage(deep=True).sum()
    # free memory in bytes
    free_memory = psutil.virtual_memory().available * memory_limit
    num_units = int(free_memory // unit_memory)
    true_n_jobs = max(1, min(num_units, n_jobs, n_grids))
    return true_n_jobs


def _get_string(x):
    """
    Convert a numeric value to a string with proper formatting.

    Parameters
    ----------
    x : float or int
        Numeric value to be converted to a string.

    Returns
    -------
    str
        Formatted string representation of the input value.
    """
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
    """
    Expand parameter values for interactive usage.

    Parameters
    ----------
    params : dict
        Dictionary containing parameter names and their corresponding values.

    Returns
    -------
    dict
        Dictionary with expanded parameter values, ensuring each value is a list of length 2.
    """
    for name, value in params.items():
        if not isinstance(value, list):
            params[name] = _expand_values(name, value, 2)
        else:
            assert len(value) == 2, f"{name}: length should be 2."

    return params


def _calc_preds_each(model, X, pred_func, from_model, predict_kwds):
    """Calculate model predictions for a single chunk of data."""
    if from_model:
        preds = pred_func(X, **predict_kwds)
    else:
        # if pred_func is provided by user
        # it should be a function that takes three arguments
        preds = pred_func(model, X, **predict_kwds)
    return preds


def _calc_preds(model, X, pred_func, from_model, predict_kwds=None, chunk_size=-1):
    """
    Calculate model predictions with an optional chunk size.

    Parameters
    ----------
    model : object
        Trained model object.
    X : array-like
        Input data to make predictions on.
    pred_func : callable
        Function to make predictions using the model.
    from_model : bool
        Whether the pred_func is obtained from the model or provided by the user.
    predict_kwds : dict
        Additional keyword arguments to pass to pred_func.
    chunk_size : int, optional, default=-1
        Number of samples to predict at a time. If -1, predict all samples at once.

    Returns
    -------
    np.array
        Predictions made by the model.
    """
    total = len(X)
    if predict_kwds is None:
        predict_kwds = {}

    if 0 < chunk_size < total:
        preds = np.concatenate(
            [
                _calc_preds_each(
                    model, X[i : i + chunk_size], pred_func, from_model, predict_kwds
                )
                for i in range(0, total, chunk_size)
            ]
        )
    else:
        preds = _calc_preds_each(model, X, pred_func, from_model, predict_kwds)

    return preds


def _check_cluster_params(n_cluster_centers, cluster_method):
    """
    Check the parameters for clustering.

    Parameters
    ----------
    n_cluster_centers : int or None
        The number of cluster centers to be used. If None, a `ValueError` is raised.
    cluster_method : str
        The method used for clustering. Should be either "approx" or "accurate".
        If it's not, a `ValueError` is raised.

    Raises
    ------
    ValueError
        If `n_cluster_centers` is None or `cluster_method` is not "approx" or "accurate".
    """
    if n_cluster_centers is None:
        raise ValueError("n_cluster_centers should be specified.")
    if not isinstance(n_cluster_centers, int):
        raise TypeError("n_cluster_centers should be int.")
    if n_cluster_centers <= 0:
        raise ValueError("n_cluster_centers should be larger than 0.")
    if cluster_method not in ["approx", "accurate"]:
        raise ValueError('Clustering method should be "approx" or "accurate".')


def _check_plot_engine(engine):
    """
    Check the validity of the plot engine.

    Parameters
    ----------
    engine : str
        The value of the plot engine to check.

    Raises
    ------
    ValueError
        If `engine` is not 'plotly' or 'matplotlib'.
    """
    if engine not in ["plotly", "matplotlib"]:
        raise ValueError("plot_engine should be either 'plotly' or 'matplotlib'.")


def _check_pdp_interact_plot_type(plot_type):
    """
    Check the validity of the `plot_type` for `PDPInteract`.

    Parameters
    ----------
    plot_type : str
        The value of the plot type to check.

    Raises
    ------
    ValueError
        If `plot_type` is not 'grid' or 'contour'.
    """
    if plot_type not in ["grid", "contour"]:
        raise ValueError("plot_type should be either 'grid' or 'contour'.")
