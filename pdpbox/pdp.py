from .pdp_utils import (
    PDPIsolatePlotEngine,
    PDPInteractPlotEngine,
)
from .utils import (
    _check_frac_to_plot,
    _make_list,
    _check_classes,
    _calc_n_jobs,
    FeatureInfo,
    _check_model,
    _calc_preds,
    _expand_params_for_interact,
    _check_dataset,
    _check_cluster_params,
    _check_plot_engine,
    _check_pdp_interact_plot_type,
)

import pandas as pd
import numpy as np
from pqdm.processes import pqdm
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings("ignore")


class PDPResult:
    """
    Stores the results of the PDP analysis.

    Attributes
    ----------
    class_id : int or None
        The class index for multi-class problems. For binary and resgression
        problems, it is None.
    ice_lines : pd.DataFrame
        A DataFrame that contains the calculated ICE lines. The shape of the
        DataFrame is (n_samples, n_grids).
    pdp : numpy.ndarray
        The calculated PDP values. The shape of the array is (n_grids,).
    """

    def __init__(
        self,
        class_id,
        ice_lines,
        pdp,
    ):
        self.class_id = class_id
        self.ice_lines = ice_lines
        self.pdp = pdp


class _PDPBase:
    def __init__(
        self,
        model,
        model_features,
        pred_func=None,
        n_classes=None,
        memory_limit=0.5,
        chunk_size=-1,
        n_jobs=1,
        predict_kwds=None,
        data_transformer=None,
    ):
        self.model = model
        self.n_classes = n_classes
        self.pred_func = pred_func
        self.model_features = model_features
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.predict_kwds = predict_kwds
        self.data_transformer = data_transformer
        self.dist_num_samples = 1000

    def _prepare_calculate(self):
        self.n_classes, self.pred_func, self.from_model = _check_model(
            self.model, self.n_classes, self.pred_func
        )
        self.target = list(np.arange(self.n_classes)) if self.n_classes > 2 else [0]

    def _calc_ice_lines(
        self,
        df,
        feats,
        grids,
        grid_idx,
    ):
        df[feats] = grids
        if self.data_transformer is not None:
            df = self.data_transformer(df)
        preds = _calc_preds(
            self.model,
            df,
            self.pred_func,
            self.from_model,
            self.predict_kwds,
            self.chunk_size,
        )

        if self.n_classes == 0:
            grid_results = pd.DataFrame(preds, columns=[grid_idx])
        elif self.n_classes == 2:
            grid_results = pd.DataFrame(preds[:, 1], columns=[grid_idx])
        else:
            grid_results = []
            for n_class in range(self.n_classes):
                grid_result = pd.DataFrame(preds[:, n_class], columns=[grid_idx])
                grid_results.append(grid_result)

        return grid_results

    def _collect_pdp_results(self, features, grids):
        self.n_jobs = _calc_n_jobs(
            self.df, self.n_grids, self.memory_limit, self.n_jobs
        )
        grid_results = []
        pbar = tqdm(total=len(grids))
        for i in range(0, len(grids), self.n_jobs):
            calc_args = []
            for j in range(i, i + self.n_jobs):
                if j >= len(grids):
                    break
                calc_args.append(
                    {
                        "grids": grids[j],
                        "grid_idx": j,
                        "df": self.df[self.model_features],
                        "feats": features,
                    }
                )

            batch_results = pqdm(
                calc_args,
                self._calc_ice_lines,
                n_jobs=self.n_jobs,
                argument_type="kwargs",
                disable=True,
            )
            pbar.update(len(batch_results))
            grid_results += batch_results
        pbar.close()

        grid_indices = np.arange(self.n_grids)
        self.results = []
        if self.n_classes > 2:
            for cls_idx in range(self.n_classes):
                ice_lines = pd.concat([res[cls_idx] for res in grid_results], axis=1)
                pdp = ice_lines[grid_indices].mean().values
                self.results.append(PDPResult(cls_idx, ice_lines, pdp))
        else:
            ice_lines = pd.concat(grid_results, axis=1)
            pdp = ice_lines[grid_indices].mean().values
            self.results.append(PDPResult(None, ice_lines, pdp))


class PDPIsolate(_PDPBase):
    """
    Performs Partial Dependence Plot (PDP) analysis on a single feature.

    Attributes
    ----------
    model : object
        A trained model object. The model should have a `predict` or
        `predict_proba` method. Otherwise a custom prediction function should be
        provided through `pred_func`.
    n_classes : int
        Number of classes. If it is None, will infer from `model.n_classes_`.
        Please set it as 0 for regression.
    pred_func : callable
        A custom prediction function. If not provided, `predict` or `predict_proba`
        method of `model` is used to generate the predictions.
    model_features : list of str
        A list of features used in model prediction.
    memory_limit : float
        The maximum proportion of memory that can be used by the calculation
        process.
    chunk_size : int
        The number of samples to predict at each iteration. -1 means all samples at
        once.
    n_jobs : int
        The number of jobs to run in parallel for computation. If set to -1, all
        CPUs are used.
    predict_kwds : dict
        Additional keyword arguments to pass to the `model`'s predict function.
    data_transformer : callable
        A function to transform the input data before prediction.
    dist_num_samples : int
        The number of samples to use for estimating the distribution of the data.
        This is used to handle large datasets by sampling a smaller subset for
        efficiency.
    plot_type : str
        The type of the plot to be generated.
    feature_info : :class:`FeatureInfo`
        An instance of the `FeatureInfo` class.
    count_df : pd.DataFrame
        A DataFrame that contains the count as well as the normalized count
        (percentage) of samples within each feature bucket.
    n_grids : int
        The number of feature grids. For interact plot, it is the product of
        `n_grids` of two features.
    dist_df : pandas.Series
        The distribution of the data points.
    from_model : bool
        A flag indicating if the prediction function was obtained from the model or
        was provided as input.
    target : list of int
        List of target indices. For binary and regression problems, the list will
        be just [0]. For multi-class targets, the list is the class indices.
    results : list of :class:`PDResults`
        The results of the Partial Dependence Plot (PDP) analysis. For binary and
        regression problems, the list will contain a single `PDResults` object. For
        multi-class targets, the list will contain a `PDResults` object for each
        class.

    Methods
    -------
    plot(**kwargs) :
        Generates the PDP plot.
    """

    def __init__(
        self,
        model,
        df,
        model_features,
        feature,
        feature_name,
        pred_func=None,
        n_classes=None,
        memory_limit=0.5,
        chunk_size=-1,
        n_jobs=1,
        predict_kwds=None,
        data_transformer=None,
        cust_grid_points=None,
        grid_type="percentile",
        num_grid_points=10,
        percentile_range=None,
        grid_range=None,
    ):
        """
        Initializes a `PDPIsolate` instance.

        Parameters
        ----------
        model : object
            A trained model object. The model should have a `predict` or
            `predict_proba` method. Otherwise a custom prediction function should be
            provided through `pred_func`.
        df : pd.DataFrame
            A DataFrame that at least contains columns specified by `model_features`.
        model_features : list of str
            A list of features used in model prediction.
        feature : str or list of str
            The column name(s) of the chosen feature. It is a list of str when the
            chosen feature is one-hot encoded.
        feature_name : str
            A custom name for the chosen feature.
        pred_func : callable, optional
            A custom prediction function. If not provided, `predict` or `predict_proba`
            method of `model` is used to generate the predictions. Default is None.
        n_classes : int, optional
            Number of classes. If it is None, will infer from `model.n_classes_`.
            Please set it as 0 for regression. Default is None.
        memory_limit : float, optional
            The maximum proportion of memory that can be used by the calculation
            process. Default is 0.5.
        chunk_size : int, optional
            The number of samples to predict at each iteration. -1 means all samples at
            once. Default is -1.
        n_jobs : int, optional
            The number of jobs to run in parallel for computation. If set to -1, all
            CPUs are used. Default is 1.
        predict_kwds : dict, optional
            Additional keyword arguments to pass to the `model`'s predict function.
            Default is None.
        data_transformer : callable, optional
            A function to transform the input data before prediction. Default is None.
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
        """
        super().__init__(
            model,
            model_features,
            pred_func,
            n_classes,
            memory_limit,
            chunk_size,
            n_jobs,
            predict_kwds,
            data_transformer,
        )
        self.plot_type = "pdp_isolate"
        _check_dataset(df)
        self.feature_info = FeatureInfo(
            feature,
            feature_name,
            df,
            cust_grid_points,
            grid_type,
            num_grid_points,
            percentile_range,
            grid_range,
        )
        self.df = df
        self.model_features = model_features
        self._prepare_feature()
        self._prepare_calculate()
        self._calculate()

    def _prepare_feature(self):
        _, self.count_df, _ = self.feature_info.prepare(self.df)
        self.n_grids = len(self.feature_info.grids)
        dist_df = self.df[self.feature_info.col_name]
        if len(dist_df) > self.dist_num_samples:
            dist_df = dist_df.sample(self.dist_num_samples, replace=False)
        self.dist_df = dist_df

    def _calculate(self):
        features = _make_list(self.feature_info.col_name)
        feature_grids = []
        for i, grid in enumerate(self.feature_info.grids):
            if self.feature_info.type == "onehot":
                grids = [0] * len(features)
                grids[i] = 1
            else:
                grids = [grid]
            feature_grids.append(grids)
        self._collect_pdp_results(features, feature_grids)
        # delete df to save memory
        del self.df

    def plot(
        self,
        center=True,
        plot_lines=False,
        frac_to_plot=1,
        cluster=False,
        n_cluster_centers=None,
        cluster_method="accurate",
        plot_pts_dist=False,
        to_bins=False,
        show_percentile=False,
        which_classes=None,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """
        Generates the Partial Dependence Plot (PDP).

        Parameters
        ----------
        center : bool, optional
            If True, the PDP will be centered by deducting the values of `grids[0]`.
            Default is True.
        plot_lines : bool, optional
            If True, ICE lines will be plotted. Default is False.
        frac_to_plot : int or float, optional
            Fraction of ICE lines to plot. Default is 1.
        cluster : bool, optional
            If True, ICE lines will be clustered. Default is False.
        n_cluster_centers : int or None, optional
            Number of cluster centers. Need to provide when `cluster` is True. Default
            is None.
        cluster_method : {'accurate', 'approx'}, optional
            Method for clustering. If 'accurate', use KMeans. If 'approx', use
            MiniBatchKMeans. Default is accurate.
        plot_pts_dist : bool, optional
            If True, distribution of points will be plotted. Default is False.
        to_bins : bool, optional
            If True, the axis will be converted to bins. Only applicable for numeric
            feature. Default is False.
        show_percentile : bool, optional
            If True, percentiles are shown in the plot. Default is False.
        which_classes : list of int, optional
            List of class indices to plot. If None, all classes will be plotted.
            Default is None.
        figsize : tuple or None, optional
            The figure size for matplotlib or plotly figure. If None, the default
            figure size is used. Default is None.
        dpi : int, optional
            The resolution of the plot, measured in dots per inch. Only applicable when
            `engine` is 'matplotlib'. Default is 300.
        ncols : int, optional
            The number of columns of subplots in the figure. Default is 2.
        plot_params : dict or None, optional
            Custom plot parameters that control the style and aesthetics of the plot.
            Default is None.
        engine : {'matplotlib', 'plotly'}, optional
            The plotting engine to use. Default is plotly.
        template : str, optional
            The template to use for plotly plots. Only applicable when `engine` is
            'plotly'. Reference: https://plotly.com/python/templates/ Default is
            plotly_white.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            A Matplotlib or Plotly figure object depending on the plot engine being
            used.
        dict of matplotlib.axes.Axes or None
            A dictionary of Matplotlib axes objects. The keys are the names of the
            axes. The values are the axes objects. If `engine` is 'ploltly', it is
            None.
        """
        if plot_params is None:
            plot_params = {}

        _check_frac_to_plot(frac_to_plot)
        if cluster:
            _check_cluster_params(n_cluster_centers, cluster_method)
        _check_plot_engine(engine)

        plot_params.update(
            {
                "ncols": ncols,
                "figsize": figsize,
                "dpi": dpi,
                "template": template,
                "engine": engine,
                "n_grids": self.n_grids,
                "plot_lines": plot_lines,
                "frac_to_plot": frac_to_plot,
                "plot_pts_dist": plot_pts_dist,
                "to_bins": to_bins,
                "show_percentile": show_percentile,
                "center": center,
                "clustering": {
                    "on": cluster,
                    "n_centers": n_cluster_centers,
                    "method": cluster_method,
                },
            }
        )

        which_classes = _check_classes(which_classes, self.n_classes)
        plot_engine = PDPIsolatePlotEngine(self, which_classes, plot_params)
        return plot_engine.plot()


class PDPInteract(_PDPBase):
    """
    Performs Partial Dependence Plot (PDP) analysis for interaction between two features.

    Attributes
    ----------
    model : object
        A trained model object. The model should have a `predict` or
        `predict_proba` method. Otherwise a custom prediction function should be
        provided through `pred_func`.
    n_classes : int
        Number of classes. If it is None, will infer from `model.n_classes_`.
        Please set it as 0 for regression.
    pred_func : callable
        A custom prediction function. If not provided, `predict` or `predict_proba`
        method of `model` is used to generate the predictions.
    model_features : list of str
        A list of features used in model prediction.
    memory_limit : float
        The maximum proportion of memory that can be used by the calculation
        process.
    chunk_size : int
        The number of samples to predict at each iteration. -1 means all samples at
        once.
    n_jobs : int
        The number of jobs to run in parallel for computation. If set to -1, all
        CPUs are used.
    predict_kwds : dict
        Additional keyword arguments to pass to the `model`'s predict function.
    data_transformer : callable
        A function to transform the input data before prediction.
    dist_num_samples : int
        The number of samples to use for estimating the distribution of the data.
        This is used to handle large datasets by sampling a smaller subset for
        efficiency.
    plot_type : str
        The type of the plot to be generated.
    features : list
        List of column name(s) for the 2 chosen features. The length of the list
        should be strictly 2.
    feature_names : list
        List of custom names for the 2 chosen features. The length of the list
        should be strictly 2.
    pdp_isolate_objs : list of :class:`PDPIsolate`
        A list of `PDPIsolate` objects, for two features.
    n_grids : int
        The number of feature grids. For interact plot, it is the product of
        `n_grids` of two features.
    feature_grid_combos : numpy.ndarray
        A 2D array that contains the combinations of feature grids. The shape of
        the array is (n_grids, ...).
    from_model : bool
        A flag indicating if the prediction function was obtained from the model or
        was provided as input.
    target : list of int
        List of target indices. For binary and regression problems, the list will
        be just [0]. For multi-class targets, the list is the class indices.
    results : list of :class:`PDResults`
        The results of the Partial Dependence Plot (PDP) analysis. For binary and
        regression problems, the list will contain a single `PDResults` object. For
        multi-class targets, the list will contain a `PDResults` object for each
        class.

    Methods
    -------
    plot(**kwargs) :
        Generates the PDP plot.
    """

    def __init__(
        self,
        model,
        df,
        model_features,
        features,
        feature_names,
        pred_func=None,
        n_classes=None,
        memory_limit=0.5,
        chunk_size=-1,
        n_jobs=1,
        predict_kwds=None,
        data_transformer=None,
        num_grid_points=10,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
    ):
        """
        Initializes a `PDPInteract` instance.

        Parameters
        ----------
        model : object
            A trained model object. The model should have a `predict` or
            `predict_proba` method. Otherwise a custom prediction function should be
            provided through `pred_func`.
        df : pd.DataFrame
            A DataFrame that at least contains columns specified by `model_features`.
        model_features : list of str
            A list of features used in model prediction.
        features : list
            List of column name(s) for the 2 chosen features. The length of the list
            should be strictly 2.
        feature_names : list
            List of custom names for the 2 chosen features. The length of the list
            should be strictly 2.
        pred_func : callable, optional
            A custom prediction function. If not provided, `predict` or `predict_proba`
            method of `model` is used to generate the predictions. Default is None.
        n_classes : int, optional
            Number of classes. If it is None, will infer from `model.n_classes_`.
            Please set it as 0 for regression. Default is None.
        memory_limit : float, optional
            The maximum proportion of memory that can be used by the calculation
            process. Default is 0.5.
        chunk_size : int, optional
            The number of samples to predict at each iteration. -1 means all samples at
            once. Default is -1.
        n_jobs : int, optional
            The number of jobs to run in parallel for computation. If set to -1, all
            CPUs are used. Default is 1.
        predict_kwds : dict, optional
            Additional keyword arguments to pass to the `model`'s predict function.
            Default is None.
        data_transformer : callable, optional
            A function to transform the input data before prediction. Default is None.
        num_grid_points : int or list of int, optional
            The number of grid points to use. Only applicable for numeric feature. For
            interact plot, it can also be a list of two integers, indicating the number
            of grid points for each feature. Default is 10.
        grid_types : str or list of str, optional
            Same as `grid_type`, but could be a list of two strings, indicating the
            grid type for each feature. Default is percentile.
        percentile_ranges : tuple or a list of tuples, optional
            Same as `percentile_range`, but could be a list of two tuples, indicating
            the percentile range for each feature. Default is None.
        grid_ranges : tuple or list of tuples, optional
            Same as `grid_range`, but could be a list of two tuples, indicating the
            grid range for each feature. Default is None.
        cust_grid_points : array-like or list of arrays, optional
            Custom grid points for the feature. For interact plot, it can also be a
            list of two arrays, indicating the grid points for each feature. Default is
            None.
        """
        super().__init__(
            model,
            model_features,
            pred_func,
            n_classes,
            memory_limit,
            chunk_size,
            n_jobs,
            predict_kwds,
            data_transformer,
        )
        self.plot_type = "pdp_interact"
        _check_dataset(df)
        self.df = df
        self.model_features = model_features
        self.features = features
        self.feature_names = feature_names
        kwargs = {
            "num_grid_points": num_grid_points,
            "grid_types": grid_types,
            "percentile_ranges": percentile_ranges,
            "grid_ranges": grid_ranges,
            "cust_grid_points": cust_grid_points,
        }
        self._prepare_feature(kwargs)
        self._prepare_calculate()
        self._calculate()

    def _prepare_feature(self, kwargs):
        params = _expand_params_for_interact(kwargs)
        self.pdp_isolate_objs = [
            PDPIsolate(
                self.model,
                self.df,
                self.model_features,
                self.features[i],
                self.feature_names[i],
                self.pred_func,
                self.n_classes,
                self.memory_limit,
                self.chunk_size,
                self.n_jobs,
                self.predict_kwds,
                self.data_transformer,
                num_grid_points=params["num_grid_points"][i],
                grid_type=params["grid_types"][i],
                percentile_range=params["percentile_ranges"][i],
                grid_range=params["grid_ranges"][i],
                cust_grid_points=params["cust_grid_points"][i],
            )
            for i in range(2)
        ]
        for obj in self.pdp_isolate_objs:
            obj.df = None
        self.n_grids = np.prod([obj.n_grids for obj in self.pdp_isolate_objs])
        self.feature_grid_combos = self._get_grid_combos()

    def _get_grid_combos(self):
        grids = [obj.feature_info.grids for obj in self.pdp_isolate_objs]
        types = [obj.feature_info.type for obj in self.pdp_isolate_objs]
        for i, grid in enumerate(grids):
            if types[i] == "onehot":
                grids[i] = np.eye(len(grid)).astype(int).tolist()

        grid_combos = []
        for g1 in grids[0]:
            for g2 in grids[1]:
                grid_combos.append(_make_list(g1) + _make_list(g2))

        return np.array(grid_combos)

    def _calculate(self):
        features = []
        for obj in self.pdp_isolate_objs:
            features += _make_list(obj.feature_info.col_name)
        self._collect_pdp_results(features, self.feature_grid_combos)
        # delete df to save memory
        del self.df

    def plot(
        self,
        plot_type="contour",
        plot_pdp=False,
        to_bins=True,
        show_percentile=False,
        which_classes=None,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """
        Generates the Partial Dependence Plot (PDP).

        Parameters
        ----------
        plot_type : {'grid', 'contour'}, optional
            The type of interaction plot to be generated. Default is contour.
        plot_pdp : bool, optional
            If it is True, pdp for each feature will be plotted. Default is False.
        to_bins : bool, optional
            If True, the axis will be converted to bins. Only applicable for numeric
            feature. Default is True.
        show_percentile : bool, optional
            If True, percentiles are shown in the plot. Default is False.
        which_classes : list of int, optional
            List of class indices to plot. If None, all classes will be plotted.
            Default is None.
        figsize : tuple or None, optional
            The figure size for matplotlib or plotly figure. If None, the default
            figure size is used. Default is None.
        dpi : int, optional
            The resolution of the plot, measured in dots per inch. Only applicable when
            `engine` is 'matplotlib'. Default is 300.
        ncols : int, optional
            The number of columns of subplots in the figure. Default is 2.
        plot_params : dict or None, optional
            Custom plot parameters that control the style and aesthetics of the plot.
            Default is None.
        engine : {'matplotlib', 'plotly'}, optional
            The plotting engine to use. Default is plotly.
        template : str, optional
            The template to use for plotly plots. Only applicable when `engine` is
            'plotly'. Reference: https://plotly.com/python/templates/ Default is
            plotly_white.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            A Matplotlib or Plotly figure object depending on the plot engine being
            used.
        dict of matplotlib.axes.Axes or None
            A dictionary of Matplotlib axes objects. The keys are the names of the
            axes. The values are the axes objects. If `engine` is 'ploltly', it is
            None.
        """
        if plot_params is None:
            plot_params = {}

        _check_plot_engine(engine)
        _check_pdp_interact_plot_type(plot_type)
        feature_types = [obj.feature_info.type for obj in self.pdp_isolate_objs]

        if (
            not all(v == "numeric" for v in feature_types)
            or plot_pdp
            or plot_type == "grid"
        ):
            to_bins = True

        if not to_bins and any(v == "numeric" for v in feature_types):
            show_percentile = False

        plot_params.update(
            {
                "ncols": ncols,
                "figsize": figsize,
                "dpi": dpi,
                "template": template,
                "engine": engine,
                "n_grids": [obj.n_grids for obj in self.pdp_isolate_objs],
                "plot_pdp": plot_pdp,
                "to_bins": to_bins,
                "plot_type": plot_type,
                "show_percentile": show_percentile,
            }
        )

        which_classes = _check_classes(which_classes, self.n_classes)
        plot_engine = PDPInteractPlotEngine(self, which_classes, plot_params)
        return plot_engine.plot()
