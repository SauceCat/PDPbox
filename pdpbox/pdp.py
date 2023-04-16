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
)

import pandas as pd
import numpy as np
from pqdm.processes import pqdm
import copy
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings("ignore")


class PDPResult:
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
        predict_kwds={},
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

    def prepare_calculate(self):
        self.n_classes, self.pred_func, self.from_model = _check_model(
            self.model, self.n_classes, self.pred_func
        )
        self.target = np.arange(self.n_classes) if self.n_classes > 2 else [0]

    def _calc_ice_lines(
        self,
        df,
        feats,
        grids,
        grid_idx,
    ):
        for i, feat in enumerate(feats):
            df[feat] = grids[i]

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

    def collect_pdp_results(self, features, grids):
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
        predict_kwds={},
        data_transformer=None,
        cust_grid_points=None,
        grid_type="percentile",
        num_grid_points=10,
        percentile_range=None,
        grid_range=None,
    ):
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
        self.prepare_feature()
        self.prepare_calculate()
        self.calculate()

    def prepare_feature(self):
        _, self.count_df, _ = self.feature_info.prepare(self.df)
        self.n_grids = len(self.feature_info.grids)
        dist_df = self.df[self.feature_info.col_name]
        num_samples = 1000
        if len(dist_df) > num_samples:
            dist_df = dist_df.sample(num_samples, replace=False)
        self.dist_df = dist_df

    def calculate(self):
        features = _make_list(self.feature_info.col_name)
        feature_grids = []
        for i, grid in enumerate(self.feature_info.grids):
            if self.feature_info.type == "onehot":
                grids = [0] * len(features)
                grids[i] = 1
            else:
                grids = [grid]
            feature_grids.append(grids)
        self.collect_pdp_results(features, feature_grids)

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
        if plot_params is None:
            plot_params = {}

        _check_frac_to_plot(frac_to_plot)
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
        predict_kwds={},
        data_transformer=None,
        num_grid_points=10,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
    ):
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
        self.prepare_feature(kwargs)
        self.prepare_calculate()
        self.calculate()

    def prepare_feature(self, kwargs):
        params = _expand_params_for_interact(kwargs)
        self.pdp_isolate_objs = []
        self.n_grids = 1

        for i in range(2):
            obj = PDPIsolate(
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
            obj.df = None
            self.n_grids *= obj.n_grids
            self.pdp_isolate_objs.append(obj)

        self.feature_grid_combos = self._get_grid_combos()

    def _get_grid_combos(self):
        grids = [self.pdp_isolate_objs[i].feature_info.grids for i in range(2)]
        types = [self.pdp_isolate_objs[i].feature_info.type for i in range(2)]
        for i in range(2):
            if types[i] == "onehot":
                grids[i] = np.eye(len(grids[i])).astype(int).tolist()

        grid_combos = []
        for g1 in grids[0]:
            for g2 in grids[1]:
                grid_combos.append(_make_list(g1) + _make_list(g2))

        return np.array(grid_combos)

    def calculate(self):
        features = []
        for i in range(2):
            features += _make_list(self.pdp_isolate_objs[i].feature_info.col_name)
        self.collect_pdp_results(features, self.feature_grid_combos)

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

        if plot_params is None:
            plot_params = {}

        feature_types = [self.pdp_isolate_objs[i].feature_info.type for i in range(2)]

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
                "n_grids": [self.pdp_isolate_objs[i].n_grids for i in range(2)],
                "plot_pdp": plot_pdp,
                "to_bins": to_bins,
                "plot_type": plot_type,
                "show_percentile": show_percentile,
            }
        )

        which_classes = _check_classes(which_classes, self.n_classes)
        plot_engine = PDPInteractPlotEngine(self, which_classes, plot_params)
        return plot_engine.plot()
