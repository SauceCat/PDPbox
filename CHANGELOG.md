# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.2.0] 
### Added
- Formal documentation hosted on readthedocs.org
- Keep trace of historical documentations
- Unit tests
- `info_plots.target_plot_interact`: visualise average target value across interaction between two features
- `info_plots.actual_plot_interact`: visualise prediction distribution across interaction between two features
- `get_dataset`: store models and datasets for three different problems 
    (binary classification, multi-class classification, regression)
- Tutorials in jupyter notebook format

### Changed
- Move all information related plots under `info_plots`, including
    - `info_plots.target_plot`
    - `info_plots.target_plot_interact`
    - `info_plots.actual_plot`
    - `info_plots.actual_plot_interact`
- Move all utility functions under `xx_utils.py`
    - `utils.py`: general utility functions
    - `info_plot_utils.py`: utility functions for information plots
    - `pdp_calc_utils.py`: utility functions for pdp related calculation
    - `pdp_plot_utils.py`: utility functions for pdp related plots
- `class PDPIsolate`
    - Rename `class pdp_isolate_obj` as `class PDPIsolate`
    - Remove `self.classifier`, `self.model_features`, `self.actual_columns`: useless
    - Add `self.which_class`, `self.percentile_info`, `self.count_data`, `self.hist_data`:
      store class information for multi-class problem, 
      store percentile information for grid points, 
      store value count information as well as feature values for numeric feature
- `class PDPInteract`
    - Rename `class pdp_interact_obj` as `class PDPInteract`
    - Remove `self.classifier`, `self.model_features`: useless
    - Add `self.which_class`: store class information for multi-class problem
    - Combine `self.pdp_isolate_out1` and `self.pdp_isolate_out2` into `self.pdp_isolate_outs`
- `pdp.pdp_isolate`
    - Replace `train_X` as `dataset` to store whole dataset 
        instead of only the subset for model training, 
        thus add `model_features` to indicate features used for model training
	- Add `grid_type`, `grid_range`: define type and range for grid points
	- Add `memory_limit`, `n_jobs`: limit memory usage, support parallel processing
	- Set `predict_kwds` default value into `None` instead of `{}`
	- Add `data_transformer`: support dataset transformation
- `pdp.pdp_plot`
    - Add `plot_pts_dist`: enable to plot distribution of data points
    - Remove `plot_org_pts`: no longer support plotting original data points
    - Set `cluster_method` default value as 'accurate' instead of None
    - Add `show_percentile`: show percentile information of grid points
    - Set `ncols` default value as 2 instead of None
    - Add `which_classes`, remove `multi_flag`, `which_class`: 
        plot for a single class is now supported by `which_classes`
- `pdp.pdp_interact`
    - Replace `train_X` as `dataset` to store whole dataset 
        instead of only the subset for model training, 
        thus add `model_features` to indicate features used for model training
    - Set `num_grid_points` default value as None instead of `[10, 10]`
	- Add `grid_type`, `grid_range`: define type and range for grid points
    - Set `percentile_ranges` default value as None instead of `[None, None]`
    - Set `cust_grid_points` default value as None instead of `[None, None]`
	- Set `predict_kwds` default value into `None` instead of `{}`
- `pdp.pdp_interact_plot`
    - Add `plot_type`, `plot_pdp`, remove `only_inter`: define plot type and whether to plot pdp
        for both features, only showing contour plot now is supported by `plot_type` and `plot_pdp`
    - Add `which_classes`, remove `multi_flag`, `which_class`: 
        plot for a single class is now supported by `which_classes`
    - Set `ncols` default value as 2 instead of None
    - Remove `center`, `plot_org_pts`, `plot_lines`, `frac_to_plot`, `cluster`, `n_cluster_centers`, 
    `cluster_method`: no longer support plotting separate pdp plots
- `info_plots.target_plot`
	- Add `grid_type`, `grid_range`: define type and range for grid points
    - Add `show_percentile`: show percentile information of grid points
    - Add `show_outliers`: whether to show data points outside the grid range
    - Add `endpoint`: whether stop is the last grid point
    - Add `ncols`: define number of columns for multiple plots
- `info_plots.actual_plot`
    - Add `model`, `X`, `feature`, remove `pdp_isolate_out`: no longer depend on `pdp.pdp_isolate`, 
    thus need to define all necessary parameters for calculating the results
    - Add `num_grid_points`, `grid_type`, `percentile_range`, `grid_range`, `cust_grid_points`, 
    `show_percentile`, `show_outliers`, `endpoint`, `which_classes`, `predict_kwds`
    - Set `ncols` default value as 2 instead of None
    - Add `which_classes`, remove `multi_flag`, `which_class`: 
        plot for a single class is now supported by `which_classes`
	- Set `predict_kwds` default value into `None` instead of `{}`

### Fixed
- Python3 compatibility
- All plotting related functions would return a `matplotlib.figure.Figure` object 
    as well as `Matplotlib.axes` for further modification