# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.2.0] 
### Added
- Formal documentation hosted on readthedocs.org
- Keep trace of historical documentations
- Unit tests
- `info_plots.target_plot_interact` for visualising average target value across interaction between two features
- `info_plots.actual_plot_interact` for visualising prediction distribution across interaction between two features
- `get_dataset` for storing model and dataset for three different problems 
    (binary classification, multi-class classification, regression)
- Tutorials in jupyter notebook format

### Changed
- Move all information related plots under `info_plots`
- `class PDPIsolate`
    - Rename `class pdp_isolate_obj` as `class PDPIsolate`
    - Remove `self.classifier`, `self.model_features`, `self.actual_columns`
    - Add `self.which_class`: store class information for multi-class problem
    - Add `self.percentile_info`: store percentile information for grids
    - Add `self.count_data`: store value count information for data points
    - Add `self.hist_data`: store value information for numeric feature
- `class PDPInteract`
    - Rename `class pdp_interact_obj` as `class PDPInteract`
    - Remove `self.classifier`, `self.model_features`
    - Add `self.which_class`: store class information for multi-class problem
    - Combine `self.pdp_isolate_out1` and `self.pdp_isolate_out2` into `self.pdp_isolate_outs`
- `def pdp_isolate`
    - Replace param `train_X` as `dataset` to store whole dataset 
        instead of only the subset for model training, 
        thus add param `model_features` to indicate features used for model training
	- Add param `grid_type`: select type of grids
	- Add param `grid_range`: select value range of grids
	- Add param `memory_limit`: limit memory usage
	- Add param `n_jobs`: support parallel processing
	- Change param `predict_kwds` default value into `None`
	- Add param `data_transformer`: support dataset transformation

		
### Removed


### Fixed