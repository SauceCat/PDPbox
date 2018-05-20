from pdpbox import info_plots, get_dataset

test_titanic = get_dataset.titanic()

titanic_data = test_titanic['data']
titanic_features = test_titanic['features']
titanic_target = test_titanic['target']
titanic_model = test_titanic['xgb_model']

fig, axes, summary_df = info_plots.target_plot(df=titanic_data, feature='Sex',
											   feature_name='Sex', target=titanic_target)