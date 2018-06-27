
# PDPbox
[![PyPI version](https://badge.fury.io/py/PDPbox.svg)](https://badge.fury.io/py/PDPbox)

python partial dependence plot toolbox

## Motivation

This repository is inspired by ICEbox. The goal is to visualize the impact of certain features towards model
prediction for any supervised learning algorithm. (now support all scikit-learn algorithms)


## The common headache

When using black box machine learning algorithms like random forest and boosting, it is hard to understand the
relations between predictors and model outcome.

For example, in terms of random forest, all we get is the feature importance.
Although we can know which feature is significantly influencing the outcome based on the importance
calculation, it really sucks that we don’t know in which direction it is influencing. And in most of the real cases,
the effect is non-monotonic.

We need some powerful tools to help understanding the complex relations
between predictors and model prediction.


## Highlight

1. Helper functions for visualizing target distribution as well as prediction distribution.
2. Proper way to handle one-hot encoding features.
3. Solution for handling complex mutual dependency among features.
4. Support multi-class classifier.
5. Support two variable interaction partial dependence plot.


## Documentation

- Latest version: http://pdpbox.readthedocs.io/en/latest/
- Historical versions:
  - [v0.1](https://github.com/SauceCat/PDPbox/blob/master/docs_history/v0.1/docs.md)


## Installation

- through pip (stable version)
  ```
  $ pip install pdpbox
  ```

- through git (latest develop version)
  ```
  $ git clone https://github.com/SauceCat/PDPbox.git
  $ cd PDPbox
  $ python setup.py install
  ```

## TODO
- [ ] complete unit test
- [ ] test compatibility with python3
- [ ] change logs
- [ ] release new version

## Gallery
- **PDP:** PDP for a single feature
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/pdp_plot.png' width=90%>

- **PDP:** PDP for a multi-class
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/pdp_plot_multiclass.png' width=90%>

- **PDP Interact:** PDP Interact for two features with contour plot
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/pdp_interact_contour.png' width=60%>

- **PDP Interact:** PDP Interact for two features with grid plot
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/pdp_interact_grid.png' width=60%>

- **PDP Interact:** PDP Interact for multi-class
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/pdp_interact_multiclass.png' width=90%>

- **Information plot:** target plot for a single feature
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/target_plot.png' width=90%>

- **Information plot:** target interact plot for two features
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/target_plot_interact.png' width=90%>

- **Information plot:** actual prediction plot for a single feature
    <img src='https://github.com/SauceCat/PDPbox/blob/master/images/actual_plot.png' width=90%>
