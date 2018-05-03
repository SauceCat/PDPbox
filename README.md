# PDPbox
[![PyPI version](https://badge.fury.io/py/PDPbox.svg)](https://badge.fury.io/py/PDPbox)  
**Current development version is slightly different from the one on pypi**

python partial dependence plot toolbox  
This repository is inspired by [ICEbox](https://github.com/kapelner/ICEbox). The goal is to visualize the impact of certain features towards model prediction for any supervised learning algorithm. (now support all scikit-learn algorithms)

## The common problem
When using black box machine learning algorithms like random forest and boosting, it is hard to understand the relations between predictors and model outcome. For example, in terms of random forest, all we get is the feature importance. Although we can know which feature is significantly influencing the outcome based on the importance calculation, it really sucks that we don’t know in which direction it is influencing. And in most of the real cases, the effect is non-monotonic. We need some powerful tools to help understanding the complex relations between predictors and model prediction.  
PDPbox aims to wrap up and enrich some useful functions mentioned in [ICEbox](https://github.com/kapelner/ICEbox) in Python.

## Highlight
1. Support one-hot encoding features.
2. For numeric features, create grids with percentile points.
3. Directly handle multiclass classifier.
4. Support two variable interaction plot.
5. Support actual prediction plot.
6. Support target plot, ploting true target rate through selected grid points.

## Documentation
For details about the ideas, please refer to [Introducing PDPbox](https://medium.com/@SauceCat/introducing-pdpbox-2aa820afd312).  
For description about the functions and parameters, please refer to [PDPbox functions and parameters](https://github.com/SauceCat/PDPbox/blob/master/parameter.md).   
For test and demo, please refer to https://github.com/SauceCat/PDPbox/tree/master/test.

## Install PDPbox
- Through pip :100:
  ```bash
  pip install pdpbox
  ```
- Through git
  ```bash
  git clone https://github.com/SauceCat/PDPbox.git
  cd PDPbox
  python setup.py install
  ```

## Examples

### target_plot
Plot average target value across different feature values (grid buckets)  
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/target_plot_inter.png" width="80%"><br>

### target_plot_interact
Plot average target value across different feature value combinations  
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/target_plot_uni.png" width="80%"><br>

### Binary feature
- single variable plot with original points and individual lines    
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/binary_03.png" width="80%"><br>
- single variable plot with clustered individual lines    
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/binary_04.png" width="80%"><br>
- actual predictions plot for a single variable  
  <img src="https://github.com/SauceCat/PDPbox/blob/master/images/actual_preds_01.PNG" width="80%"><br>

### Numeric feature 
- single variable plot with percentile_range=(5, 95)    
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/numeric_05.png" width="80%"><br>
- single variable plot with customized grid points    
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/numeric_06.png" width="80%"><br>
- actual predictions plot for a single variable   
  <img src="https://github.com/SauceCat/PDPbox/blob/master/images/actual_preds_03.PNG" width="80%">

### One-hot encoding feature
- single variable plot with individual lines and original points    
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/onehot_01.png" width="80%"><br>
- single variable plot without centering the lines    
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/onehot_02.png" width="80%"><br>
- actual predictions plot for a single variable   
  <img src="https://github.com/SauceCat/PDPbox/blob/master/images/actual_preds_02.PNG" width="80%"><br>

### Multiclass
- single variable plot with individual lines and original points  
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/multi_02.png" width="80%"><br>

### Interaction between two variables
- the complete plot   
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/inter_01.png"><br>
- multiclass with only contour plots    
  <img src="https://github.com/SauceCat/pdpBox/blob/master/images/multi_03.png"><br>
