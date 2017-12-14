# PDPbox
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
5. Support actual prediction plot. (new)
6. Support target plot, ploting true target rate through selected grid points. Thanks to @Gavin2318 (new).

## Documentation
For details about the ideas, please refer to [Introducing PDPbox](https://medium.com/@SauceCat/introducing-pdpbox-2aa820afd312).  
For description about the functions and parameters, please refer to [PDPbox functions and parameters](https://github.com/SauceCat/PDPbox/blob/master/parameter.md).   
For test and demo, please refer to https://github.com/SauceCat/PDPbox/tree/master/test.

## Install PDPbox
```bash
git clone https://github.com/SauceCat/PDPbox.git
cd PDPbox
python setup.py install
```

## Examples
#### **Binary feature:** single variable plot with original points and individual lines
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/binary_03.png" width="80%">

#### **Binary feature:** single variable plot with clustered individual lines
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/binary_04.png" width="80%">

#### **Binary feature:** actual predictions plot for a single variable
<img src="https://github.com/SauceCat/PDPbox/blob/master/images/actual_preds_01.PNG" width="80%">

#### **Binary feature:** target plot for a single variable (true survived rate through different values of a variable)
<img src="https://github.com/SauceCat/PDPbox/blob/master/images/target_plot_01.png" width="80%">

#### **Numeric feature:** single variable plot with x_quantile=True, original points and individual lines
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/numeric_03.png" width="80%">

#### **Numeric feature:** single variable plot with percentile_range=(5, 95)
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/numeric_05.png" width="80%">

#### **Numeric feature:** single variable plot with customized grid points
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/numeric_06.png" width="80%">

#### **Numeric feature:** actual predictions plot for a single variable
<img src="https://github.com/SauceCat/PDPbox/blob/master/images/actual_preds_03.PNG" width="80%">

#### **Numeric feature:** target plot for a single variable (true survived rate through different values of a variable)
<img src="https://github.com/SauceCat/PDPbox/blob/master/images/target_plot_02.png" width="80%">

#### **Numeric feature:** target plot for a single variable (multiclass)
<img src="https://github.com/SauceCat/PDPbox/blob/master/images/target_plot_04.png" width="80%">

#### **Onehot encoding feature:** single variable plot with individual lines and original points
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/onehot_01.png" width="80%">

#### **Onehot encoding feature:** single variable plot without centering the lines
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/onehot_02.png" width="80%">

#### **Onehot encoding feature:** actual predictions plot for a single variable
<img src="https://github.com/SauceCat/PDPbox/blob/master/images/actual_preds_02.PNG" width="80%">

#### **Onehot encoding feature:** target plot for a single variable (true survived rate through different values of a variable)
<img src="https://github.com/SauceCat/PDPbox/blob/master/images/target_plot_03.png" width="80%">

#### **Multiclass:** single variable plot with individual lines and original points
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/multi_02.png">

#### **Interaction between two variables:** the complete plot
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/inter_01.png">

#### **Interaction between two variables:** multiclass with only contour plots
<img src="https://github.com/SauceCat/pdpBox/blob/master/images/multi_03.png">
