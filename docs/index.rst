======
PDPbox
======

python partial dependence plot toolbox
This repository is inspired by ICEbox. The goal is to visualize the impact of certain features towards model
prediction for any supervised learning algorithm. (now support all scikit-learn algorithms)


The common problem
------------------
When using black box machine learning algorithms like random forest and boosting, it is hard to understand the
relations between predictors and model outcome. For example, in terms of random forest, all we get is the feature
importance. Although we can know which feature is significantly influencing the outcome based on the importance
calculation, it really sucks that we don’t know in which direction it is influencing. And in most of the real cases,
the effect is non-monotonic. We need some powerful tools to help understanding the complex relations
between predictors and model prediction.
PDPbox aims to wrap up and enrich some useful functions mentioned in ICEbox in Python.



.. toctree::
   :maxdepth: 2

   info_plots
   pdp