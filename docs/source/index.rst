PDPbox
======

Python <b>P</b>artial <b>D</b>ependence <b>P</b>lot tool<b>box</b>. 

Motivation
----------

PDPbox is inspired by `ICEbox <https://cran.r-project.org/web/packages/ICEbox/index.html>`_.
The goal is to visualize the influence of certain features on model predictions for supervised machine learning algorithms, 
utilizing partial dependence plots.

While PDPbox is initially designed to be compatible with all scikit-learn models, 
it is important to understand that different models may have different prediction interfaces. 
Standard scikit-learn models typically provide `model.predict` or `model.predict_proba` methods for prediction generation.
Therefore, if your model is a standard scikit-learn model, PDPbox will automatically detect the prediction interface and generate predictions accordingly.

For other models, the prediction interface may vary. 
Thus, PDPbox provides the ability to incorporate a customized prediction function via the `pred_func` parameter in PDPbox methods, 
ensuring broad applicability across various models.

For a more detailed understanding of the prediction generation process, please refer to the following functions:

- `pdpbox.utils._check_model` 
- `pdpbox.utils._calc_preds_each` 
- `pdpbox.utils._calc_preds` 


The common headache
-------------------

When employing "black box" machine learning algorithms such as random forest and boosting, 
deciphering the relationships between predictors and the model outcome can pose a significant challenge. 
While these algorithms provide insights in the form of feature importance, 
it only offers a partial view - telling us which features are impactful, 
but leaving us in the dark about the direction and complexity of their influence.

For instance, a random forest algorithm will tell us that a particular feature is important, 
but it does not tell us whether increasing or decreasing that feature value would result in an increase or decrease in the predicted outcome. 
Moreover, in real-world scenarios, the effects are typically non-monotonic, making the relationships even more intricate.

To tackle this, we need powerful tools capable of illuminating these complex relationships between predictors and model predictions, 
giving us a better understanding of how our model is making its decisions.


.. _home-docs:

.. toctree::
   :maxdepth: 2
   :caption: Home

   Introduction <self>

.. _content-docs:

.. toctree::
   :maxdepth: 2
   :caption: Contents

   papers
   tools
   api