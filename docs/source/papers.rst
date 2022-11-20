References and Notes
====================

References
----------

.. [R1] Friedman, J. (2001). **Greedy Function Approximation: A Gradient Boosting Machine.**
   The Annals of Statistics, 29(5):1189–1232. (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

.. [R2] Goldstein, A., Kapelner, A., Bleich, J., and Pitkin, E., **Peeking Inside the Black
   Box: Visualizing Statistical Learning With Plots of Individual Conditional Expectation.**
   (2015) Journal of Computational and Graphical Statistics, 24(1): 44-65
   (https://arxiv.org/abs/1309.6392)

.. [R3] Christoph Molnar. (2018). **Interpretable Machine Learning: A Guide for Making
   Black Box Models Explainable.** 5.1 Partial Dependence Plot (PDP) (https://christophm.github
   .io/interpretable-ml-book/pdp.html)

.. [R4] Christoph Molnar. (2018). **Interpretable Machine Learning: A Guide for Making
   Black Box Models Explainable.** 5.2 Individual Conditional Expectation (ICE)
   (https://christophm.github.io/interpretable-ml-book/ice.html)

Notes and Highlights
--------------------

-  One assumption made for the PDP is that the features in :math:`X_{C}` are uncorrelated with the
   features in :math:`X_{S}`. If this assumption is violated, the averages, which are computed
   for the partial dependence plot, incorporate data points that are very unlikely or even
   impossible.

   For example, it's unreasonable to claim that height and weight is uncorrelated. If height is the
   feature to plot, only changing height through different values would create data points like
   someone is 2 meters but weighting below 50kg. Considering PDP is calculated by averaging
   through all data points, with these kind of unreasonable data points, the result might not
   be trustworthy. [R3]_

   .. note:: check :code:`data_transformer` parameter in :code:`pdp_isolate` and :code:`pdp_interact`.

------------

-  Some PD visualisations don’t include the feature distribution. Omitting the distribution can be
   misleading, because you might over-interpret the line in regions, with almost no feature
   values. [R3]_

   .. note:: check :code:`plot_pts_dist` parameter in :code:`pdp_plot`.

------------

-  There is one issue with ICE plots: It can be hard to see if the individual conditional
   expectation curves differ between individuals, because they start at different :math:`\hat{f}
   (x)`. [R4]_

   .. note:: check :code:`center` parameters in :code:`pdp_plot` and :code:`pdp_interact_plot`.

------------


-  When many ICE curves are drawn the plot can become overcrowded and you don’t see anything any
   more. [R4]_

   .. note:: check :code:`frac_to_plot` and :code:`cluster` parameters in :code:`pdp_plot` and :code:`pdp_interact_plot`.



