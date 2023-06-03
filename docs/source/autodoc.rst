
Semi-automated docstring generation
===================================

The primary motivation behind semi-automated docstring generation is to centralize the documentation of parameters, 
thereby facilitating streamlined docstring generation for functions and classes based on the relevant parameters. 
This strategy is particularly effective in eliminating redundancy, 
especially when the same parameter is utilized by multiple functions or classes.

Additionally, it helps avoid discrepancies in the documentation where the same parameter is described differently 
across various functions or classes. 
Most importantly, this automatic generation method guarantees a standardized format for all docstrings, 
thereby enhancing consistency and readability.


Centralized Docstring Storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this project, I have consolidated the docstrings for attributes, parameters, and return values into a central JSON file, 
referred to as :code:`parameters.json`. This file consists of a list of dictionaries, with each dictionary structured as follows:

.. code-block:: json

    {
        "param": "show_percentile",
        "type": "bool",
        "desp": "If True, percentiles are shown in the plot.",
        "target": {
            "params": [
                "info_plots._InfoPlot.plot",
                "info_plots._InteractInfoPlot.plot",
                "pdp.PDPIsolate.plot",
                "pdp.PDPInteract.plot"
            ],
            "returns": [],
            "attrs": [
                "styles.InfoPlotStyle",
                "styles.InteractInfoPlotStyle",
                "styles.PDPIsolatePlotStyle",
                "styles.PDPInteractPlotStyle"
            ]
        }
    }

The dictionary keys are explained as follows:

- :code:`param`: The name of the parameter.

- :code:`type`: The type of the parameter.

- :code:`desp`: The description of the parameter.

Given that the :code:`numpydoc` format is used in this project, each key corresponds to a specific section required 
in the docstring according to :code:`numpydoc` specifications.

.. code-block:: 

    Parameters
    ----------
    <param> : <type>
        <desp>

    Returns
    -------
    <type>
        <desp>

    Attributes
    ----------
    <param> : <type>
        <desp>

- :code:`target`: The mapping between the parameter and the corresponding functions or classes. 
  This is a dictionary with three keys: :code:`params`, :code:`returns`, and :code:`attrs`.
  Each key points to a list of strings, each representing the full names of functions or classes.
  The format for classes is :code:`module_name.class_name` and for functions, 
  it is :code:`module_name.function_name` or :code:`module_name.class_name.function_name`.

    - :code:`params`: The functions which take this parameter.

    - :code:`returns`: The functions which return this parameter.

    - :code:`attrs`: The classes which have this attribute.

For a complete list of parameters, please refer to :code:`assets/docs_helper/parameters.json`.


Function Docstring
~~~~~~~~~~~~~~~~~~

A function's docstring generally contains :code:`Parameters` and :code:`Returns`. 
When it comes to :code:`Parameters`, the list of parameter names and their default values can be extracted from the function signature. 
Subsequently, the docstring for each parameter can be referenced from the central :code:`parameters.json` file.

As for :code:`Returns`, we don't currently have an efficient method to extract the return value solely from the code. 
However, the list of return values can be gathered from the :code:`parameters.json` file as the target functions are already included 
in the :code:`returns` key of the :code:`target` dictionary. 
Although this could be applied for :code:`Parameters` as well, it's generally more reliable to utilize the function signature to ensure accuracy.

A placeholder is required to indicate the placement of the docstring. This can be achieved as follows:

.. code-block:: python

    def plot(
        self,
        which_classes=None,
        show_percentile=False,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """
        The plot function for `TargetPlot` and `PredictPlot`.
        <_InfoPlot.plot-DOC_FUNC>
        <_InfoPlot.plot-DOC_RETURN>
        """
        return self._plot(
            which_classes,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
            num_bins=self.feature_info.num_bins,
        )

The format for placeholders is as follows:

- :code:`Parameters`: :code:`<_InfoPlot.plot-DOC_FUNC>`, format: :code:`<function_name-DOC_FUNC>` or :code:`<class_name.function_name-DOC_FUNC>`.

- :code:`Returns`: :code:`<_InfoPlot.plot-DOC_RETURN>`, format: :code:`<function_name-DOC_RETURN>` or :code:`<class_name.function_name-DOC_RETURN>`.


Class Docstring
~~~~~~~~~~~~~~~

The logic for generating a class method's docstring is essentially the same as for function docstring generation. 
The key difference lies in the addition of docstrings for the class attributes. 
Similar to the approach for :code:`Returns` in the function docstring, 
we cannot easily extract attributes from the code, so the same logic is applied. 
However, compared to :code:`Returns`, :code:`Attributes` generally include more items. 

To ensure accuracy, we manually note down the list of attribute names within the class definition, 
positioned between two attribute placeholders, as shown below:

.. code-block:: python

    class TargetPlot(_InfoPlot):
        """
        Generates plots displaying the average values of target variables
        across distinct groups (or buckets) of a single feature.  
        
        These plots provide insights into how the target's average values change with
        respect to the different groups of a chosen feature. This understanding is
        crucial for comprehensive feature analysis and facilitates the interpretation
        of model predictions.
        <TargetPlot-DOC_ATTR>
        df
        feature_info
        feature_cols
        target
        n_classes
        plot_type
        plot_engines
        count_df
        summary_df
        target_lines
        <TargetPlot-DOC_ATTR>

        Methods
        -------
        plot(**kwargs) :
            Generates the plot.
        """

The format for the attribute placeholder is as follows: 

.. code-block:: 

   <class_name-DOC_ATTR>
   ...
   list of attribute names
   ...
   <class_name-DOC_ATTR>


Docstring Generation
~~~~~~~~~~~~~~~~~~~~

Here's the summarized sequence of actions for generating docstrings:

-   Record all parameters, returns, and attributes in the central :code:`parameters.json` file.

-   Insert placeholders in target functions and classes. It isn't necessary to add docstrings to every function and class. 
    We only automatically generate docstrings for those functions and classes that contain placeholders.

-   Run the following command to generate docstrings:

    .. code-block:: bash

        cd assets/docs_helper
        python generate_docstring.py --param_file parameters.json --py_files <py_file1> <py_file2> ...

You can specify multiple python files. The docstrings will be generated and incorporated into the specified python files, 
saved as new python files in the same directory as the original python files. The new file's name is :code:`<py_file>_updated.py`.

For instance, the python file with embedded docstrings newly generated for :code:`info_plots.py` 
will be named :code:`info_plots_updated.py` and placed in the same directory.
You can use "diff" tools, like the comparison function in Visual Studio Code, 
to review the generated docstrings and decide whether you want to utilize the generated docstrings. 
If you accept the changes, simply replace the original python file with the newly generated one.


Examples
~~~~~~~~

**Function Docstring**

Before:

.. code-block:: python

    def plot(
        self,
        which_classes=None,
        show_percentile=False,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """
        The plot function for `TargetPlot` and `PredictPlot`.
        <_InfoPlot.plot-DOC_FUNC>
        <_InfoPlot.plot-DOC_RETURN>
        """
        return self._plot(
            which_classes,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
            num_bins=self.feature_info.num_bins,
        )

After:

.. code-block:: python

    def plot(
        self,
        which_classes=None,
        show_percentile=False,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """
        The plot function for `TargetPlot` and `PredictPlot`.

        Parameters
        ----------
        which_classes : list of int, optional
            List of class indices to plot. If None, all classes will be plotted.
            Default is None.
        show_percentile : bool, optional
            If True, percentiles are shown in the plot. Default is False.
        figsize : tuple or None, optional
            The figure size for matplotlib or plotly figure. If None, the default
            figure size is used. Default is None.
        dpi : int, optional
            The resolution of the plot, measured in dots per inch. Only applicable when
            `engine` is 'matplotlib'. Default is 300.
        ncols : int, optional
            The number of columns of subplots in the figure. Default is 2.
        plot_params : dict or None, optional
            Custom plot parameters that control the style and aesthetics of the plot.
            Default is None.
        engine : {'matplotlib', 'plotly'}, optional
            The plotting engine to use. Default is plotly.
        template : str, optional
            The template to use for plotly plots. Only applicable when `engine` is
            'plotly'. Reference: https://plotly.com/python/templates/ Default is
            plotly_white.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            A Matplotlib or Plotly figure object depending on the plot engine being
            used.
        dict of matplotlib.axes.Axes or None
            A dictionary of Matplotlib axes objects. The keys are the names of the
            axes. The values are the axes objects. If `engine` is 'ploltly', it is
            None.
        pd.DataFrame
            A DataFrame that contains the summary statistics of target (for target
            plot) or predict (for predict plot) values for each feature bucket.
        """
        return self._plot(
            which_classes,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
            num_bins=self.feature_info.num_bins,
        )


**Class Docstring**

Before:

.. code-block:: python

    class TargetPlot(_InfoPlot):
        """
        Generates plots displaying the average values of target variables
        across distinct groups (or buckets) of a single feature.  
        
        These plots provide insights into how the target's average values change with
        respect to the different groups of a chosen feature. This understanding is
        crucial for comprehensive feature analysis and facilitates the interpretation
        of model predictions.
        <TargetPlot-DOC_ATTR>
        df
        feature_info
        feature_cols
        target
        n_classes
        plot_type
        plot_engines
        count_df
        summary_df
        target_lines
        <TargetPlot-DOC_ATTR>

        Methods
        -------
        plot(**kwargs) :
            Generates the plot.
        """

After:

.. code-block:: python

    class TargetPlot(_InfoPlot):
        """
        Generates plots displaying the average values of target variables
        across distinct groups (or buckets) of a single feature.

        These plots provide insights into how the target's average values change with
        respect to the different groups of a chosen feature. This understanding is
        crucial for comprehensive feature analysis and facilitates the interpretation
        of model predictions.

        Attributes
        ----------
        df : pd.DataFrame
            A processed DataFrame that includes feature and target (for target plot) or
            predict (for predict plot) columns, feature buckets, along with the count
            of samples within each bucket.
        feature_info : :class:`FeatureInfo`
            An instance of the `FeatureInfo` class.
        feature_cols : list of str
            List of feature columns.
        target : list of int
            List of target indices. For binary and regression problems, the list will
            be just [0]. For multi-class targets, the list is the class indices.
        n_classes : int
            The number of classes inferred from the target columns.
        plot_type : str
            The type of the plot to be generated.
        plot_engines : dict
            A dictionary that maps plot types to their plotting engines.
        count_df : pd.DataFrame
            A DataFrame that contains the count as well as the normalized count
            (percentage) of samples within each feature bucket.
        summary_df : pd.DataFrame
            A DataFrame that contains the summary statistics of target (for target
            plot) or predict (for predict plot) values for each feature bucket.
        target_lines : list of pd.DataFrame
            A list of DataFrames, each DataFrame includes aggregate metrics across
            feature buckets for a target (for target plot) or predict (for predict
            plot) variable. For binary and regression problems, the list will contain a
            single DataFrame. For multi-class targets, the list will contain a
            DataFrame for each class.

        Methods
        -------
        plot(**kwargs) :
            Generates the plot.
        """