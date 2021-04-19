PYTORCH EXPLAIN DOCUMENTATION
===============================

`PyTorch, Explain!` is an extension library for PyTorch to develop explainable deep learning models.

It consists of various methods for explainability from a variety of published papers, including the APIs
required to get first-order logic explanations from deep neural networks.

Quick start
-----------

You can install Deep Logic along with all its dependencies from
`PyPI <https://pypi.org/project/deep-logic/>`__:

.. code:: bash

    pip install -r requirements.txt deep-logic


Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/pietrobarbiero/deep-logic>`__.


.. toctree::
    :caption: User Guide
    :maxdepth: 2

    user_guide/installation
    user_guide/tutorial_deepnn
    user_guide/tutorial_psi
    user_guide/contributing
    user_guide/running_tests

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    modules/logic/relu_nn
    modules/logic/psi_nn
    modules/nn/linear
    modules/utils/base
    modules/utils/relu_nn
    modules/utils/psi_nn
    modules/utils/selection


.. toctree::
    :caption: Copyright
    :maxdepth: 1

    user_guide/authors
    user_guide/licence


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`