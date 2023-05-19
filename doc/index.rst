PYTORCH EXPLAIN DOCUMENTATION
===============================


`PyTorch, Explain!` is an extension library for PyTorch to develop
explainable deep learning models going beyond the current accuracy-interpretability trade-off.

The library includes a set of tools to develop:

* Deep Concept Reasoner (Deep CoRe): an interpretable concept-based model going
  **beyond the current accuracy-interpretability trade-off**;
* Concept Embedding Models (CEMs): a class of concept-based models going
  **beyond the current accuracy-explainability trade-off**;
* Logic Explained Networks (LENs): a class of concept-based models generating
  accurate compound logic explanations for their predictions
  **without the need for a post-hoc explainer**.


Quick start
-----------

You can install ``torch_explain`` along with all its dependencies from
`PyPI <https://pypi.org/project/torch_explain/>`__:

.. code:: bash

    pip install torch-explain


Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/pietrobarbiero/pytorch_explain>`__.


.. toctree::
    :caption: User Guide
    :maxdepth: 2

    user_guide/installation
    user_guide/tutorial_lens
    user_guide/tutorial_cem
    user_guide/tutorial_dcr
    user_guide/contributing
    user_guide/running_tests

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    modules/logic/nn/entropy
    modules/logic/nn/psi
    modules/logic/nn/utils
    modules/logic/metrics
    modules/logic/utils
    modules/nn/logic
    modules/nn/functional/loss
    modules/nn/functional/prune
    modules/nn/concepts
    modules/nn/semantics


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


Benchmark datasets
-------------------------

We provide a suite of 3 benchmark datasets to evaluate the performance of our models
in the folder `torch_explain/datasets`. These 3 datasets were proposed as benchmarks
for concept-based models in the paper "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off".

Real-world datasets can be downloaded from the links provided in the supplementary material of the paper.


Theory
--------
Theoretical foundations can be found in the following papers.

Deep Concept Reasoning::

    @article{barbiero2023interpretable,
      title={Interpretable Neural-Symbolic Concept Reasoning},
      author={Barbiero, Pietro and Ciravegna, Gabriele and Giannini, Francesco and Zarlenga, Mateo Espinosa and Magister, Lucie Charlotte and Tonda, Alberto and Lio, Pietro and Precioso, Frederic and Jamnik, Mateja and Marra, Giuseppe},
      journal={arXiv preprint arXiv:2304.14068},
      year={2023}
    }

Concept Embedding Models::

    @article{espinosa2022concept,
      title={Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off},
      author={Espinosa Zarlenga, Mateo and Barbiero, Pietro and Ciravegna, Gabriele and Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and Shams, Zohreh and Precioso, Frederic and Melacci, Stefano and Weller, Adrian and others},
      journal={Advances in Neural Information Processing Systems},
      volume={35},
      pages={21400--21413},
      year={2022}
    }


Logic Explained Networks::

    @article{ciravegna2023logic,
      title={Logic explained networks},
      author={Ciravegna, Gabriele and Barbiero, Pietro and Giannini, Francesco and Gori, Marco and Li{\'o}, Pietro and Maggini, Marco and Melacci, Stefano},
      journal={Artificial Intelligence},
      volume={314},
      pages={103822},
      year={2023},
      publisher={Elsevier}
    }

Entropy-based LENs::

    @inproceedings{barbiero2022entropy,
      title={Entropy-based logic explanations of neural networks},
      author={Barbiero, Pietro and Ciravegna, Gabriele and Giannini, Francesco and Li{\'o}, Pietro and Gori, Marco and Melacci, Stefano},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={36},
      number={6},
      pages={6046--6054},
      year={2022}
    }

Psi network ("learning of constraints")::

    @inproceedings{ciravegna2020constraint,
      title={A constraint-based approach to learning and explanation},
      author={Ciravegna, Gabriele and Giannini, Francesco and Melacci, Stefano and Maggini, Marco and Gori, Marco},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={34},
      number={04},
      pages={3658--3665},
      year={2020}
    }


Learning with constraints::

    @inproceedings{marra2019lyrics,
      title={LYRICS: A General Interface Layer to Integrate Logic Inference and Deep Learning},
      author={Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and Gori, Marco},
      booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
      pages={283--298},
      year={2019},
      organization={Springer}
    }

Constraints theory in machine learning::

    @book{gori2017machine,
      title={Machine Learning: A constraint-based approach},
      author={Gori, Marco},
      year={2017},
      publisher={Morgan Kaufmann}
    }


Authors
-------

* `Pietro Barbiero <http://www.pietrobarbiero.eu/>`__, University of Cambridge, UK.
* Mateo Espinosa Zarlenga, University of Cambridge, UK.
* Giuseppe Marra, Katholieke Universiteit Leuven, BE.
* Steve Azzolin, University of Trento, IT.
* Francesco Giannini, University of Florence, IT.
* Gabriele Ciravegna, University of Florence, IT.
* Dobrik Georgiev, University of Cambridge, UK.


Licence
-------

Copyright 2020 Pietro Barbiero, Mateo Espinosa Zarlenga, Giuseppe Marra,
Steve Azzolin, Francesco Giannini, Gabriele Ciravegna, and Dobrik Georgiev.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
