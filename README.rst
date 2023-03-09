.. image:: https://raw.githubusercontent.com/pietrobarbiero/pytorch_explain/master/doc/_static/img/pye_logo_text_dark.svg
    :align: center
    :height: 100px
    :scale: 50 %



-------------



|Build|
|Coverage|

|Docs|
|Dependendencies|

|PyPI license|
|PyPI-version|


.. |Build| image:: https://img.shields.io/travis/com/pietrobarbiero/pytorch_explain?label=Master%20Build&style=for-the-badge
    :alt: Travis (.com)
    :target: https://app.travis-ci.com/pietrobarbiero/pytorch_explain.svg?branch=master

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/pietrobarbiero/pytorch_explain?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/pietrobarbiero/pytorch_explain

.. |Docs| image:: https://img.shields.io/readthedocs/pytorch_explain/latest?style=for-the-badge
    :alt: Read the Docs (version)
    :target: https://pytorch_explain.readthedocs.io/en/latest/

.. |Dependendencies| image:: https://img.shields.io/requires/github/pietrobarbiero/pytorch_explain?style=for-the-badge
    :alt: Requires.io
    :target: https://requires.io/github/pietrobarbiero/pytorch_explain/requirements/?branch=master

.. |PyPI license| image:: https://img.shields.io/pypi/l/torch_explain.svg?style=for-the-badge
   :target: https://pypi.org/project/torch-explain/

.. |PyPI-version| image:: https://img.shields.io/pypi/v/torch_explain?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/torch-explain/

.. image:: https://zenodo.org/badge/356630474.svg
   :target: https://zenodo.org/badge/latestdoi/356630474


`PyTorch, Explain!` is an extension library for PyTorch to develop
explainable deep learning models called Logic Explained Networks (LENs).

It consists of various methods for explainability from a variety of published papers, including the APIs
required to get first-order logic explanations from deep neural networks.

Quick start
-----------

You can install ``torch_explain`` along with all its dependencies from
`PyPI <https://pypi.org/project/torch_explain/>`__:

.. code:: bash

    pip install torch-explain


Example
-----------

For this simple experiment, let's solve the XOR problem
(augmented with 100 dummy features):

.. code:: python

    import torch
    import torch_explain as te
    from torch.nn.functional import one_hot

    x0 = torch.zeros((4, 100))
    x_train = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=torch.float)
    x_train = torch.cat([x_train, x0], dim=1)
    y_train = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    y_train_1h = one_hot(y_train).to(torch.float)

We can instantiate a simple feed-forward neural network
with 3 layers using the ``EntropyLayer`` as the first one:

.. code:: python

    layers = [
        te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=y_train_1h.shape[1]),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(4, 1),
    ]
    model = torch.nn.Sequential(*layers)

We can now train the network by optimizing the cross entropy loss and the
``entropy_logic_loss`` loss function incorporating the human prior towards
simple explanations:

.. code:: python

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(2001):
        optimizer.zero_grad()
        y_pred = model(x_train).squeeze(-1)
        loss = loss_form(y_pred, y_train_1h) + 0.0001 * te.nn.functional.entropy_logic_loss(model)
        loss.backward()
        optimizer.step()

Once trained we can extract first-order logic formulas describing
how the network composed the input features to obtain the predictions:

.. code:: python

    from torch_explain.logic.nn import entropy
    from torch.nn.functional import one_hot

    y1h = one_hot(y_train)
    global_explanations, local_explanations = entropy.explain_classes(model, x_train, y_train, c_threshold=0.5, y_threshold=0.)

Explanations will be logic formulas in disjunctive normal form.
In this case, the explanation will be ``y=1`` if and only if ``(f1 AND ~f2) OR (f2  AND ~f1)``
corresponding to ``f1 XOR f2``.

The function automatically assesses the quality of logic explanations in terms
of classification accuracy and rule complexity.
In this case the accuracy is 100% and the complexity is 4.


Experiments
------------

Training
~~~~~~~~~~

To train the model(s) in the paper, run the scripts and notebooks inside the folder `experiments`.

Results
~~~~~~~~~~

Results on test set and logic formulas will be saved in the folder `experiments/results`.

Data
~~~~~~~~~~

The original datasets can be downloaded from the links provided in the supplementary material of the paper.


Theory
--------
Theoretical foundations can be found in the following papers.

Logic Explained Networks::

    @article{ciravegna2021logic,
      title={Logic explained networks},
      author={Ciravegna, Gabriele and Barbiero, Pietro and Giannini, Francesco and Gori, Marco and Li{\'o}, Pietro and Maggini, Marco and Melacci, Stefano},
      journal={arXiv preprint arXiv:2108.05149},
      year={2021}
    }

Entropy-based LENs::

    @article{barbiero2021entropy,
      title={Entropy-based Logic Explanations of Neural Networks},
      author={Barbiero, Pietro and Ciravegna, Gabriele and Giannini, Francesco and Li{\'o}, Pietro and Gori, Marco and Melacci, Stefano},
      journal={arXiv preprint arXiv:2106.06804},
      year={2021}
    }

Psi network ("learning of constraints")::

    @inproceedings{ciravegna2020constraint,
      title={A Constraint-Based Approach to Learning and Explanation.},
      author={Ciravegna, Gabriele and Giannini, Francesco and Melacci, Stefano and Maggini, Marco and Gori, Marco},
      booktitle={AAAI},
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
* Francesco Giannini, University of Florence, IT.
* Gabriele Ciravegna, University of Florence, IT.
* Dobrik Georgiev, University of Cambridge, UK.


Licence
-------

Copyright 2020 Pietro Barbiero, Francesco Giannini, Gabriele Ciravegna, and Dobrik Georgiev.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
