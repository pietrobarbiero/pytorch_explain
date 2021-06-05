.. image:: https://raw.githubusercontent.com/pietrobarbiero/pytorch_explain/master/doc/_static/img/pye_logo_text_dark.svg
    :align: left
    :height: 100px
    :scale: 50 %


|Build|
|Coverage|

|Docs|
|Dependendencies|

|PyPI license|
|PyPI-version|


.. |Build| image:: https://img.shields.io/travis/pietrobarbiero/pytorch_explain?label=Master%20Build&style=for-the-badge
    :alt: Travis (.org)
    :target: https://travis-ci.org/pietrobarbiero/pytorch_explain

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/pietrobarbiero/pytorch_explain?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/pietrobarbiero/pytorch_explain

.. |Docs| image:: https://img.shields.io/readthedocs/pytorch_explain/latest?style=for-the-badge
    :alt: Read the Docs (version)
    :target: https://pytorch_explain.readthedocs.io/en/latest/

.. |Dependendencies| image:: https://img.shields.io/requires/github/pietrobarbiero/pytorch_explain?style=for-the-badge
    :alt: Requires.io
    :target: https://requires.io/github/pietrobarbiero/pytorch_explain/requirements/?branch=master

.. |PyPI license| image:: https://img.shields.io/pypi/l/pytorch_explain.svg?style=for-the-badge
   :target: https://pypi.python.org/pypi/pytorch_explain/

.. |PyPI-version| image:: https://img.shields.io/pypi/v/pytorch_explain?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.python.org/pypi/pytorch_explain/


`PyTorch, Explain!` is an extension library for PyTorch to develop
explainable deep learning models called Logic Explained Networks (LENs).

It consists of various methods for explainability from a variety of published papers, including the APIs
required to get first-order logic explanations from deep neural networks.

Quick start
-----------

You can install ``torch_explain`` along with all its dependencies from
`PyPI <https://pypi.org/project/pytorch_explain/>`__:

.. code:: bash

    pip install -r requirements.txt torch_explain


Example
-----------

For this simple experiment, let's solve the XOR problem
(augmented with 100 dummy features):

.. code:: python

    import torch
    import torch_explain as te

    x0 = torch.zeros((4, 100))
    x_train = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=torch.float)
    x_train = torch.cat([x_train, x0], dim=1)
    y_train = torch.tensor([0, 1, 1, 0], dtype=torch.long)

We can instantiate a simple feed-forward neural network
with 3 layers using the ``EntropyLayer`` as the first one:

.. code:: python

    layers = [
        te.nn.EntropyLinear(x.shape[1], 10, n_classes=2),
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1001):
        optimizer.zero_grad()
        y_pred = model(x).squeeze(-1)
        loss = loss_form(y_pred, y) + /
               0.00001 * te.nn.functional.entropy_logic_loss(model)
        loss.backward()
        optimizer.step()

Once trained we can extract first-order logic formulas describing
how the network composed the input features to obtain the predictions:

.. code:: python

    y1h = one_hot(y)
    explanation, _ = explain_class(model, x, y1h, x, y1h, target_class=1)

Explanations will be logic formulas in disjunctive normal form.
In this case, the explanation will be ``y=1 IFF (f1 AND ~f2) OR (f2  AND ~f1)``
corresponding to ``y=1 IFF f1 XOR f2``.


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

Learning of constraints::

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
