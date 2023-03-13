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
    :target: https://app.travis-ci.com/github/pietrobarbiero/pytorch_explain

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
explainable deep learning models going beyond the current accuracy-explainability trade-off.

The library includes a set of tools to develop:

* Concept Embedding Models (CEMs): a class of concept-based models going beyond
  the accuracy-explainability trade-off;
* Logic Explained Networks (LENs): a class of concept-based models generating
  accurate compound logic explanations for their predictions **without the need for a post-hoc explainer**.

Table of Content
-----------------
* `Quick start <https://github.com/pietrobarbiero/pytorch_explain#quick-start>`_
* `Quick tutorial on Concept Embedding Models <https://github.com/pietrobarbiero/pytorch_explain#quick-tutorial-on-concept-embedding-models>`_
* `Quick tutorial on Logic Explained Networks <https://github.com/pietrobarbiero/pytorch_explain#quick-tutorial-on-logic-explained-networks>`_
* `Benchmark datasets <https://github.com/pietrobarbiero/pytorch_explain#benchmark-datasets>`_
* `Theory <https://github.com/pietrobarbiero/pytorch_explain#theory>`_
* `Authors <https://github.com/pietrobarbiero/pytorch_explain#authors>`_
* `Licence <https://github.com/pietrobarbiero/pytorch_explain#licence>`_

Quick start
---------------

You can install ``torch_explain`` along with all its dependencies from
`PyPI <https://pypi.org/project/torch_explain/>`__:

.. code:: bash

    pip install torch-explain


Quick tutorial on Concept Embedding Models
-----------------------------------------------

Using concept embeddings we can solve concept-based problems very efficiently!
For this simple tutorial, let's approach the trigonometry benchmark dataset:

.. code:: python

    import torch
    import torch_explain as te
    from torch_explain import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    x, c, y = datasets.trigonometry(500)
    x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

We just need to define a task predictor and a concept encoder using a
concept embedding layer:

.. code:: python

    import torch
    import torch_explain as te

    concept_encoder = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 10),
        torch.nn.LeakyReLU(),
        te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(c.shape[1]*embedding_size, 1),
    )
    model = torch.nn.Sequential(concept_encoder, task_predictor)

We can now train the network by optimizing the cross entropy loss
on concepts and tasks:

.. code:: python

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(501):
        optimizer.zero_grad()

        # generate concept and task predictions
        c_emb, c_pred = concept_embedder(x_train)
        y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))

        # compute loss
        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        loss = concept_loss + 0.5*task_loss

        loss.backward()
        optimizer.step()

Once trained we can check the performance of the model on the test set:

.. code:: python

    c_emb, c_pred = concept_embedder.forward(x_test)
    y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))

    task_accuracy = accuracy_score(y_test, y_pred > 0)
    concept_accuracy = accuracy_score(c_test, c_pred > 0.5)

As you can see the performance of the model is now great as the task
task accuracy is around ~100%.


Quick tutorial on Logic Explained Networks
---------------------------------------------

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


Benchmark datasets
-------------------------

We provide a suite of 3 benchmark datasets to evaluate the performance of our models
in the folder `torch_explain/datasets`. These 3 datasets were proposed as benchmarks
for concept-based models in the paper "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off".

Real-world datasets can be downloaded from the links provided in the supplementary material of the paper.


Theory
--------
Theoretical foundations can be found in the following papers.

Concept Embedding Models::

    @inproceedings{zarlengaconcept,
      title={Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off},
      author={Zarlenga, Mateo Espinosa and Barbiero, Pietro and Ciravegna, Gabriele and Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and Shams, Zohreh and Precioso, Frederic and Melacci, Stefano and Weller, Adrian and others},
      booktitle={Advances in Neural Information Processing Systems}
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
* Mateo Espinosa Zarlenga, University of Cambridge, UK.
* Steve Azzolin, University of Trento, IT.
* Francesco Giannini, University of Florence, IT.
* Gabriele Ciravegna, University of Florence, IT.
* Dobrik Georgiev, University of Cambridge, UK.


Licence
-------

Copyright 2020 Pietro Barbiero, Mateo Espinosa Zarlenga, Steve Azzolin, Francesco Giannini, Gabriele Ciravegna, and Dobrik Georgiev.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
