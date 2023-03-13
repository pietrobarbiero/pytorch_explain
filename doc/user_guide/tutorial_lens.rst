Logic Explained Network (LENs) tutorial
==========================================

Entropy-based LENs
-----------------------

For this simple tutorial, let's solve the XOR problem
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


:math:`\psi` LENs
-----------------------

For this simple tutorial, let's solve the XOR problem
using a :math:`\psi` LEN:

.. code:: python

    import torch
    import torch_explain as te

    x_train = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=torch.float)
    y_train = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1)

We can instantiate a simple :math:`\psi` network
with 3 layers using **sigmoid activation functions only**:

.. code:: python

    layers = [
        torch.nn.Linear(x_train.shape[1], 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 5),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    ]
    model = torch.nn.Sequential(*layers)

We can now train the network by optimizing the binary cross entropy loss and the
``l1_loss`` loss function incorporating the human prior towards
simple explanations. The :math:`\psi` networks needs to be pruned during training
to simplify the internal architecture (here pruning happens at epoch 1000):

.. code:: python

    from torch_explain.nn.functional import prune_equal_fanin

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCELoss()
    model.train()
    for epoch in range(6001):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_form(y_pred, y_train) + 0.000001 * te.nn.functional.l1_loss(model)
        loss.backward()
        optimizer.step()

        model = prune_equal_fanin(model, epoch, prune_epoch=1000, k=2)

Once trained we can extract first-order logic formulas describing
how the network composed the input features to obtain the predictions:

.. code:: python

    from torch_explain.logic.nn import psi
    from torch.nn.functional import one_hot

    y1h = one_hot(y_train.squeeze().long())
    explanation = psi.explain_class(model, x_train)

Explanations will be logic formulas in disjunctive normal form.
In this case, the explanation will be ``y=1 IFF (f1 AND ~f2) OR (f2  AND ~f1)``
corresponding to ``y=1 IFF f1 XOR f2``.

The quality of the logic explanation can **quantitatively** assessed in terms
of classification accuracy and rule complexity as follows:

.. code:: python

    from torch_explain.logic.metrics import test_explanation, complexity

    accuracy, preds = test_explanation(explanation, x_train, y1h, target_class=1)
    explanation_complexity = complexity(explanation)

In this case the accuracy is 100% and the complexity is 4.