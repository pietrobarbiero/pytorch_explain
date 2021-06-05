LENs tutorial
==========================================

For this simple tutorial, let's solve the XOR problem
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
        te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=2),
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
        y_pred = model(x_train).squeeze(-1)
        loss = loss_form(y_pred, y_train) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
        loss.backward()
        optimizer.step()

Once trained we can extract first-order logic formulas describing
how the network composed the input features to obtain the predictions:

.. code:: python

    from torch_explain.logic import explain_class
    from torch.nn.functional import one_hot

    y1h = one_hot(y_train)
    explanation, _ = explain_class(model, x_train, y1h, x_train, y1h, target_class=1)

Explanations will be logic formulas in disjunctive normal form.
In this case, the explanation will be ``y=1 IFF (f1 AND ~f2) OR (f2  AND ~f1)``
corresponding to ``y=1 IFF f1 XOR f2``.
