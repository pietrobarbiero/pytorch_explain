Concept Embeddings tutorial
==========================================

Limits of Concept Bottleneck Models
------------------------------------------

For this simple tutorial, let's approach
the trigonometry benchmark dataset with a concept bottleneck model:

.. code:: python

    import torch
    import torch_explain as te
    from torch_explain import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    x, c, y = datasets.trigonometry(500)
    x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

We can instantiate a simple concept encoder
to map the input features to the concept space and then
a task predictor to map concepts to task predictions:

.. code:: python

    concept_encoder = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 8),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(8, c.shape[1]),
        torch.nn.Sigmoid(),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(c.shape[1], 1),
    )
    model = torch.nn.Sequential(concept_encoder, task_predictor)

We can now train the network by optimizing the cross entropy loss
on both concepts and tasks:

.. code:: python

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(501):
        optimizer.zero_grad()

        # generate concept and task predictions
        c_pred = concept_encoder(x_train)
        y_pred = task_predictor(c_pred)

        # update loss
        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        loss = concept_loss + 0.5*task_loss

        loss.backward()
        optimizer.step()

Once trained we can check the performance of the model on the test set:

.. code:: python

    c_pred = concept_encoder(x_test)
    y_pred = task_predictor(c_pred)

    concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
    task_accuracy = accuracy_score(y_test, y_pred > 0)

As you can see the performance of the model is not great as the task
task accuracy is around ~80%. Can we do better?


Concept Embeddings
------------------------------

Using concept embeddings we can solve our problem much more efficiently.
We just need to define a task predictor and a concept encoder using a
concept embedding layer:

.. code:: python

    import torch
    import torch_explain as te

    embedding_size = 8
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
        c_emb, c_pred = concept_encoder(x_train)
        y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))

        # compute loss
        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        loss = concept_loss + 0.5*task_loss

        loss.backward()
        optimizer.step()

Once trained we can check the performance of the model on the test set:

.. code:: python

    c_emb, c_pred = concept_encoder.forward(x_test)
    y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))

    concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
    task_accuracy = accuracy_score(y_test, y_pred > 0)

As you can see the performance of the model is now great as the task
task accuracy is around ~100%.
