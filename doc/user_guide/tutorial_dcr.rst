Deep Concept Reasoning tutorial
==========================================

Limits of Concept Embeddings
--------------------------------

For this simple tutorial, let's use
the XOR benchmark dataset:

.. code:: python

    import torch
    import torch_explain as te
    from torch_explain import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    x, c, y = datasets.xor(500)
    x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

Using concept embeddings we can solve our problem efficiently.
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

    task_accuracy = accuracy_score(y_test, y_pred > 0)
    concept_accuracy = accuracy_score(c_test, c_pred > 0.5)

As you can see the performance of the model is now great as the task
task accuracy is around ~100%.

However, we cannot explain exactly the reasoning process of the
model! How are concept embeddings used to predict the task?
To answer this question we need to use Deep Concept Reasoning.


Deep Concept Reasoning
----------------------------

Using deep concept reasoning we can solve the same problem as above,
but with an intrinsically interpretable model! In fact, Deep Concept Reasoners (Deep CoRes)
make task predictions by means of interpretable logic rules using concept embeddings.

Using the same example as before, we can just change the task predictor
using a Deep CoRe layer:

.. code:: python

    from torch_explain.nn.concepts import ConceptReasoningLayer
    import torch.nn.functional as F

    y_train = F.one_hot(y_train.long().ravel()).float()
    y_test = F.one_hot(y_test.long().ravel()).float()

    task_predictor = ConceptReasoningLayer(embedding_size, y_train.shape[1])
    model = torch.nn.Sequential(concept_encoder, task_predictor)


We can now train the network by optimizing the cross entropy loss
on concepts and tasks:

.. code:: python

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCELoss()
    model.train()
    for epoch in range(501):
        optimizer.zero_grad()

        # generate concept and task predictions
        c_emb, c_pred = concept_encoder(x_train)
        y_pred = task_predictor(c_emb, c_pred)

        # compute loss
        concept_loss = loss_form(c_pred, c_train)
        task_loss = loss_form(y_pred, y_train)
        loss = concept_loss + 0.5*task_loss

        loss.backward()
        optimizer.step()

Once trained the Deep CoRe layer can explain its predictions by
providing both local and global logic rules:


.. code:: python

    local_explanations = task_predictor.explain(c_emb, c_pred, 'local')
    global_explanations = task_predictor.explain(c_emb, c_pred, 'global')


For global explanations, the reasoner will return a dictionary with entries such as
``{'class': 'y_0', 'explanation': '~c_0 & ~c_1', 'count': 94}``, specifying
for each logic rule, the task it is associated with and the number of samples
associated with the explanation.


