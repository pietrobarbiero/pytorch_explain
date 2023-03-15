import unittest

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch_explain as te
from torch_explain import datasets


def train_concept_bottleneck_model(x, c, y, embedding_size=1):
    x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

    if embedding_size > 1:
        # concept embedding model
        encoder = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 10),
            torch.nn.LeakyReLU(),
        )
        concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], embedding_size)
        task_predictor = torch.nn.Sequential(
            torch.nn.Linear(c.shape[1]*embedding_size, 1),
        )
        model = torch.nn.Sequential(encoder, concept_embedder, task_predictor)
    else:
        # standard concept bottleneck model
        concept_embedder = torch.nn.Sequential(
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
        model = torch.nn.Sequential(concept_embedder, task_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(501):
        optimizer.zero_grad()

        if embedding_size > 1:
            h = encoder(x_train)
            c_emb, c_pred = concept_embedder.forward(h, [0, 1], c_train, train=True)
            y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))
        else:
            c_pred = concept_embedder(x_train)
            y_pred = task_predictor(c_pred)

        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        loss = concept_loss + 0.5*task_loss
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            if embedding_size > 1:
                h = encoder(x_test)
                c_emb, c_pred = concept_embedder.forward(h, [0, 1], c_test, train=False)
                y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))
            else:
                c_pred = concept_embedder(x_test)
                y_pred = task_predictor(c_pred)

            task_accuracy = accuracy_score(y_test, y_pred > 0)
            concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
            print(f'Epoch {epoch}: loss {loss:.4f} task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')

    return model


class TestTemplateObject(unittest.TestCase):
    def test_concept_embedding(self):
        x, c, y = datasets.xor(500)
        train_concept_bottleneck_model(x, c, y, embedding_size=1)
        train_concept_bottleneck_model(x, c, y, embedding_size=8)

        x, c, y = datasets.trigonometry(500)
        train_concept_bottleneck_model(x, c, y, embedding_size=1)
        train_concept_bottleneck_model(x, c, y, embedding_size=8)

        x, c, y = datasets.dot(500)
        train_concept_bottleneck_model(x, c, y, embedding_size=1)
        train_concept_bottleneck_model(x, c, y, embedding_size=8)

        return

    def test_concept_interventions(self):
        x, c, y = datasets.dot(500)

        # concept embedding model
        encoder = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 10),
            torch.nn.LeakyReLU(),
        )
        h = encoder(x)

        concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], 8)
        c_emb, c_pred = concept_embedder.forward(h, [0, 1], c, train=True)

        concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], 8,
                                                  active_intervention_values=1, inactive_intervention_values=0,
                                                  intervention_idxs=[0, 1])
        c_emb, c_pred = concept_embedder.forward(h, train=True)

        concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], 8, training_intervention_prob=0)
        c_emb, c_pred = concept_embedder.forward(h)

        concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], 8)
        c_emb, c_pred = concept_embedder.forward(h, train=True)

        concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], 8)
        c_emb, c_pred = concept_embedder.forward(h, intervention_idxs=[10])

        return


if __name__ == '__main__':
    unittest.main()
