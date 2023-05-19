import unittest

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import torch_explain as te
from torch_explain import datasets
from torch_explain.nn.concepts import ConceptReasoningLayer
from torch_explain.nn.semantics import GodelTNorm, ProductTNorm


def train_concept_bottleneck_model(x, c, y, embedding_size=1, logic=GodelTNorm(), temperature=100):
    x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

    y_train = F.one_hot(y_train.long().ravel()).float()
    y_test = F.one_hot(y_test.long().ravel()).float()

    # concept embedding model
    encoder = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 10),
        torch.nn.LeakyReLU(),
    )
    concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], embedding_size)
    task_predictor = ConceptReasoningLayer(embedding_size, y_train.shape[1], logic, temperature)
    model = torch.nn.Sequential(encoder, concept_embedder, task_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.BCELoss()
    model.train()
    for epoch in range(501):
        optimizer.zero_grad()

        h = encoder(x_train)
        c_emb, c_pred = concept_embedder.forward(h, [0, 1], c_train, train=True)
        y_pred = task_predictor(c_emb, c_pred)

        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        loss = concept_loss + 0.5*task_loss
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            h = encoder(x_test)
            c_emb, c_pred = concept_embedder.forward(h, [0, 1], c_test, train=False)
            y_pred = task_predictor(c_emb, c_pred)

            task_accuracy = accuracy_score(y_test, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
            print(f'Epoch {epoch}: loss {loss:.4f} task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')

    local_explanations = task_predictor.explain(c_emb, c_pred, 'local')
    global_explanations = task_predictor.explain(c_emb, c_pred, 'global')
    print(global_explanations)

    return model


class LukasiewiczTNorm:
    pass


class TestTemplateObject(unittest.TestCase):
    def test_deep_core(self):
        x, c, y = datasets.xor(1000)
        train_concept_bottleneck_model(x, c, y, embedding_size=16)

        x, c, y = datasets.trigonometry(1000)
        train_concept_bottleneck_model(x, c, y, embedding_size=16)
        #
        x, c, y = datasets.dot(1000)
        train_concept_bottleneck_model(x, c, y, embedding_size=16)

        return

    def test_semantics(self):
        x, c, y = datasets.xor(200)
        for logic in [GodelTNorm(), ProductTNorm()]:
            train_concept_bottleneck_model(x, c, y, embedding_size=16, logic=logic)

        return


if __name__ == '__main__':
    unittest.main()
