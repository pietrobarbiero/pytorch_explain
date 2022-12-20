import numpy as np
import torch
from torch import nn
import torch_explain as te
import torch_explain.nn.logic
from torch_explain.logic.parser import serialize_rules
from torch_explain.logic.semantics import VectorLogic, ProductTNorm, GodelTNorm, SumProductSemiring
from torch_explain.nn.logic import PropositionalLayer
from torch_explain.logic.nn import entropy
import pytorch_lightning as pl
from torch.nn import functional as F
from collections import defaultdict
from itertools import product
from torch.nn import CrossEntropyLoss, BCELoss


class MNISTEncoder(nn.Module):
    def __init__(self):
        super(MNISTEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        return x



class MNISTClassifier(nn.Module):
    def __init__(self, output_features=10, with_softmax=True):
        super(MNISTClassifier, self).__init__()
        super().__init__()
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_features),
        )

    def forward(self, x):

        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x

class TruthEmbedder(torch.nn.Module):
    def __init__(self, input_size, emb_size):
        super().__init__()
        self.model = nn.Sequential(
            torch.nn.Linear(input_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, x):
        return self.model.forward(x)

class Scorer(torch.nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.model = nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model.forward(x)

class AdditionAdHocReasoner(nn.Module):


    def __init__(self, tree, logic):
        super().__init__()
        ors = []
        ands = []
        for or_ in tree.roots:
            ors.append([])
            for and_ in or_.children:
                ands.append([])
                for c in and_.children:
                    ands[-1].append(c.id)
                ors[-1].append(len(ands)-1)

        self.ors = [torch.tensor(o) for o in ors]
        self.ands = torch.tensor(ands)
        self.logic = logic



    def forward(self, x, exp, log):
        and_emb = x[:, self.ands].squeeze(-1)
        and_emb = self.logic.conj(and_emb, dim=2)

        ors_emb = []
        for o in self.ors:
            l = and_emb[:, o]
            l = self.logic.disj(l)
            ors_emb.append(l)
        ors_emb = torch.concat(ors_emb, dim=1)
        return ors_emb




class DeepConceptReasoner(pl.LightningModule):
    def __init__(self, emb_size, num_digits,concept_names, explanations, learning_rate,  temperature=10, verbose: bool = False, gpu=True):
        super().__init__()
        self.embedder = MNISTEncoder()
        self.classifier = MNISTClassifier()
        self.truth_embedder = TruthEmbedder(16 * 4 * 4 * num_digits, emb_size)
        self.logic = ProductTNorm()
        self.emb_size = emb_size

        self.vector_logic = VectorLogic(self.emb_size, gpu=gpu)
        self.scorer = Scorer(self.emb_size)

        # explanations = [{'explanation': v['explanation'], 'name': v['name']} for _, v in explanations.items()]
        # self.explanations = serialize_rules(concept_names, explanations)
        # self.reasoner = AdditionAdHocReasoner(self.explanations, self.logic)


        self.reasoner = PropositionalLayer()
        explanations = [{'explanation': v['explanation'], 'name': v['name']} for _, v in explanations.items()]
        self.explanations = serialize_rules(concept_names, explanations)

        self.concept_names = concept_names
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")



    def forward(self, X, y):


        embeddings = []
        classes = []
        for x in X:
            x = self.embedder(x)
            cl = self.classifier(x)
            embeddings.append(x)
            classes.append(cl)

        sc = torch.concat(classes, dim=-1).unsqueeze(-1)
        # nc = 1 - sc
        # x_wmc = sc * self.logic.current_truth.T + nc * self.logic.current_false.T
        x_wmc = sc
        y_truth = self.reasoner(x_wmc, self.explanations, self.logic)
        y_pred_logic = y_truth.squeeze()


        truth_emb = self.truth_embedder(torch.concat(embeddings, dim=-1))
        truth = self.vector_logic.get_truth_from_embeddings(truth_emb)
        false = self.vector_logic.get_false_from_truth(truth)
        y_emb = y_truth * truth.unsqueeze(1) + (1 - y_truth) * false.unsqueeze(1)
        y_pred_neural = self.scorer(y_emb).squeeze(-1)

        return y_pred_logic, y_pred_neural

    def training_step(self, I, batch_idx):
        X = I[:-1]
        y_one = I[-1]
        y = y_one.unsqueeze(-1)
        y_pred_logic, y_pred_neural= self.forward(X,y)
        out_logic = torch.gather(y_pred_logic, dim=1, index=y)
        loss_logic = self.bce(out_logic, torch.ones_like(out_logic))
        loss_neural = self.cross_entropy(y_pred_neural, y_one)
        loss = loss_logic + loss_neural
        pred_logic = torch.max(y_pred_logic, dim=-1)[1]
        pred_neural = torch.max(y_pred_neural, dim=-1)[1]
        accuracy_logic = (pred_logic == y.squeeze()).float().mean()
        accuracy_neural = (pred_neural == y.squeeze()).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_logic_accuracy", accuracy_logic, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_neural_accuracy", accuracy_neural, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, I, batch_idx):
        X = I[:-1]
        y = I[-1]
        y_pred_logic, y_pred_neural= self.forward(X,y)
        pred_logic = torch.max(y_pred_logic, dim=-1)[1]
        pred_neural = torch.max(y_pred_neural, dim=-1)[1]
        accuracy_logic = (pred_logic == y.squeeze()).float().mean()
        accuracy_neural = (pred_neural == y.squeeze()).float().mean()
        self.log("test_logic_accuracy", accuracy_logic, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_neural_accuracy", accuracy_neural, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


