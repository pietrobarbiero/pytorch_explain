import numpy as np
import torch
from torch import nn
import torch_explain as te
from torch_explain.logic.parser import serialize_rules
from torch_explain.logic.semantics import VectorLogic
from torch_explain.nn.logic import PropositionalLayer
from torch_explain.logic.nn import entropy
import pytorch_lightning as pl
from torch.nn import functional as F


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


class RuleLearner(torch.nn.Module):
    def __init__(self, in_concepts, out_concepts, temperature=10):
        super().__init__()
        self.model = nn.Sequential(
            te.nn.EntropyLinear(in_concepts, 50, n_classes=out_concepts, temperature=temperature),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid(), # TODO: remove and work with logits?
        )

    def forward(self, x):
        return self.model.forward(x).squeeze(-1)

    def learn_rules(self, x, y, concepts_names, class_names):
        train_mask = torch.LongTensor(np.arange(x.shape[0]))
        test_mask = torch.LongTensor(train_mask)
        self.explanations_, _ = entropy.explain_classes(self.model.cpu(), x.cpu(), y.cpu(), train_mask.cpu(), test_mask.cpu(),
                                                        c_threshold=0.5, y_threshold=0.5, verbose=False,
                                                        concept_names=concepts_names, class_names=class_names,
                                                        material=True, good_bad_terms=False, max_accuracy=True)
        self.model.cuda()
        explanations = [{'explanation': v['explanation'], 'name': v['name']} for _, v in self.explanations_.items()]
        return serialize_rules(concepts=concepts_names, rules=explanations) # FIXME: save learnt rules to disk!


class DeepConceptReasoner(pl.LightningModule):
    def __init__(self, in_concepts, out_concepts, emb_size, concept_names, class_names,
                 learning_rate, learner_epochs, loss_form, mode='learner', temperature=10, verbose: bool = False):
        super().__init__()
        self.scorer = Scorer(emb_size)
        self.rule_learner = RuleLearner(in_concepts, out_concepts, temperature)
        self.logic = VectorLogic(emb_size)
        self.reasoner = PropositionalLayer()
        self.concept_names = concept_names
        self.class_names = class_names
        self.learner_epochs = learner_epochs
        self.learning_rate = learning_rate
        self.loss_form = loss_form
        self.mode = mode
        self.verbose = verbose
        self.learner_preds = []
        self.logic_preds = []
        self.neural_preds = []
        self.concept_preds = []
        self.task_labels = []
        self.concept_labels = []

    def forward(self, x):
        if self.mode == 'learner':
            x_pred = x
            y_pred_logic = y_pred_neural = y_emb = self.rule_learner(x)
        elif self.mode == 'reasoner':
            sc = self.scorer(x)
            nc = 1 - sc
            x_wmc = sc * self.logic.current_truth.T + nc * self.logic.current_false.T
            x_pred = self.logic.predict_proba(x_wmc)
            y_emb = self.reasoner(x_wmc, self.learnt_rules_, self.logic)
            y_pred_logic = self.logic.predict_proba(y_emb)
            y_pred_neural = self.scorer(y_emb).squeeze(-1) # TODO: change it and make it private
        else:
            raise NotImplementedError

        return y_pred_logic, y_pred_neural, y_emb, x_pred

    def training_step(self, batch, batch_idx):
        x_sem, x_emb, c, y = batch
        if self.mode == 'learner':
            y_pred_logic, _, _, _ = self.forward(x_sem)
            loss = self.loss_form(y_pred_logic, y)
            self.log("train_loss", loss)
        elif self.mode == 'reasoner':
            y_pred_logic, y_pred_neural, _, x_pred = self.forward(x_emb)
            logic_loss = self.loss_form(y_pred_logic, y)
            neural_loss = self.loss_form(y_pred_neural, y)
            concept_loss = self.loss_form(x_pred, c)
            loss = .1 * logic_loss + .1 * neural_loss + concept_loss
            self.log("train_loss", loss)
            self.log("logic_loss", logic_loss)
            self.log("neural_loss", neural_loss)
            self.log("concept_loss", concept_loss)
        else:
            raise NotImplementedError

        if self.current_epoch > self.learner_epochs:
            self.mode = 'reasoner'
            self.learnt_rules_ = self.rule_learner.learn_rules(x_sem, y, self.concept_names, self.class_names)
            self.cuda()

        return loss

    def test_step(self, batch, batch_idx):
        x_sem, x_emb, c, y = batch
        self.mode = 'learner'
        y_pred_learner, _, _, _ = self.forward(x_sem)
        self.mode = 'reasoner'
        y_pred_logic, y_pred_neural, _, x_pred = self.forward(x_emb)
        self.learner_preds.append(y_pred_learner)
        self.logic_preds.append(y_pred_logic)
        self.neural_preds.append(y_pred_neural)
        self.concept_preds.append(x_pred)
        self.task_labels.append(y)
        self.concept_labels.append(c)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)    # TODO: consider whether we need this or not
        return optimizer
