import numpy as np
import torch
from torch import nn
import torch_explain as te
from torch_explain.nn.logic import R2NPropositionalLayer
from torch_explain.logic.nn import entropy


def get_rule_learner(in_concepts, out_concepts, temperature=10):
    return nn.Sequential(
        te.nn.EntropyLinear(in_concepts, 50, n_classes=out_concepts, temperature=temperature),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 20),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, 1),
    )


def get_reasoner():
    return R2NPropositionalLayer()


def get_scorer(emb_size):
    return nn.Sequential(torch.nn.Linear(emb_size, 1))


class DeepConceptReasoner(torch.nn.Module):
    def __init__(self, rule_learner, reasoner, scorer, concept_names, class_names, lr, epochs, verbose: bool = False):
        super().__init__()
        self.rule_learner = rule_learner
        self.reasoner = reasoner
        self.scorer = scorer
        self.concept_names = concept_names
        self.class_names = class_names
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

    def learn_rules(self, x, y):
        train_mask = torch.LongTensor(np.arange(x.shape[0]))
        test_mask = torch.LongTensor(train_mask)
        self.explanations_, _ = entropy.explain_classes(self.rule_learner, x, y, train_mask, test_mask,
                                                        c_threshold=0.5, y_threshold=0., verbose=True,
                                                        concept_names=self.concept_names, class_names=self.class_names,
                                                        material=True, good_bad_terms=False, max_accuracy=True)
        return [{'explanation': v['explanation'], 'name': v['name']} for _, v in self.explanations_.items()]

    def forward(self, x, predictor: str):
        if predictor == 'learner':
            y_pred = self.rule_learner(x).squeeze(-1)
        elif predictor == 'ProbLog':
            raise NotImplementedError
        elif predictor == 'reasoner':
            x_wmc = torch.sigmoid(x)
            y_emb = self.reasoner(x_wmc, self.concept_names, self.learnt_rules_)
            y_pred = self.scorer(y_emb).squeeze(-1)
        else:
            raise NotImplementedError

        return y_pred

    def fit(self, x_sem, x_emb, y):
        y_pred_reasoner, y_pred_learner = None, None

        # train rule learner
        print('\nRule learning...')
        self.rule_optimizer = torch.optim.AdamW(self.rule_learner.parameters(), lr=self.lr)
        self.rule_learner.train()
        loss_form = torch.nn.BCEWithLogitsLoss()
        for epoch in range(self.epochs):
            self.rule_optimizer.zero_grad()
            y_pred_learner = self.forward(x_sem, 'learner')
            loss = loss_form(y_pred_learner, y)
            loss.backward()
            self.rule_optimizer.step()

            # compute accuracy
            if epoch % 100 == 0 and self.verbose:
                train_accuracy = (y_pred_learner > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {train_accuracy:.4f}')

        # get learnt rules
        self.learnt_rules_ = self.learn_rules(x_sem, y)

        # train reasoner
        print('\nReasoning...')
        params = list(self.reasoner.parameters()) + list(self.scorer.parameters())
        self.reasoner_optimizer = torch.optim.AdamW(params, lr=self.lr)
        self.reasoner.train()
        self.scorer.train()
        loss_form = torch.nn.BCEWithLogitsLoss()
        for epoch in range(self.epochs):
            self.reasoner_optimizer.zero_grad()
            y_pred_reasoner = self.forward(x_emb, 'reasoner')
            loss = loss_form(y_pred_reasoner, y)
            loss.backward()
            self.reasoner_optimizer.step()

            # compute accuracy
            if epoch % 100 == 0 and self.verbose:
                train_accuracy = (y_pred_reasoner > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {train_accuracy:.4f}')

        return y_pred_reasoner, y_pred_learner

    def predict(self, x_sem, x_emb, y):
        # inference with rule learner
        print('\nRule learner inference...')
        self.rule_learner.eval()
        y_pred_learner = self.forward(x_sem, 'learner')

        # update learnt rules?
        # OOD tasks?
        # self.learnt_rules_ = self.update_rules()

        # inference with reasoner
        print('Reasoning inference...')
        self.reasoner.eval()
        self.scorer.eval()
        y_pred_reasoner = self.forward(x_emb, 'reasoner')

        return y_pred_reasoner, y_pred_learner
