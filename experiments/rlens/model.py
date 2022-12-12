import numpy as np
import torch
from torch import nn
import torch_explain as te
from torch_explain.nn.logic import R2NPropositionalLayer
from torch_explain.logic.nn import entropy
from torch.nn import functional as F


def get_rule_learner(in_concepts, out_concepts, temperature=10):
    return nn.Sequential(
        te.nn.EntropyLinear(in_concepts, 50, n_classes=out_concepts, temperature=temperature),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 20),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, 1),
    )


def get_reasoner(logic, scorer, loss_form):
    return R2NPropositionalLayer(logic, scorer, loss_form)


# class Scorer(nn.Module):
#     def __init__(self, emb_size):
#         super(Scorer, self).__init__()
#         self.w = torch.nn.Parameter(torch.randn(emb_size), requires_grad=True)
#
#     def forward(self, x):
#         # xt = torch.sigmoid(x)
#         xt = x.reshape(x.shape[0]*x.shape[1], -1)
#         xt = xt.matmul(self.w)
#         xt = xt.reshape(x.shape[0], x.shape[1])
#         return torch.sigmoid(xt) #torch.log(xt) - 1


def get_scorer(emb_size):
    return nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, 1),
            torch.nn.Sigmoid()
    )
    # return nn.Sequential(Scorer(emb_size))


class DeepConceptReasoner(torch.nn.Module):
    def __init__(self, rule_learner, reasoner, scorer, concept_names, class_names, verbose: bool = False):
        super().__init__()
        self.rule_learner = rule_learner
        self.reasoner = reasoner
        self.scorer = scorer
        self.concept_names = concept_names
        self.class_names = class_names
        self.verbose = verbose
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.Sigmoid(),
        )

    def learn_rules(self, x, y):
        train_mask = torch.LongTensor(np.arange(x.shape[0]))
        test_mask = torch.LongTensor(train_mask)
        self.explanations_, _ = entropy.explain_classes(self.rule_learner, x, y, train_mask, test_mask,
                                                        c_threshold=0.5, y_threshold=0., verbose=True,
                                                        concept_names=self.concept_names, class_names=self.class_names,
                                                        material=True, good_bad_terms=False, max_accuracy=True)
        return [{'explanation': v['explanation'], 'name': v['name']} for _, v in self.explanations_.items()]

    def forward(self, x, predictor: str, c=None):
        y_bool, y_pred_refined = None, None
        if predictor == 'learner':
            y_pred = self.rule_learner(x).squeeze(-1)
        elif predictor == 'ProbLog':
            raise NotImplementedError
        elif predictor == 'reasoner':
            y_emb, y_bool = self.reasoner(x, self.concept_names, self.learnt_rules_, c)
            y_pred = self.reasoner.predict_proba(y_emb).squeeze(-1)
            sc = self.lin(y_emb)
            nc = 1 - sc
            y_emb_refined = sc * self.reasoner.logic.current_truth.T + nc * self.reasoner.logic.current_false.T
            y_pred_refined = self.reasoner.predict_proba(y_emb_refined)
        else:
            raise NotImplementedError

        return y_pred, y_bool, y_pred_refined

    def fit(self, x_sem, x_emb, c, y, lr, epochs, use_learnt_rules=True):
        y_pred_reasoner, y_pred_learner, y_pred_refined = None, None, None

        # get learnt rules
        if use_learnt_rules:
            # train rule learner
            print('\nRule learning...')
            self.rule_optimizer = torch.optim.AdamW(self.rule_learner.parameters(), lr=lr)
            self.rule_learner.train()
            loss_form = torch.nn.BCEWithLogitsLoss()
            for epoch in range(epochs):
                self.rule_optimizer.zero_grad()
                y_pred_learner, _, _ = self.forward(x_sem, 'learner')
                loss = loss_form(y_pred_learner, y)
                loss.backward()
                self.rule_optimizer.step()

                # compute accuracy
                if epoch % 100 == 0 and self.verbose:
                    train_accuracy = (y_pred_learner > 0.).eq(y).sum().item() / (y.size(0) * y.size(1))
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {train_accuracy:.4f}')

            self.learnt_rules_ = self.learn_rules(x_sem, y)

        # train reasoner
        print('\nReasoning...')
        params = list(self.lin.parameters()) + list(self.scorer.parameters())
        # self.reasoner_optimizer = torch.optim.AdamW(self.reasoner.logic.parameters(), lr=lr)
        self.reasoner_optimizer = torch.optim.AdamW(params, lr=lr)
        # self.reasoner.train()
        self.lin.train()
        self.scorer.train()
        loss_form = torch.nn.BCELoss()
        for epoch in range(epochs):
            self.reasoner_optimizer.zero_grad()

            # a = self.scorer(x_emb)
            # b = 1 - a
            #
            # truth = self.lin(x_emb.reshape(x_emb.shape[0], -1))
            # false = self.reasoner.logic.false(truth)
            #
            # x_wmc = a * truth.unsqueeze(1) + b * false.unsqueeze(1)
            # self.reasoner.logic.update(truth, false)

            sc = self.scorer(x_emb)
            nc = 1 - sc
            # sc = x_emb1.matmul(self.reasoner.logic.current_truth)**2
            # nc = x_emb1.matmul(self.reasoner.logic.current_false)**2
            x_wmc = sc * self.reasoner.logic.current_truth.T + nc * self.reasoner.logic.current_false.T
            # sc1 = x_wmc.matmul(self.reasoner.logic.current_truth)
            # nc1 = x_wmc.matmul(self.reasoner.logic.current_false)

            c_pred_reasoner = self.reasoner.predict_proba(x_wmc)
            y_pred_reasoner, y_pred_bool, y_pred_refined = self.forward(x_wmc, 'reasoner', (c > 0.5).float())
            loss1 = loss_form(y_pred_reasoner, y)
            loss3 = loss_form(y_pred_refined, y)
            loss2 = loss_form(c_pred_reasoner, (c > 0.5).float())
            loss = 0.1 * loss1 + 0.1 * loss3 + loss2 #+ self.reasoner.loss_
            loss.backward(retain_graph=True)
            self.reasoner_optimizer.step()

            # compute accuracy
            if epoch % 100 == 0 and self.verbose:
                train_accuracy = (y_pred_reasoner > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
                refined_accuracy = (y_pred_refined > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
                bool_accuracy = (y_pred_bool > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
                c_accuracy = (c_pred_reasoner > 0.5).eq(c > 0.5).sum().item() / (c.size(0) * c.size(1))
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {train_accuracy:.4f} refined accuracy: {refined_accuracy:.4f} concept accuracy: {c_accuracy:.4f} (train bool: {bool_accuracy:.4f})')

        return y_pred_reasoner, y_pred_learner, y_pred_refined

    def predict(self, x_sem, x_emb):
        # inference with rule learner
        print('\nRule learner inference...')
        self.rule_learner.eval()
        y_pred_learner, _, _ = self.forward(x_sem, 'learner')

        # update learnt rules?
        # OOD tasks?
        # self.learnt_rules_ = self.update_rules()

        # inference with reasoner
        print('Reasoning inference...')
        # self.reasoner.eval()
        # self.scorer.eval()
        self.lin.train()
        self.scorer.train()

        # x_emb1 = self.scorer(x_emb)

        # sc = x_emb1.matmul(self.reasoner.logic.current_truth) ** 2
        # nc = x_emb1.matmul(self.reasoner.logic.current_false) ** 2
        # sumc = sc + nc
        # x_wmc = (sc * self.reasoner.logic.current_truth.T +
        #          nc * self.reasoner.logic.current_false.T) / sumc

        sc = self.scorer(x_emb)
        nc = 1 - sc
        # sc = x_emb1.matmul(self.reasoner.logic.current_truth)**2
        # nc = x_emb1.matmul(self.reasoner.logic.current_false)**2
        x_wmc = sc * self.reasoner.logic.current_truth.T + nc * self.reasoner.logic.current_false.T
        # sc1 = x_wmc.matmul(self.reasoner.logic.current_truth)
        # nc1 = x_wmc.matmul(self.reasoner.logic.current_false)

        c_pred_reasoner = self.reasoner.predict_proba(x_wmc)
        y_pred_reasoner, y_pred_bool, y_pred_refined = self.forward(x_wmc, 'reasoner', (x_sem > 0.5).float())

        return y_pred_reasoner, y_pred_learner, c_pred_reasoner, y_pred_bool, y_pred_refined
