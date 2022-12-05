import abc
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def serialize_len_rules(concepts, rules):
    concept_name_to_object = {}
    for i, c in enumerate(concepts):
        concept_name_to_object[c] = Concept(name=c, id=i)

    roots = []
    for rule in rules:
        task = rule["name"]
        ands = []
        for and_ in rule["explanation"].split("|"):
            and_ = and_.replace("(","").replace(")", "")
            literals = []
            for literal in and_.split("&"):
                literal = literal.strip()
                if "~" in literal:
                    l = Not([concept_name_to_object[literal.replace("~", "")]])
                else:
                    l = concept_name_to_object[literal]
                literals.append(l)
            ands.append(And(children=literals))
        roots.append(Or(ands, name = task))

    tree = ExpressionTree(roots=roots)
    return tree


class TreeNode():

    def __init__(self, name=None):
        self.name = name

class Operator(TreeNode):
    def __init__(self, children, name=None):
        super().__init__(name)
        self.children = children




class Or(Operator):

    def __str__(self):
        return " | ".join([str(c) for c in self.children])

class And(Operator):

    def __str__(self):
        return "(" + " & ".join([str(c) for c in self.children]) + ")"

class Not(Operator):
    def __init__(self, children):
        super(Not, self).__init__(children)
        assert len(self.children) == 1

    def __str__(self):
        return "~" + str(self.children[0])

class Concept(TreeNode):

    def __init__(self, name, id):
        super().__init__(name)
        self.id = id

    def __str__(self):
        return self.name

class ExpressionTree():

    def __init__(self, roots):
        self.roots = roots


class Logic():


    @abc.abstractmethod
    def conj(self, a):
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, a):
        raise NotImplementedError

    @abc.abstractmethod
    def neg(self, a):
        raise NotImplementedError

class ProductTNorm(Logic):


    def conj(self, a):
        return torch.prod(a,dim=1, keepdim=True)

    def disj(self, a):
        return 1 - torch.prod(1 - a, dim=1, keepdim=True)

    def neg(self, a):
        return 1 - a


class R2NPropositionalLayer(nn.Module):

    def __init__(self, expression_tree:ExpressionTree, logic = ProductTNorm()):
        super(R2NPropositionalLayer, self).__init__()
        self.logic = logic
        self.tree = expression_tree



    def forward(self, x):

        tasks = []
        for r in self.tree.roots:
            tasks.append(self._visit(r, x))
        return torch.concat(tasks, dim=1)




    def _visit(self, node, x):

        if isinstance(node, Concept):
            return x[:,node.id:node.id+1]
        else:
            visited = []
            for c in node.children:
                visited.append(self._visit(c, x))
            visited = torch.concat(visited, dim=1)
            if isinstance(node, Not):
                return self.logic.neg(visited)
            elif isinstance(node, And):
                return self.logic.conj(visited)
            elif isinstance(node, Or):
                return self.logic.disj(visited)
            else:
                raise Exception("Node class not known." %  node.__type__)



def main():
    datasets = ['xor', 'trig', 'vec']
    # datasets = ['trig']
    folds = [i + 1 for i in range(5)]
    epoch = 500
    for dataset in datasets:
        results = pd.DataFrame()
        explanations = pd.read_csv(f'./results/{dataset}_activations_final_rerun/explanations/explanations.csv', index_col=None)
        for fold in folds:
            c_test = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_vectors_on_epoch_{epoch}.npy')
            y_cem = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_model_output_on_epoch_{epoch}.npy')
            # c1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_test.npy')
            # c2 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_val.npy')
            # y1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/y_test.npy')
            y = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/y_val.npy')
            concepts = [f'x{i}' for i in range(c_test.shape[1])]

            explanation_list = [{'explanation': e['explanation'], 'name': e['name']} for _, e in explanations[explanations['split']==fold].iterrows()]

            # Parse the rules into a forest (one tree for each task)
            tree = serialize_len_rules(concepts=concepts, rules=explanation_list)

            # Instantiate a reasoner on the tree, given a semiring
            reasoner = R2NPropositionalLayer(expression_tree=tree, logic=ProductTNorm())

            y = F.one_hot(torch.LongTensor(y)).float()
            c_test = torch.FloatTensor(c_test)
            c_test = torch.sigmoid(c_test)
            c_test = c_test.reshape(c_test.shape[0], y.shape[1], -1)
            predictions = reasoner(c_test)

            model = torch.nn.Sequential(torch.nn.Linear(c_test.shape[2], 50), torch.nn.Linear(50, 1))
            model = torch.nn.Sequential(torch.nn.Linear(c_test.shape[2], 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
            loss_form = torch.nn.BCEWithLogitsLoss()
            model.train()

            train_mask = set(np.random.choice(np.arange(predictions.shape[0]), int(predictions.shape[0] * 0.8), replace=False))
            test_mask = set(np.arange(predictions.shape[0])) - train_mask
            train_mask = torch.LongTensor(list(train_mask))
            test_mask = torch.LongTensor(list(test_mask))
            for epoch_train in range(10001):
                # train step
                optimizer.zero_grad()
                y_pred = model(predictions).squeeze(-1)
                loss = loss_form(y_pred[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch_train % 100 == 0:
                    train_accuracy = (y_pred[train_mask] > 0.).eq(y[train_mask]).sum().item() / (
                                y[train_mask].size(0) * y[train_mask].size(1))
                    test_accuracy = (y_pred[test_mask] > 0.).eq(y[test_mask]).sum().item() / (
                                y[test_mask].size(0) * y[test_mask].size(1))
                    print(f'Epoch {epoch_train}: loss {loss:.4f} train accuracy: {train_accuracy:.4f} test accuracy: {test_accuracy:.4f}')

            cem_accuracy = (torch.FloatTensor(y_cem).squeeze() > 0.).eq(y[:, 1]).sum().item() / len(y)

            res_dir = f'./results/{dataset}_activations_final_rerun/explanations/'
            os.makedirs(res_dir, exist_ok=True)
            out_file = os.path.join(res_dir, 'reasoner_results.csv')
            res1 = pd.DataFrame([[explanation_list, test_accuracy, fold, 'R2N']], columns=['rules', 'accuracy', 'fold', 'model'])
            res2 = pd.DataFrame([[explanation_list, cem_accuracy, fold, 'CEM']], columns=['rules', 'accuracy', 'fold', 'model'])

            if len(results) == 0:
                results = res1
                results = pd.concat((results, res2))

            results = pd.concat((results, res1, res2))

            results.to_csv(out_file)




if __name__ == '__main__':


    # Output from CEM (logic to test)
    # concepts = ["a", "b", "c", "d", "e", "f"]
    # a = torch.tensor([[1], [0]])
    # b = torch.tensor([[1], [0]])
    # c = torch.tensor([[0], [1]])
    # d = torch.tensor([[0], [0]])
    # e = torch.tensor([[1], [1]])
    # f = torch.tensor([[1], [0]])
    # concepts_embeddings = torch.stack((a,b,c,d,e,f), dim=-2)

    # Output from CEM (embedding like)
    batch_size  = 20
    embedding_size = 32
    concepts = ["a", "b", "c", "d", "e", "f"]
    a = torch.rand([batch_size, embedding_size])
    b = torch.rand([batch_size, embedding_size])
    c = torch.rand([batch_size, embedding_size])
    d = torch.rand([batch_size, embedding_size])
    e = torch.rand([batch_size, embedding_size])
    f = torch.rand([batch_size, embedding_size])
    concepts_embeddings = torch.stack((a,b,c,d,e,f), dim=-2)

    # Output from LEN
    explanations = [{'explanation': '(a & b) | (c & d)', 'name': 'y'},
                    {'explanation': '(a & b) | (e & f)', 'name': 'z'}
                    ]



    # Parse the rules into a forest (one tree for each task)
    tree = serialize_len_rules(concepts=concepts, rules=explanations)


    # Instantiate a reasoner on the tree, given a semiring
    reasoner = R2NPropositionalLayer(expression_tree=tree, logic=ProductTNorm())

    # Use the semiring for computing the tasks
    predictions = reasoner(concepts_embeddings)
    print(predictions.shape)

    main()
