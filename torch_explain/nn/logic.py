import math
from collections import Counter

import torch
from torch import Tensor
from torch import nn

from ..logic.semantics import Logic
from ..logic.parser import ExpressionTree, Concept, Not, And, Or
from .concepts import Conceptizator


class DCR(torch.nn.Module):
    def __init__(self, in_features, n_hidden, emb_size, n_classes, logic: Logic, temperature=1.):
        super().__init__()
        self.in_features = in_features
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.temperature = temperature
        # self.w_value = torch.nn.Parameter(torch.empty((in_features, in_features)))
        self.w_value = torch.nn.Sequential(
            torch.nn.Linear(in_features, n_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_hidden, in_features),
        )
        self.w_key = torch.nn.Parameter(torch.empty((emb_size, emb_size)))
        self.w_query = torch.nn.Parameter(torch.empty((emb_size, n_classes)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # torch.nn.init.kaiming_uniform_(self.w_value, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_key, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_query, a=math.sqrt(5))

    def forward(self, x, c, return_attn=False):
        # values = c @ self.w_value
        values = self.w_value(c)
        values = values.unsqueeze(-1).repeat(1, 1, self.n_classes)

        # compute attention scores
        keys = x @ self.w_key
        attn = keys @ self.w_query
        # attn_scores_norm = torch.sigmoid(attn)  # or softmax
        attn_scores = torch.exp(attn / self.temperature) / torch.sum(torch.exp(attn / self.temperature), dim=1, keepdim=True)
        attn_scores_norm = attn_scores / attn_scores.max(dim=1)[0].unsqueeze(1)
        attn_mask = attn_scores_norm > 0.5

        # filter values
        neg_scores = self.logic.neg(attn_scores_norm)
        filtered_values = self.logic.disj_pair(values, neg_scores)

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze()
        if return_attn:
            return preds, attn_mask
        else:
            return preds

    def explain(self, x, c, mode):
        assert mode in ['local', 'global']

        y_preds, attn_mask = self.forward(x, c, return_attn=True)

        # extract local explanations
        predictions = y_preds.argmax(dim=-1).detach()
        explanations = []
        all_class_explanations = {c: [] for c in range(self.n_classes)}
        for concept_scores, attn, prediction in zip(c, attn_mask, predictions):
            # select mask for predicted class only
            # and generate minterm
            attn_filtered = attn[:, prediction]
            minterm = []
            for idx, (concept_score, attn_score) in enumerate(zip(concept_scores, attn_filtered)):
                if attn_score:
                    if concept_score > 0.5:
                        minterm.append(f'f{idx}')
                    else:
                        minterm.append(f'~f{idx}')
            minterm = ' & '.join(minterm)
            # add explanation to list
            all_class_explanations[prediction.item()].append(minterm)
            explanations.append({
                'class': prediction.item(),
                'explanation': minterm,
            })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations


class EntropyLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, temperature: float = 0.6,
                 bias: bool = True, conceptizator: str = 'identity_bool') -> None:
        super(EntropyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.conceptizator = Conceptizator(conceptizator)
        self.alpha = None
        self.weight = nn.Parameter(torch.Tensor(n_classes, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_classes, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        self.conceptizator.concepts = input
        # compute concept-awareness scores
        gamma = self.weight.norm(dim=1, p=1)
        self.alpha = torch.exp(gamma / self.temperature) / torch.sum(torch.exp(gamma / self.temperature), dim=1, keepdim=True)

        # weight the input concepts by awareness scores
        self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        self.concept_mask = self.alpha_norm > 0.5
        # if len(input.shape) == 3:
        #     x = input.unsqueeze(1).multiply(self.alpha_norm.unsqueeze(0).unsqueeze(-1))
        #     x = torch.einsum('btce,thc->bteh', x, self.weight)
        #     x = x.reshape(-1, x.shape[-1])
        # else:
        x = input.multiply(self.alpha_norm.unsqueeze(1))
        x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        x = x.permute(1, 0, 2)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}'.format(
            self.in_features, self.out_features, self.n_classes
        )


class PropositionalLayer(nn.Module):
    def __init__(self):
        super(PropositionalLayer, self).__init__()

    def forward(self, x, expression_tree: ExpressionTree, logic: Logic):
        # logic.update()    # TODO: check if we really need to train logic
        tasks = []
        for r in expression_tree.roots:
            tasks.append(self._visit(r, x, logic))
        return torch.concat(tasks, dim=1)

    def _visit(self, node, x, logic):
        # for each node in the expression tree either:
        # - return a concept (leaf) or
        # - perform logic composition of child nodes
        if isinstance(node, Concept):
            return x[:, node.id:node.id + 1]
        else:
            visited = []
            for c in node.children:
                x_viz = self._visit(c, x, logic)
                visited.append(x_viz)
            visited = torch.concat(visited, dim=1)
            if isinstance(node, Not):
                ops_result = logic.neg(visited)
            elif isinstance(node, And) and visited.shape[1] > 0:
                ops_result = logic.conj(visited)
            elif isinstance(node, Or) and visited.shape[1] > 0:
                ops_result = logic.disj(visited)
            else:
                raise NotImplementedError
            return ops_result


if __name__ == '__main__':
    data = torch.rand((10, 5))
    layer = EntropyLinear(5, 4, 2)
    out = layer(data)
    print(out.shape)
