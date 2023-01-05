import copy
import math
from collections import Counter
import torch
from torch import Tensor
from torch import nn

from ..logic.semantics import Logic
from ..logic.parser import ExpressionTree, Concept, Not, And, Or
from .concepts import Conceptizator

EPS = 1e-3


def softmaxnorm(values, temperature):
    softmax_scores = torch.exp(values / temperature) / torch.sum(torch.exp(values / temperature), dim=1, keepdim=True)
    return softmax_scores / softmax_scores.max(dim=1)[0].unsqueeze(1)


class DCR(torch.nn.Module):
    def __init__(self, in_features, emb_size, n_classes, logic: Logic,
                 temperature_complexity: float = 1., temperature_sharp: float = 1.):
        super().__init__()
        self.in_features = in_features
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.temperature_sharp = temperature_sharp
        self.temperature_complexity = temperature_complexity
        self.w_key_logic = torch.nn.Parameter(torch.empty((emb_size, emb_size)))
        self.w_query_logic = torch.nn.Parameter(torch.empty((emb_size, n_classes)))
        self.w_key_filter = torch.nn.Parameter(torch.empty((emb_size, emb_size)))
        self.w_query_filter = torch.nn.Parameter(torch.empty((emb_size, n_classes)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # torch.nn.init.kaiming_uniform_(self.w_value, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_key_logic, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_query_logic, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_key_filter, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_query_filter, a=math.sqrt(5))

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)
        # x = c.unsqueeze(-1).repeat(1, 1, self.emb_size)

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            logic_keys = x @ self.w_key_logic   # TODO: might be independent of input x (but requires OR)
            sign_attn = torch.sigmoid(logic_keys @ self.w_query_logic)

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)   # TODO: Fra check
        sign_terms = self.logic.iff_pair(sign_attn, values)    # TODO: temperature sharp here?
        # control sharpness of truth values
        # sharper values -> lower leakage, lower accuracy
        # less sharp values -> higher leakage, higher accuracy
        # sign_terms = torch.sigmoid(self.temperature_sharp * (sign_terms - 0.5))

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filter_keys = x @ self.w_key_filter   # TODO: might be independent of input x (but requires OR)
            filter_attn = softmaxnorm(filter_keys @ self.w_query_filter, self.temperature_complexity)

        # filter values
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))

        # generate minterm
        # preds = self.logic.conj(filtered_values, dim=1).squeeze().float()
        preds = self.logic.conj(filtered_values, dim=1).softmax(dim=-1).squeeze()   # FIXME: softmax looks weird

        # TODO: add OR for global explanations

        if return_attn:
            return preds, sign_attn, filter_attn   # FIXME: handle None cases
        else:
            return preds

    def explain(self, x, c, mode):
        assert mode in ['local', 'global']

        y_preds, sign_attn_mask, filter_attn_mask = self.forward(x, c, return_attn=True)
        sign_attn_mask, filter_attn_mask = sign_attn_mask > 0.5, filter_attn_mask > 0.5

        # extract local explanations
        predictions = y_preds.argmax(dim=-1).detach()
        explanations = []
        all_class_explanations = {c: [] for c in range(self.n_classes)}
        for filter_attn, sign_attn, prediction in zip(filter_attn_mask, sign_attn_mask, predictions):
            # select mask for predicted class only
            # and generate minterm
            minterm = []
            for idx, (concept_score, attn_score) in enumerate(zip(sign_attn[:, prediction], filter_attn[:, prediction])):
                if attn_score:
                    if concept_score:
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

    def counterfact(self, x, c):
        # find original predictions and assume we just want to flip them all
        old_preds = self.forward(x, c)

        # find a (random) counterfactual: a (random) perturbation of the input that would change the prediction
        counterfactuals = {'sample_id': [], 'old_pred': [], 'new_pred': [], 'old_concepts': [], 'new_concepts': []}
        for sid, (old_concept_emb, old_concept_score, old_pred) in enumerate(zip(x, c, old_preds)):
            old_concept_emb, old_concept_score = old_concept_emb.unsqueeze(0), old_concept_score.unsqueeze(0)
            new_concept_score = copy.deepcopy(old_concept_score)
            new_concept_emb = copy.deepcopy(old_concept_emb)
            target_pred = (1 - old_pred).argmax(dim=-1)

            # select a random sequence of concepts to perturb
            rnd_concept_idxs = torch.randperm(self.in_features)
            for rnd_concept_idx in rnd_concept_idxs:
                # perturb concept score
                new_concept_score[:, rnd_concept_idx] = 1 - old_concept_score[:, rnd_concept_idx]
                # perturb concept embedding
                new_concept_emb[:, rnd_concept_idx] = torch.mean(x[old_preds.argmax(dim=-1)==target_pred, rnd_concept_idx], dim=0)
                # TODO: rule intervention? update attention weights according to new concept scores
                # new_sign_attn = (1 - new_concept_score).unsqueeze(-1).repeat(1, 1, self.n_classes)
                # new_sign_attn[:, :, target_pred] = new_concept_score
                # generate new prediction
                new_pred = self.forward(new_concept_emb, new_concept_score) # TODO: we may start with original attn and then make all concepts available
                if new_pred.argmax(dim=-1) == target_pred:
                    counterfactuals['sample_id'].append(sid)
                    counterfactuals['old_pred'].append(old_pred.tolist())
                    counterfactuals['new_pred'].append(new_pred.tolist())
                    counterfactuals['old_concepts'].append(old_concept_score.tolist()[0])
                    counterfactuals['new_concepts'].append(new_concept_score.tolist()[0])
                    break
        return counterfactuals


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
