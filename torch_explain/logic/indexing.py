import copy
from typing import List, Dict, Tuple
import torch
from torch import Tensor
from torch_explain.logic.commons import Rule

def tuple_to_indexes(t, atom_index, query_index, relation_index):
    # TODO: add documentation
    if t in query_index:
        atom_index[t] = query_index[t]
    else:
        query_index[t] = len(query_index)
        atom_index[t] = query_index[t]
    if t[0] not in relation_index:
        relation_index[t[0]] = len(relation_index)
    return [(atom_index[t], relation_index[t[0]], int(i), j) for j, i in enumerate(t[1:])]


def query_str_to_tuple(query_str: str) -> Tuple:
    query_name = query_str.split('(')[0]
    query_input_ids = query_str.split('(')[1][:-1]
    # query_tuple = tuple([query_name] + [int(i) for i in query_input_ids.split(',')])
    query_tuple = tuple([query_name] + [i for i in query_input_ids.split(',')])
    return query_tuple


# def tuple_str_to_int(t: Tuple) -> Tuple:
#     return tuple(t[0]) + tuple([int(i) for i in t[1:]])


def sort_index(index: torch.Tensor) -> torch.Tensor:
    # TODO: check whether we need extra arguments to control the columns to sort
    # We need them to be sorted first, by relation (column=1) and, then, by position (column=3)
    index = index[torch.argsort(index[:, 3])]
    index = index[torch.argsort(index[:, 1], stable=True)]
    return index

    # _, indices = torch.sort(t[:, dim])
    # t = t[indices]
    # ids = t[:, dim].unique()
    # mask = t[:, None, dim] == ids
    # splits = torch.argmax(mask.float(), dim=0)
    # r = torch.tensor_split(t, splits[1:])
    # return r

def group_by_no_for(groupby_values, tensor_to_group=None, dim=None):
    # TODO: add documentation
    if tensor_to_group is None:
        tensor_to_group = groupby_values
    if dim is not None:
        _, sorted_groupby_indices = torch.sort(groupby_values[:, dim])
    else:
        _, sorted_groupby_indices = torch.sort(groupby_values)
    sorted_groupby_values = groupby_values[sorted_groupby_indices]

    if dim is not None:
        split_group = sorted_groupby_values
        unique_groupby_values = sorted_groupby_values[:, dim].unique()
        mask_for_split = sorted_groupby_values[:, None, dim] == unique_groupby_values
    else:
        split_group = tensor_to_group[sorted_groupby_indices]
        unique_groupby_values = groupby_values.unique()
        mask_for_split = sorted_groupby_values[:, None] == unique_groupby_values

    splits = torch.argmax(mask_for_split.float(), dim=0)
    return torch.tensor_split(split_group, splits[1:])


def intersect_1d_no_loop(a: torch.tensor, b: torch.tensor):
    return a[(a.view(1, -1) == b.view(-1, 1)).any(dim=0)]


class Indexer:

    def __init__(self, groundings: Dict[str, List[Tuple[Tuple, Tuple]]], queries: List[str]):
        # TODO: add documentation with simple example of how to use the class

        self.groundings = groundings
        self.queries = queries
        # dictionary of unique grounded queries: {query_tuple: index}
        # a query tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n')
        self.unique_query_index = {}

        # dictionary containing the index of all the unique grounded atoms (query and not!): {relation_tuple: index}
        # a relation tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n')
        # the dictionary is initialized with the unique grounded queries
        # the dictionary is updated with the other grounded atoms found in grounded rules
        self.grounded_relation_index = {}

        # dictionary containing the index of unique grounded atoms found in the rules: {relation_tuple: index}
        # a relation tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n')
        # the dictionary may not contain all the queries (it contains only queries found in the rules)
        self.unique_atom_index = {}

        # dictionary containing the index of all relation names: {'relation_name': index}
        # the dictionary is updated while looping over the grounded rules
        self.relation_index = {}

        # dictionary containing the index of all the unique bodies found in grounded rules: {body_tuple: index}
        # a body tuple is a tuple of the form: (('relation_name', 'input_id_1', ..., 'input_id_n'), ...)
        # the dictionary is updated while looping over the grounded rules
        self.bodies_index = {}

        # dictionary containing the index of all the unique grounded rules: {'formula_name': index}
        # the dictionary is updated while looping over the grounded rules
        self.formulas_index = {}

        self.indices = {}
        self.indices_groups = {}
        self.supervised_queries_ids = None

    def index_all(self) -> Tuple[Dict[str, Tensor], Dict[str, List[Tensor]]]:
        # TODO: add documentation
        init_index_queries = torch.tensor(self.init_index_queries())
        index_atoms = sort_index(torch.tensor(self.index_atoms()))
        index_formulas, dict_formula_tuples  = self.index_formulas()
        index_formulas = sort_index(torch.tensor(index_formulas))
        self.indices = {
            'queries': init_index_queries,  # [query_id_1, ..., query_id_n]
            'atoms': index_atoms,  # [(atom_index, relation_index, input_id, position), ...]
            'formulas': index_formulas,# [(body_index, formula_index, grounded_relation_index, position, head_index), ...]
            "substitutions": {k:torch.tensor(v) for k,v in dict_formula_tuples.items()} #dict[formula_id, List[Tuple[costant_id]]]
        }
        self.indices_groups = {
            'atoms': group_by_no_for(self.indices['atoms'], dim=1),
            'formulas': group_by_no_for(self.indices['formulas'], dim=1),
        }
        self.supervised_queries_ids = torch.tensor(list(self.unique_query_index.values()))
        return self.indices, self.indices_groups,

    def get_supervised_slice(self, y, y_ids):
        supervised_y_ids = intersect_1d_no_loop(y_ids, self.supervised_queries_ids)
        return y[supervised_y_ids]

    def apply_index(self, X, index_name, group_id):
        # TODO: add documentation
        # rel_id = index[0, 1]
        tuples = group_by_no_for(self.indices_groups[index_name][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        if tuples.shape[-1] > 4:
            atom_ids = tuples[:, 0, -1]
        else:
            atom_ids = tuples[:, 0, 0]
        tuples = tuples[:, :, 2]

        return X[tuples].view(tuples.shape[0], -1), tuples, atom_ids

    def apply_index_atoms(self, X, group_id):
        # TODO: add documentation
        # rel_id = index[0, 1]
        tuples = group_by_no_for(self.indices_groups["atoms"][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        atom_ids = tuples[:, 0, 0]
        tuples = tuples[:, :, 2]

        return X[tuples].view(tuples.shape[0], -1), tuples, atom_ids

    def apply_index_formulas(self, constant_embedddings, atom_predictions, formula_id):

        group_id = self.formulas_index[formula_id]
        # TODO: add documentation
        # rel_id = index[0, 1]
        tuples = group_by_no_for(self.indices_groups["formulas"][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        # formulas_ids = tuples[:, 0, -1]
        tuples = tuples[:, :, 2]
        substitutions = self.indices["substitutions"][group_id]

        return constant_embedddings[substitutions].view(substitutions.shape[0], -1), atom_predictions[tuples].view(tuples.shape[0], -1) #, tuples, atom_ids, substitutions

    def group_or(self, grounding_predictions, formula_id):

        group_id = self.formulas_index[formula_id]
        tuples = group_by_no_for(self.indices_groups["formulas"][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        head_ids = tuples[:, 0, -1]
        y_preds_group = group_by_no_for(groupby_values=head_ids, tensor_to_group=grounding_predictions)
        y_preds_group = torch.stack(y_preds_group, dim=0)
        y_preds_mean = y_preds_group.mean(dim=1)

        return constant_embedddings[substitutions].view(substitutions.shape[0], -1), atom_predictions[tuples].view(
            tuples.shape[0], -1)  # , tuples, atom_ids, substitutions

    def init_index_queries(self) -> List[Tuple[int, int, int, int]]:
        """
        Initialize the index of unique grounded queries (a.k.a. the concepts/relations to be supervised).
        This function loops over the input queries of the form: ['relation_name(input_id_1, ..., input_id_n)', ...]
        and creates a dictionary of unique grounded queries: {query_tuple: index}
        where query_tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n').
        :return: index of unique grounded queries
        """
        indices_queries = []
        for q in self.queries:
            query_tuple = query_str_to_tuple(q)

            # append query to unique_tuples if not already present
            if query_tuple not in self.unique_query_index:
                self.unique_query_index[query_tuple] = len(self.unique_query_index)

            # append query ID to indices_queries
            indices_queries.append(self.unique_query_index[query_tuple])

        return indices_queries

    def index_atoms(self) -> List[Tuple[int, int, int, int]]:
        """
        Index all the grounded atoms found in the bodies/heads of grounded rules.

        This function loops over the grounded rules of the form: {'formula_name': [((head_tuple_1, ...), (body_tuple_1, ...)), ...], ...}.
        where each tuple is of the form: ('relation_name', 'input_id_1', ..., 'input_id_n').

        This function creates an index of all the grounded atoms of the form: (atom_index, relation_index, input_id, position).
        The atom_index is given by self.unique_atom_index.
        The relation_index is given by self.relation_index.
        The input_id is given by int(input_id) in the tuple.
        The position is given by the position of the input_id in the tuple.
        :return: index of unique grounded atoms
        """
        # initialize the index of unique grounded atoms with the unique grounded queries
        # this way the grounded_relation_index indeces are aligned with the unique_query_index indeces
        self.grounded_relation_index = copy.deepcopy(self.unique_query_index)

        # loop over all atoms found in heads/bodies of grounded rules
        indices_atoms = []
        for k, (groundings, _) in self.groundings.items():
            for head, body in groundings:

                for h_tuple in head:
                    # skip if the current atom was already indexed
                    # otherwise generate index of the form: (atom_index, relation_index, input_id, position)
                    # and append index to indices_atoms
                    if h_tuple not in self.unique_atom_index:
                        # TODO: check whether we actually need all these indexes
                        idx = tuple_to_indexes(h_tuple, self.unique_atom_index, self.grounded_relation_index, self.relation_index)
                        indices_atoms.extend(idx)

                for b_tuple in body:
                    # skip if the current atom was already indexed
                    # otherwise generate index of the form: (atom_index, relation_index, input_id, position)
                    # and append index to indices_atoms
                    if b_tuple not in self.unique_atom_index:
                        idx = tuple_to_indexes(b_tuple, self.unique_atom_index, self.grounded_relation_index, self.relation_index)
                        indices_atoms.extend(idx)

        return indices_atoms

    def index_formulas(self) -> List[Tuple[int, int, int, int, int]]:
        """
        Index all the grounded formulas.

        This function loops over the grounded rules of the form: {'formula_name': [((head_tuple_1, ...), (body_tuple_1, ...)), ...], ...}.
        where each tuple is of the form: ('relation_name', 'input_id_1', ..., 'input_id_n').

        This function creates an index of all the grounded formulas of the form: (body_index, formula_index, grounded_relation_index, position, head_index).
        The body_index is given by self.bodies_index.
        The formula_index is given by self.formulas_index.
        The grounded_relation_index is given by self.grounded_relation_index.
        The position is given by the position of the body_tuple in the formula.
        The head_index is given by self.unique_query_index.

        :return: index of unique grounded formulas
        """
        # loop over all formulas
        indices_formulas = []
        indices_tuples_formulas = {}
        indices_formulas_dict = {}
        for k, (groundings, substitutions) in self.groundings.items():
            substitutions = [[int(l) for l in k] for k in substitutions]
            # add {'formula_name', index} to the dictionary formulas_index if not already present
            if k not in self.formulas_index:
                # TODO: check what happens with more than 1 formula in the dataset
                self.formulas_index[k] = len(self.formulas_index)

            indices_formulas_dict = {self.formulas_index[k]: []}


            # loop over all grounded rules of the form: ((head_tuple), (body_tuple_1, body_tuple_2, ...))
            for y, (head, body) in enumerate(groundings):
                indices_formulas_dict[self.formulas_index[k]].append([])


                # get head index (and check that there is only one head)
                assert len(head) == 1  # TODO: check what happens with more than 1 head
                index_head = self.unique_query_index[head[0]]

                # add {'body': index} to dictionary bodies_index if not already present
                if body not in self.bodies_index:
                    self.bodies_index[body] = len(self.bodies_index)

                # loop over all atoms in the body and generate index of the form: (body_index, formula_index, grounded_relation_index, position, head_index)
                for pos, b in enumerate(body):
                    # TODO: check whether to use grounded_relation_index or unique_atom_index or unique_query_index
                    indices_formulas.append((self.bodies_index[body], self.formulas_index[k], self.grounded_relation_index[b], pos, index_head))
                    indices_formulas_dict[self.formulas_index[k]][-1].append(self.grounded_relation_index[b])
            indices_tuples_formulas[self.formulas_index[k]] = substitutions
        return indices_formulas, indices_tuples_formulas

    def invert_index_relation(self, tuple_relation: Tuple[int, int, int, int]):
        inverted_unique_tuples = {v: k for k, v in self.unique_query_index.items()}
        inverted_relation_index = {v: k for k, v in self.relation_index.items()}
        return inverted_unique_tuples[tuple_relation[0]], \
            inverted_relation_index[tuple_relation[1]], \
            tuple_relation[2],\
            tuple_relation[3]

    def invert_index_formula(self, tuple_formula: Tuple[int, int, int, int, int]):
        inverted_unique_tuples = {v: k for k, v in self.unique_query_index.items()}
        inverted_indices_bodies = {v: k for k, v in self.bodies_index.items()}
        inverted_indices_formulas = {v: k for k, v in self.formulas_index.items()}
        return inverted_indices_bodies[tuple_formula[0]],\
            inverted_indices_formulas[tuple_formula[1]],\
            inverted_unique_tuples[tuple_formula[2]],\
            tuple_formula[3],\
            inverted_unique_tuples[tuple_formula[4]]
