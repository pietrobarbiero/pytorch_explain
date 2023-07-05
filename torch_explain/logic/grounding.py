from typing import List
from itertools import product
from .commons import Rule


class DomainGrounder:

    def __init__(self, domains, rules: List[Rule],
                 restrict_to_query_constants: bool=False,
                 limit: int=0):
        self.rules = rules
        # The flat grounder is not query oriented.
        self.domains = domains
        self.restrict_to_query_constants = restrict_to_query_constants # does not ground rules on constants that are not in the queries
        self.limit = limit

    #@lru_cache
    def ground(self):
        res = {}
        for clause in self.rules:
            added = 0
            groundings = []
            substitutions = []
            for ground_vars in product(*[self.domains[d] for d in clause.vars.values()]):

                var_assignments = {k:v for k,v in zip(
                    clause.vars.keys(), ground_vars)}

                # We use a lexicographical order of the variables
                constant_tuples = [v for k,v in sorted(var_assignments.items(), key= lambda x: x[0])]

                body_atoms = []
                for atom in clause.body:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    body_atoms.append(ground_atom)

                head_atoms = []
                for atom in clause.head:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    head_atoms.append(ground_atom)
                groundings.append((tuple(head_atoms), tuple(body_atoms)))
                substitutions.append(constant_tuples)
                added += 1
                if self.limit > 0 and self.limit >= added:
                    break

            res[clause.name] = (groundings, substitutions)
        return res
