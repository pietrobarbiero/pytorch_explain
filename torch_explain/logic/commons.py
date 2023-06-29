from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Iterable
import json
import re

class Atom:
    def __init__(self, r: str=None, args: List[str]=None,
                 s: str=None, format: str='functional'):
        # Consider making s,r,args be property with getters and setters.
        # It was not done yet to avoid the cost of calling python
        # non-inlinable functions.
        if s is not None:
            self.read(s, format)
        else:
            self.r = r
            self.args = args

    def read(self, s: str, format: str='functional'):
        if format == 'functional':
            self._from_string(s)
        elif format == 'triplet':
            self._from_triplet_string(s)
        else:
            raise Exception('Unknown Atom format: %s' % format)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def __hash__(self):
        if self.args is not None:
            return (hash(self.r) ^ hash(tuple(self.args)))
        else:
            return hash(self.r)

    def __eq__(self, other):
        return self.r == other.r_kge and self.args == other.args

    def __repr__(self):
        args_for_print = [(f if f is not None else '*') for f in self.args]
        args_str = ','.join(args_for_print)
        return f'{self.r}({args_str}).'

    def ground(self, var_assignments: dict) -> bool:
        self.args = [var_assignments.get(a, None) for a in self.args]
        return all(self.args)

    def _from_string(self, a: str):
        a = re.sub('\b([(),\.])', '\1', a)
        a = a.strip()
        if a[-1] == ".":
            a = a[:-1]
        tokens = a.replace(
            '(', ' ').replace(
            ')', ' ').replace(
            ',', ' ').split()
        assert len(tokens) <= 3, str(tokens)
        assert tokens[0], tokens[1]
        self.r = tokens[0]
        self.args = [t for t in tokens[1:]]

    def _from_triplet_string(self, a: str):
        a = a.strip()
        tokens = a.split()
        assert len(tokens) == 3, str(tokens)
        self.r = tokens[1]
        self.args = [tokens[0], tokens[2]]


    def toTuple(self) -> Tuple:
        a = (self.r,) + tuple(self.args)
        return a



class Domain():

    def __init__(self, name:str, constants: List[str]):
        self.name = name
        self.constants = constants


class Predicate():

    def __init__(self, name: str, domains: List[Domain]):
        self.name = name
        self.domains = domains

class Rule(object):

    @staticmethod
    def default_domain() -> str:
        return 'domain'

    def __init__(self, name: str = None, weight: float = 1.0,
                 body: List[str] = None, head: List[str] = None,
                 body_atoms: List[Tuple] = None, head_atoms: List[Tuple] = None,
                 var2domain: Dict[str, str]=None,
                 s: str = None, format: str='functional'):
        self._name = name
        self.weight = weight
        self._body = None
        self._head = None

        if s is not None:
            self.read(s, format)
        elif body_atoms is not None or head_atoms is not None:
            self.body = body_atoms
            self.head = head_atoms
        else:
            self.body = [Atom(s=atom_str, format=format).toTuple()
                         for atom_str in body]
            self.head = [Atom(s=atom_str, format=format).toTuple()
                         for atom_str in head]


        _vars = sorted(list(set([v for a in (self.head + self.body) for v in a[1:]])))
        self.vars = OrderedDict()
        for v in _vars:
            self.vars[v] = (Rule.default_domain()
                            if var2domain is None or v not in var2domain
                            else var2domain[v])


        # We cache this for speed, make sure to always recompute on mutable
        # methods.
        self._hash = self._hash_impl()

    @property
    def name(self):
        return self._name

    @property
    def body(self):
        return self._body

    @property
    def head(self):
        return self._head

    @body.setter
    def body(self, value):
        self._body = value

    @head.setter
    def head(self, value):
        self._head = value

    @name.setter
    def name(self, name):
        self._name = name

    def read(self, s: str, format: str = 'functional'):
        if format == 'functional':
            self._from_r2n_string(s)
        elif format == 'expressgnn':
            self._from_expressgnn_string(s)
        else:
            assert False, 'Unknown Rule format %s' % format
        self._hash = self._hash_impl()

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def _hash_impl(self):
        return (hash(self.name) ^
                hash(self.weight) ^
                hash(tuple(self.body)) ^
                hash(tuple(self.head)))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (self.name == other.name and
                self.weight == other.weight and
                len(self.body) == len(other.body) and
                all(a == b for a, b in zip(self.body, other.body)) and
                len(self.head) == len(other.head) and
                all(a == b for a, b in zip(self.head, other.head)))

    def __repr__(self):
        body_str = ','.join([str(b) for b in self.body])
        head_str = ';'.join([str(h) for h in self.head])
        # Add weight?
        if self.name:
            return f'{self.name}: {body_str} -> {head_str}'
        else:
            return f'{body_str} -> {head_str}'

    def _from_r2n_string(self, a: str) -> None:
        a = re.sub('\b', '', a)  # remove all spaces
        name_end_index = a.find(':')  # cut the name out
        self.name = a[:name_end_index]
        a = a[name_end_index + 1:]
        weight_end_index = a.find(':')  # cut the weight out
        self.weight = float(a[:weight_end_index])
        a = a[weight_end_index + 1:]
        body_head_str = a.split('->')  # split body and head
        assert len(body_head_str) == 2
        body_str, head_str = body_head_str
        # Split the body atoms.
        body = re.split('\),', body_str.strip())
        body = [b + ')' if b[-1] != ')' else b for b in body]
        self.body = [Atom(s=b, format='functional').toTuple() for b in body]
        # Split the head atoms.
        head = re.split('\),', head_str.strip())
        head = [h + ')' if h[-1] != ')' else h for h in head]
        self.head = [Atom(s=h, format='functional').toTuple() for h in head]

    def _from_expressgnn_string(self, a: str) -> None:
        # Remove repeated whitespaces.
        a = re.sub('\b+', ' ', a).strip()
        tokens = a.split()
        assert len(tokens) >= 3
        self.name = tokens[0]
        atoms = [Atom(s=atom_str, format='functional').toTuple()
                 for atom_str in tokens[1:]]
        self.body = atoms[:-1]
        self.head = [atoms[-1]]
        self.weight = 1.0  # weight are not provided in this format.

    @property
    def hard(self) -> bool:
        return self.weight == 1.0

class RuleLoader(object):

    @staticmethod
    def load(filepath:str, num_rules:int, format:str='functional',
             force_hard_rules:bool=False) -> List[Rule]:
        rules = []
        with open(filepath) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                r = Rule(s=line, format=format)
                if force_hard_rules:
                    r.weight = 1.0
                rules.append(r)
                if len(rules) == num_rules:
                    break
        return rules


class RuleGroundings():

    def __init__(self, name: str,
                 # List of 2-dim-tuple: list of head groundings, list of body
                 groundings: Tuple[Tuple[Tuple[str, str, str]],  # heads
                                   Tuple[Tuple[str, str, str]]  #bodies
                                   ]):
        self.name = name
        self.groundings = groundings

    def __repr__(self):
        args_for_print = ['(%s)' % str(g) for g in self.groundings]
        args_str = ','.join(args_for_print)
        return f'{self.name}({args_str}).'

    def __eq__(self, r):
        if self.name != r.name or len(self.groundings) != len(r.groundings):
            return False
        for g1,g2 in zip(self.groundings, r.groundings):
            if str(g1) != str(g2):
                return False
        return True




class FOL():

    def __init__(self, domains: List[Domain],
                 predicates: List[Predicate],
                 facts: Iterable[Union[Atom, str, Tuple]] = None,
                 format: str = 'functional'):
        self.predicates = predicates
        self.domains = domains
        _facts = facts if facts is not None else []
        self._facts = [
            a if isinstance(a, Tuple) else a.toTuple() if isinstance(a, Atom) else Atom(s=a, format=format).toTuple()
            for a in _facts]
        self.format = format

    @property
    def facts(self):
        return self._facts
