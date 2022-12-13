def serialize_rules(concepts, rules):
    concept_name_to_object = {}
    for i, c in enumerate(concepts):
        concept_name_to_object[c] = Concept(name=c, id=i)

    roots = []
    for rule in rules:
        task = rule["name"]
        ands = []
        for and_ in rule["explanation"].split("|"):
            and_ = and_.replace("(", "").replace(")", "")
            literals = []
            for literal in and_.split("&"):
                literal = literal.strip()
                if "~" in literal:
                    l = Not([concept_name_to_object[literal.replace("~", "")]])
                else:
                    l = concept_name_to_object[literal]
                literals.append(l)
            ands.append(And(children=literals))
        roots.append(Or(ands, name=task))

    tree = ExpressionTree(roots=roots)
    return tree


class TreeNode:

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
