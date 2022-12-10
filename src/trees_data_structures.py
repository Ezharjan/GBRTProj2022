class RegressionTreeNode:
    def __init__(self):
        self.j = None
        self.s = None
        self.left_descendant = None
        self.right_descendant = None
        self.const = None

    def make_terminal(self, const):
        self.const = const

    def split(self, j, s):
        self.j = j
        self.s = s
        self.left_descendant = RegressionTreeNode()
        self.right_descendant = RegressionTreeNode()

    def print_sub_tree(self, depth=0):
        indentation = '\t' * depth

        if self.is_terminal():
            print('{}return {}'.format(indentation, self.const))
        else:
            print("{}if x['{}'] <= {} then:".format(indentation, self.j, self.s))
            self.left_descendant.print_sub_tree(depth=depth + 1)
            print("{}if x['{}'] > {} then:".format(indentation, self.j, self.s))
            self.right_descendant.print_sub_tree(depth=depth + 1)

    def is_terminal(self):
        return self.const is not None

    def evaluate(self, x):
        if self.is_terminal():
            return self.const
        elif x[self.j] <= self.s:
            return self.left_descendant.evaluate(x)
        else:
            return self.right_descendant.evaluate(x)

    def set_const(self, label_col):
        self.const = label_col.mean()


class RegressionTree:
    def __init__(self):
        self.root = RegressionTreeNode()

    def get_root(self):
        return self.root

    def evaluate(self, x):
        return self.root.evaluate(x)


class RegressionTreeEnsemble:
    def __init__(self):
        self.trees = []
        self.weights = []
        self.M = 0
        self.c = 0

    def add_tree(self, tree, weight):
        self.trees.append(tree)
        self.weights.append(weight)
        self.M += 1

    def set_initial_constant(self, c=0):
        self.c = c

    def evaluate(self, x, m):
        evals = [tree.evaluate(x) * weight for tree, weight in zip(self.trees[:m], self.weights[:m])]

        return self.c + sum(evals)


