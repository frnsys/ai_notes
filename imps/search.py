tree_data = {
    'state': 'S',
    'children': [{
        'state': 'A',
        'children': [{
            'state': 'B',
            'children': [{
                'state': 'C',
                'children': []
            }, {
                'state': 'J',
                'children': []
            }]
        }, {
            'state': 'F',
            'children': [{
                'state': 'H',
                'children': [{
                    'state': 'G',
                    'children': []
                }]
            }, {
                'state': 'I',
                'children': []
            }]

        }]
    }, {
        'state': 'E',
        'children': [{
            'state': 'F',
            'children': [{
                'state': 'H',
                'children': [{
                    'state': 'G',
                    'children': []
                }]
            }, {
                'state': 'I',
                'children': []
            }]
        }, {
            'state': 'D',
            'children': []
        }, {
            'state': 'K',
            'children': []
        }]
    }, {
        'state': 'L',
        'children': [{
            'state': 'G',
            'children': []
        }, {
            'state': 'O',
            'children': [{
                'state': 'M',
                'children': []
            }, {
                'state': 'N',
                'children': []
            }]
        }]
    }]
}

value_for_state = {
    'S': 0,
    'G': 9,
    'A': 4,
    'B': 1,
    'C': 3,
    'D': 8,
    'E': 5,
    'F': 2,
    'H': 7,
    'I': 0,
    'J': 2,
    'K': 8,
    'L': 4,
    'M': 5,
    'N': 3,
    'O': 7
}

pos_for_state = {
    'S': (2,1),
    'G': (6,3),
    'A': (2,3),
    'B': (3,4),
    'C': (3,5),
    'D': (0,4),
    'E': (1,2),
    'F': (2,5),
    'H': (2,6),
    'I': (3,6),
    'J': (4,6),
    'K': (2,4),
    'L': (3,2),
    'M': (4,5),
    'N': (5,6),
    'O': (4,3)
}

def score(node):
    """computes a value/utility for a node"""
    return value_for_state[node.state]

def heuristic(node_a, node_b):
    """computes an admissible heuristic cost/distance between two nodes.
    here, using manhattan distance."""
    x1, y1 = node_a.pos
    x2, y2 = node_b.pos
    return abs(x1-x2) + abs(y1-y2)

import math

def distance(node_a, node_b):
    """compute cost/distance to move from one state to another.
    here, using euclidean distance."""
    x1, y1 = node_a.pos
    x2, y2 = node_b.pos
    return math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))


class Node():
    def __init__(self, state, children=[], parent=None):
        self.state = state
        self.pos = pos_for_state[state]
        self.value = score(self)
        self.children = sorted([Node(**ch, parent=self) for ch in children],
                               key=lambda ch: score(ch),
                               reverse=True)

    def __repr__(self):
        return 'Node({})'.format(self.state)

import pydot
def render(node, graph=None, path=[], fname=None, ignore_distance=True, ignore_value=True):
    if graph is None:
        graph = pydot.Dot(graph_type='graph')
    attrs = {}
    if node in path:
        attrs = {
            'style': 'filled',
            'fillcolor': 'gold'
        }

    label = node.state if ignore_value else '{} ({})'.format(node.state, node.value)
    n = pydot.Node(name=node.state, label=label, shape='circle', **attrs)
    graph.add_node(n)
    for child in node.children:
        ch = render(child, graph=graph, path=path,
                    ignore_distance=ignore_distance, ignore_value=ignore_value)
        dist = distance(n, ch) if not ignore_distance else ''
        graph.add_edge(pydot.Edge(n, ch, label=dist))

    if fname is not None:
        graph.write_png(fname)

    return n

def successors(node):
    # in this simple example we have the whole tree in memory,
    # but the more typical case involves expanding nodes as we go.
    # this function is meant to demonstrate that.
    return node.children

def depth_first(root, goal):
    """DFS, assuming uniform costs"""

    # FIFO data structure (stack)
    fringe = [[root]]
    seen = set()
    while fringe:
        path = fringe.pop()
        node = path[-1]

        # extended list filtering:
        # skip nodes we have already seen
        if node.state in seen: continue
        seen.add(node.state)

        yield path

        if node.state == goal.state:
            break

        # b/c this is FIFO, reverse the order so the highest value comes first
        fringe.extend(reversed([path + [child] for child in successors(node)]))


def iterative_deepening(root, goal):
    """iterative deepening iteratively runs depth-first search at increasing depths
    until a solution is found. this implementation is a bit weird in order to yield all
    attempted paths (for visualization purposes)."""

    def dfs(path, depth, seen):
        if depth == 0:
            return

        node = path[-1]

        # extended list filtering:
        # skip nodes we have already seen
        if node.state in seen:
            return
        seen.add(node.state)

        yield path

        if node.state == goal.state:
            return

        for child in successors(node):
            paths = [p for p in dfs(path + [child], depth-1, seen)]
            yield from paths
            if paths and paths[-1][-1].state == goal.state:
                return

    # while no solution has been found,
    # try greater depths
    depth = 1
    solution = None
    while solution is None:
        paths = [p for p in dfs([root], depth, set())]
        yield from paths
        if paths and paths[-1][-1].state == goal.state:
            solution = paths[-1]
        depth += 1


def breadth_first(root, goal):
    """BFS, assuming uniform costs"""

    # LIFO data structure (queue)
    fringe = [[root]]
    seen = set()
    while fringe:
        path = fringe.pop(0)
        node = path[-1]

        # extended list filtering:
        # skip nodes we have already seen
        if node.state in seen: continue
        seen.add(node.state)

        yield path

        if node.state == goal.state:
            break

        fringe.extend([path + [child] for child in successors(node)])


def british_museum(root, goal):
    """exhaustive search"""
    fringe = [[root]]
    while fringe:
        path = fringe.pop()
        yield path

        node = path[-1]
        fringe.extend(reversed([path + [child] for child in successors(node)]))


def greedy_best_first(root, goal):
    """always choose the node which, according to the heuristic,
    appears to be closest to the goal"""
    fringe = [[root]]
    while fringe:
        path = fringe.pop(0)
        yield path

        node = path[-1]
        if node.state == goal.state:
            break

        # diff b/w best-first search and hill climbing is that
        # in best-first, we sort the entire fringe (by closeness to goal)
        fringe.extend([path + [child] for child in successors(node)])
        fringe = sorted(fringe, key=lambda path: heuristic(path[-1], goal))


def hill_climbing(root, goal):
    fringe = [[root]]
    while fringe:
        path = fringe.pop(0)
        yield path

        node = path[-1]
        if node.state == goal.state:
            break

        # diff b/w best-first search and hill climbing is that
        # in hill climbing, we only sort children (by closeness to goal)
        children = sorted(successors(node), key=lambda child: heuristic(child, goal))
        fringe.extend([path + [child] for child in children])


def branch_and_bound(root, goal):
    """always extend the cumulatively shortest path.
    once the goal is reached, extend all extendible paths
    that are are shorter than the current best path"""
    def path_length(path):
        return sum(distance(n1, n2) for n1, n2 in zip(path, path[1:]))

    fringe = [[root]]
    best_path = None
    while fringe:
        path = fringe.pop(0)
        yield path

        node = path[-1]
        if node.state == goal.state:
            best_path = path

        fringe.extend([path + [child] for child in successors(node)])

        # sort paths by length
        fringe = sorted(fringe, key=lambda path: path_length(path))

        if best_path is not None:
            # check remaining paths
            fringe = [p for p in fringe if path_length(p) < path_length(best_path)]

    yield best_path


def beam(root, goal, width=2):
    """beam search is essentially BFS with two main differences:
    it uses limited memory (specified by the width parameter),
    and it uses a heuristic to determine which paths to keep in memory.
    this has the effect of limiting the search space to the point where
    a solution is not guaranteed (i.e. beam search is not complete)"""
    fringe = [[root]]
    seen = set()
    while fringe:
        path = fringe.pop(0)
        node = path[-1]

        # extended list filtering:
        # skip nodes we have already seen
        if node.state in seen: continue
        seen.add(node.state)

        yield path

        if node.state == goal.state:
            break

        fringe.extend([path + [child] for child in successors(node)])
        fringe = sorted(fringe, key=lambda path: heuristic(path[-1], goal))
        fringe = fringe[:width]


def astar(root, goal):
    def path_length(path):
        # g(n), i.e. the exact length of the path
        return sum(distance(n1, n2) for n1, n2 in zip(path, path[1:]))

    fringe = [[root]]
    while fringe:
        path = fringe.pop(0)
        yield path

        node = path[-1]
        if node.state == goal.state:
            break

        fringe.extend([path + [child] for child in successors(node)])

        # in effect, sorting by g(n) + h(n), where h(n) is the heuristic
        # distance from node n to the goal
        fringe = sorted(fringe, key=lambda path: path_length(path) + heuristic(path[-1], goal))


def iterative_deepening_astar(root, goal):
    """with IDA* (usually just called "IDA"),
    an approach similar to iterative deepening DFS is used"""

    def ida(path, length, depth, seen):
        node = path[-1]

        f = length + heuristic(node, goal)
        if f > depth: return f, None

        if node.state == goal.state:
            return f, path

        # extended list filtering:
        # skip nodes we have already seen
        if node.state in seen: return f, None
        seen.add(node.state)

        minimum = float('inf')
        best_path = None
        for child in successors(node):
            # g(n) = distance(n)
            thresh, new_path = ida(path + [child],
                                   length + distance(node, child),
                                   depth, seen)
            if thresh < minimum:
                minimum = thresh
                best_path = new_path
        return minimum, best_path

    solution = None
    while solution is None:
        depth, solution = ida([root], 0, heuristic(root, goal), set())
    print(solution)
    return solution



def minimax(root, player=0, depth=4):
    """minimax does not seek a specific goal state; rather, it tries to find the
    best-scoring path. minimax generally assumes a fully expanded game tree but that
    is seldom possible, so in practice it generally only expands the tree out to a
    certain depth. the values leafs/terminal nodes of the tree are propagated upwards.
    it works from the bottom up so the path is build in reverse.
    this is for a two-player game, so player must be in [0,1],
    and we assume it is player 0's turn at the root.
    """
    total_depth = depth

    def mm(path, depth):
        node = path[-1]

        # reached leaves, just return their value
        if depth == 0:
            return node.value, path

        children = successors(node)

        # if no children, this is also a leaf
        if not children:
            return node.value, path

        # if we haven't reached the leaves,
        # keep expanding the tree
        paths = []
        for child in children:
            paths.append(mm(path + [child], depth - 1))

        # if it is this player's turn, maximize
        # otherwise, minimize
        turn = total_depth - depth
        is_my_turn = turn % 2 == player

        # max if our turn
        if is_my_turn:
            result = max(paths, key=lambda x: x[0])
        # min if opponents turn
        else:
            result = min(paths, key=lambda x: x[0])
        return result

    value, path = mm([root], depth)
    return value, path


def alpha_beta(root, player=0, depth=4):
    total_depth = depth

    def ab(path, depth, alpha, beta):
        node = path[-1]

        # reached leaves, just return their value
        if depth == 0:
            return node.value, path

        children = successors(node)

        # if no children, this is also a leaf
        if not children:
            return node.value, path

        # if it is this player's turn, maximize
        # otherwise, minimize
        turn = total_depth - depth
        is_my_turn = turn % 2 == player

        if is_my_turn:
            value = float('-inf')
            best_path = path
            for child in children:
                new_value, new_path = ab(path + [child], depth - 1, alpha, beta)
                if new_value > value:
                    value = new_value
                    alpha = max(alpha, value)
                    best_path = new_path
                if beta <= alpha:
                    break
        else:
            value = float('inf')
            best_path = path
            for child in children:
                new_value, new_path = ab(path + [child], depth - 1, alpha, beta)
                if new_value < value:
                    value = new_value
                    beta = min(beta, value)
                    best_path = new_path
                if beta <= alpha:
                    break
        return value, best_path

    value, path = ab([root], depth, float('-inf'), float('inf'))
    return value, path




if __name__ == '__main__':
    import os
    import shutil
    import subprocess

    methods = {
        'uninformed': [
            depth_first,
            breadth_first,
            iterative_deepening,
            british_museum
        ],
        'informed': [
            greedy_best_first,
            hill_climbing,
            branch_and_bound,
            # iterative_deepening_astar, # TODO this doesn't yield attempted paths
            beam,
            astar
        ],
        'adversarial': [
            minimax,
            alpha_beta
        ]
    }

    for type in methods.keys():
        for method in methods[type]:
            name = method.__name__
            print(name)

            path = 'outputs/{}'.format(name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path)

            root = Node(**tree_data)

            if type == 'adversarial':
                solution = method(root)
                print('->', solution)
            else:
                goal = Node(state='G')
                for i, path in enumerate(method(root, goal=goal)):
                    render(root, path=path, fname='outputs/{}/{}.png'.format(name, i))
                    subprocess.call(['./to_gif.sh', name])