import pydot
import subprocess


def render_tree(node, graph=None, path=[], fname=None, dist_func=None, ignore_value=True):
    """renders a tree"""
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
        ch = render_tree(child, graph=graph, path=path,
                    dist_func=dist_func, ignore_value=ignore_value)
        dist = dist_func(n, ch) if dist_func is not None else ''
        graph.add_edge(pydot.Edge(n, ch, label=dist))

    if fname is not None:
        graph.write_png(fname)

    return n


def to_gif(path, output):
    """creates a gif from a directory of pngs"""
    subprocess.call([
        'convert',
        '-delay', '1x5',
        path,
        '-layers', 'Optimize',
        output
    ])
