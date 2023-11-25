import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
subjects = pd.read_csv("data/classhook_subjects_sample.csv")


def plot_hierarchy(df):
    subjects = df.to_dict('records')
    subjects_by_id = {int(subject['id']): subject for subject in subjects}
    children_by_parent = defaultdict(list)
    for subject in subjects:
        parent_id = subject['parent_id']
        if pd.notnull(parent_id):
            children_by_parent[int(parent_id)].append(subject)

    def build_tree(subject_id):
        """Recursively build the tree."""
        subject = subjects_by_id[subject_id]
        children = children_by_parent.get(subject_id, [])
        return {
            "name": subject['name'],
            "id": subject_id,
            "children": [build_tree(int(child['id'])) for child in children]
        }

    trees = [build_tree(int(subject['id']))
             for subject in subjects if subject['granularity'] == '0']
    print(trees)
    G = nx.DiGraph()

    def add_to_graph(node, parent=None):
        G.add_node(node['name'])
        if parent:
            G.add_edge(parent, node['name'])
        for child in node['children']:
            add_to_graph(child, node['name'])

    for tree in trees:
        add_to_graph(tree)
    print(G.nodes())
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=40, node_color="skyblue",
            font_size=8, width=1, edge_color='gray', font_weight='bold')
    plt.title("Subject Hierarchy")
    plt.show()


plot_hierarchy(subjects)
