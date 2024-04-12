import argparse
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch_geometric.data import Data
import math
# import synthetic_structsim
from BrainIB_V2.synthetic_data import synthetic_structsim
# import featgen
from BrainIB_V2.synthetic_data import featgen
"""
BA-motif-Volume
 - In this fuction, we choose a number of nodes to generate a graph with BA function.  
 - All edges' weight in this graph will be 1, and the edges' weight will generated as random number in range 0.00 to 100.00.  
 - The y label of this graph will be the sum of all nodes' weight.  
"""
def generate_BA(node_number_generate):
    """Returns a random graph according to the Barabási–Albert preferential
    Attachment model.

    A graph of ``n`` nodes is grown by attaching new nodes each with ``m``
    Edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If ``m`` does not satisfy ``1 <= m < n``.
    """

    G = nx.barabasi_albert_graph(node_number_generate, 2)
    return G
def get_node_features(G):
    """
    get the adj_maxtrix and  nodes' neighbers of one graph G
    """
    nodes_number = G.number_of_nodes()
    nodes_list = np.linspace(0, nodes_number - 1, nodes_number)

    adj_maxtrix = np.array(nx.adjacency_matrix(G).todense())
    neighbers = nx.to_dict_of_lists(G, nodelist=nodes_list)

    node_features = {"adj_maxtrix": adj_maxtrix,
                     "neighbers": neighbers}
    return node_features
def get_edges_list(edges_number):
    """
    return an edge list with all weights are 1
    """
    edges_weight_list = np.ones(edges_number)
    return edges_weight_list
def get_nodes_weight_and_y(nodes_number, motif_number):
    """
    return nodes' feature
            and
            y_label
    """
    nodes_weight_list = []
    for i in range(nodes_number):
        nodes_weight_list.append(np.random.uniform(0.00,100.00))

    np.asarray(nodes_weight_list)
    # y_label = np.sum(nodes_weight_list[basic_BA_number:nodes_number])
    y_label = (np.sum(nodes_weight_list[nodes_number-5*motif_number:nodes_number]))/len(nodes_weight_list[nodes_number-10:nodes_number])

    return nodes_weight_list, y_label
def display_basic(G):
    nx.draw(G, with_labels=True)
    plt.show()
def display(G):
    pos = nx.spring_layout(G)
    # plt.rcParams['figure.figzise'] = [6,4]
    labels = {}
    for node in G.nodes():
        labels[node] = node

    plt.figure(dpi=300, figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=None, node_size=40, node_color='#1f78b4', node_shape='o', alpha=None,
                           cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, edgecolors=None, label=None,
                           margins=None)
    nx.draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k', style='solid', alpha=None, arrowstyle=None,
                           arrowsize=10, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=None,
                           label=None, node_size=40, nodelist=None, node_shape='o', connectionstyle='arc3',
                           min_source_margin=0, min_target_margin=0)
    nx.draw_networkx_labels(G, pos, labels=None, font_size=5, font_color='k', font_family='sans-serif',
                            font_weight='normal', alpha=None, bbox=None, horizontalalignment='center',
                            verticalalignment='center', ax=None, clip_on=True)
    # plt.axis = off

    plt.show()
def add_motif_to_BA(G , motif_number, node_number_generate):
    """
     In BA- SHAPES, we start with a base Baraba ́si-Albert (BA) graph
     on 300 nodes and a set of 80 five-node “house”-structured network motifs,
     which are attached to randomly selected nodes of the base graph.
     在 BA-SHAPES 中，我们从一个有 300 个节点的基础 Baraba ́si-Albert（BA）图和一组 80 个五节点的 "房子 "结构网络图案开始，
     将这些图案连接到随机选择的基础图节点上。
    """
    house = nx.house_x_graph()
    # display(house)
    for i in range(motif_number):
        G = nx.disjoint_union(G,house)
        node_in_G = np.random.randint(0, 10)
        node_in_motify = np.random.randint(0, 5) + node_number_generate + i * 5
        nx.add_path(G, [node_in_G, node_in_motify], weight=1)
#     display(G)

    """
    The resulting graph is further perturbed by adding 0.1N random edges.
    通过添加 0.1N 条随机边对生成的图进行进一步扰动。
    """
    number_of_nodes = nx.number_of_nodes(G)
    N = int(number_of_nodes/10)
    for i in range(N):
         while(True):
            node_a = np.random.randint(0, number_of_nodes)
            node_b = np.random.randint(0, number_of_nodes)
#             print("1:",node_a,node_b)
            if(node_a == node_b):
                continue
            else:
                nx.add_path(G, [node_a,node_b], weight=1)
#                 print("total nodes:",number_of_nodes, node_a,node_b)
                break
#     display(G)
    return G
"""
In this fucntion, run once, get an BA graph
return parameters:
    adj_maxtrix
    edges_weight_list
    nodes_weight_list
    y_label
"""
def generate_a_BA_motif_Volume(node_number_generate, motif_number):
    G = generate_BA(node_number_generate)
    G = add_motif_to_BA(G, motif_number, node_number_generate)

    node_features = get_node_features(G)
    adj_maxtrix = node_features["adj_maxtrix"]
    neighbers = node_features["neighbers"]

    edges_number = G.number_of_edges()
    nodes_number = G.number_of_nodes()

    edges_weight_list = get_edges_list(edges_number)
    nodes_weight_list, y_label = get_nodes_weight_and_y(nodes_number, motif_number)

    volume_graph = {"G": G,
                    "adj_maxtrix": adj_maxtrix,
                    "neighbers": neighbers,
                    "edges_weight_list": edges_weight_list,
                    "nodes_weight_list": nodes_weight_list,
                    "y_label": y_label}

    return volume_graph
def generate_one_graph(node_number_generate, motif_number):
    volume_graph = generate_a_BA_motif_Volume(node_number_generate, motif_number)
#     G.append(volume_graph["G"])
    adj_maxtrix = volume_graph["adj_maxtrix"]
    neighbers = volume_graph["neighbers"]
    edges_weight_list = [int(item) for item in volume_graph["edges_weight_list"]]

    nodes_weight_list = volume_graph["nodes_weight_list"]
    y_label = volume_graph["y_label"]

    """
    x
    """
    # 将行向量 v 扩展为与 A 同形状的矩阵
    V = np.tile(nodes_weight_list, (adj_maxtrix.shape[0], 1))
    B = adj_maxtrix * V
    B_ = adj_maxtrix * V.transpose()
    A = torch.tensor((B + B_)/200)
    """
    edge_index
    """
    nei_0 = []
    for j in range(len(neighbers.keys())):
        nei_0.append([list(neighbers.keys())[j]] * len(list(neighbers.values())[j]))
    nei = [int(item) for sublist in nei_0 for item in sublist]
    nei_0f = torch.tensor(nei)
    nei_1 = list(neighbers.values())
    nei_1f = torch.tensor([item for sublist in nei_1 for item in sublist])
    nei = torch.cat((nei_0f,nei_1f), -1).reshape(2,-1)

    """
    edge_attr
    """
    edge_wei = torch.tensor(np.concatenate([row[row != 0] for row in A])).unsqueeze(1)
    # edge_wei = torch.tensor([edges_weight_list]*2).reshape(-1, 1)
    return A,nei,edge_wei,y_label
"""
final function of the generate function
"""
def create_synthetic_dataset(synthetic_data_number):
    dataset = []
    node_number_generate, motif_number = 91, 5
    for i in range(synthetic_data_number):
        A,nei,edge_wei,y_label = generate_one_graph(node_number_generate, motif_number)
        y = (y_label > 125).astype(int)
        y_tensor = torch.tensor([y])
        graph = Data(x=A.float(), edge_index=nei.long(), edge_attr=edge_wei.float(), y=y_tensor.long())
        dataset.append(graph)

    print("finished create synthetic real_data")
    return dataset
# from create_dataset import read_test_dataset
# sample_dataset = read_test_dataset(1)
# sample_data = sample_dataset[0]
#
#
# create_synthetic_dataset(2)


"""
Below are the fucntions followed by GNNExplainer
"""
####################################
#
# Experiment utilities
#
####################################
def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def build_graph(
    width_basis,
    basis_type,
    list_shapes,
    start=0,
    rdm_basis_plugins=False,
    add_random_edges=0,
    m=5,
):
    """This function creates a basis (scale-free, path, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    m                :      number of edges to attach to existing node (for BA graph)
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    plugins          :      node ids with the attached shapes
    """
    if basis_type == "ba":
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis)

    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motifs
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(n_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    seen_shapes = {"basis": [0, n_basis]}

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        if shape_type == "cycle":
            if np.random.random() > 0.5:
                a = np.random.randint(1, 4)
                b = np.random.randint(1, 4)
                basis.add_edges_from([(a + start, b + plugins[shape_id])])
        temp_labels = [r + col_start for r in roles_graph_s]
        # temp_labels[0] += 100 * seen_shapes[shape_type][0]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
            print(src, dest)
            basis.add_edges_from([(src, dest)])

    return basis, role_id, plugins


def gen_syn1(args, feature_generator=None, m=5):
    """ Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
    nb_shapes = args.sz_nb_shapes
    width_basis = args.sz_width_basis
    basis_type = "ba"
    list_shapes = [["house"]] * nb_shapes

    # plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(args,
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name

def gen_nc(args, feature_generator=None, m=5):
    """ Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
    nb_shapes = args.nc_nb_shapes
    width_basis = args.nc_width_basis
    basis_type = "ba"
    list_shapes = [["house"]] * nb_shapes

    # plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(args,
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def syn_task1(args, writer=None):
    # real_data
    G, labels, name = gen_syn1(args,
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    num_classes = max(labels) + 1
    return G, labels, name, num_classes


def syn_nc(args, writer=None):
    # real_data
    G, labels, name = gen_nc(args,
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    num_classes = max(labels) + 1
    return G, labels, name, num_classes

def convert_graph(G, true_label):
    edge = torch.tensor(list(G.edges()))
    y = torch.Tensor([true_label])
    Features = torch.tensor(G.nodes[0]['feat'])
    node_number = len(G.nodes)
    # node_number = 3
    for i in range(1, node_number):
        Features = torch.cat((Features, torch.tensor(G.nodes[i]['feat'])), dim=-1)
    ROI = Features.reshape(node_number, -1)

    A = torch.sparse_coo_tensor(indices=edge[:, :2].t().long(), values=edge[:, -1].reshape(-1, ).float(), size=(node_number, node_number))
    G = (A.t() + A).coalesce()
    graph = Data(x=ROI.reshape(-1, node_number).float(), edge_index=G.indices().reshape(2, -1).long(),
                 edge_attr=G.values().reshape(-1, 1).float(),
    # edge_weak_connet=edge_weak_connet, edge_weight=edge_weight,
    y=y.long())
    return graph

def parameter_parser():
    parser = argparse.ArgumentParser(description='Synthetic')
    parser.add_argument('--input_dim', dest='input_dim', type=int, default=105,
                        help='Input feature dimension (default: 105)')
    parser.add_argument('--number_of_patients', dest='number_of_patients', type=int, default=300,
                        help='Input number of patients in synthetic dataset (default: 300)')
    parser.add_argument('--number_of_normal_control', dest='number_of_normal_control', type=int, default=300,
                        help='Input number of normal control in synthetic dataset (default: 300)')
    parser.add_argument('--sz_nb_shapes', dest='sz_nb_shapes', type=int, default=80,
                        help='Input number of motif house in synthetic dataset (default: 3)')
    parser.add_argument('--sz_width_basis', dest='sz_width_basis', type=int, default=300,
                        help='Input number of basic nodes in synthetic dataset (default: 90)')
    parser.add_argument('--nc_nb_shapes', dest='nc_nb_shapes', type=int, default=0,
                        help='Input number of motif house in synthetic dataset (default: 3)')
    parser.add_argument('--nc_width_basis', dest='nc_width_basis', type=int, default=700,
                        help='Input number of basic nodes in synthetic dataset (default: 105)')
    return parser.parse_args()

def create_syn_dataset_1(args):
    dataset = []
    for patient in range(args.number_of_patients):
        G, labels, name, num_classes = syn_task1(args, writer=None)
        graph = convert_graph(G, 1)
        dataset.append(graph)
    print(len(dataset))

    for patient in range(args.number_of_normal_control):
        G, labels, name, num_classes = syn_nc(args, writer=None)
        graph = convert_graph(G, 0)
        dataset.append(graph)
    print(len(dataset))

    random.shuffle(dataset)
    print("finish create Synthetic dataset")
    return dataset

if __name__ == '__main__':
    args = parameter_parser()
    syn_dataset = create_syn_dataset_1(args)
    print()





