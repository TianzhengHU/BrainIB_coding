import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def get_dataset(name, sparse = True, cleaned = False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    node_feature_path = path + "/MUTAG/raw/MUTAG_node_labels.txt"
    # Open the file in read mode
    with open(node_feature_path, 'r') as file:
        # Read the contents of the file
        nodes_label_content = [int(line.strip()) for line in file]

    # if dataset.data.x is None:
    #     max_degree = 0
    #     degs = [dataset.data.x is None]
    #     for data in dataset:
    #         degs += [degree(data.edge_index[0], dtype=torch.long)]
    #         max_degree = max(max_degree, degs[-1].max().item())
    #
    #     if max_degree < 1000:
    #         dataset.transform = T.OneHotDegree(max_degree)
    #     else:
    #         # Assume degs is your list which may contain a mix of tensor and possibly boolean values
    #         for i, item in enumerate(degs):
    #             if isinstance(item, bool):  # Check if the item is a boolean
    #                 del degs[i]  # Remove the item
    #                 break  # Exit the loop after removing the first boolean value
    #
    #         deg = torch.cat(degs, dim=0).to(torch.float)
    #         mean, std = deg.mean().item(), deg.std().item()
    #         dataset.transform = NormalizedDegree(mean, std)

    num_nodes = max_num_nodes = 0
    node_pointer = 0
    node_labels_list = []
    for i in range(len(dataset)):
        data = dataset[i]
        node_labels = nodes_label_content[node_pointer: (node_pointer + data.num_nodes)]
        node_pointer = node_pointer + data.num_nodes
        node_labels_list.append(node_labels)

        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)
    num_nodes = max_num_nodes

    if not sparse:
        if dataset.transform is None:

            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])


    return dataset, num_nodes, node_labels_list

def get_edge_dataset(name, sparse = True, cleaned = False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    node_feature_path = path + "/MUTAG/raw/MUTAG_node_labels.txt"
    # Open the file in read mode
    with open(node_feature_path, 'r') as file:
        # Read the contents of the file
        nodes_label_content = [int(line.strip()) for line in file]

    num_nodes = max_num_nodes = 0
    node_pointer = 0
    node_labels_list = []
    for i in range(len(dataset)):
        data = dataset[i]
        node_labels = nodes_label_content[node_pointer: (node_pointer + data.num_nodes)]
        node_pointer = node_pointer + data.num_nodes
        node_labels_list.append(node_labels)

        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)
    num_nodes = max_num_nodes

    edge_dataset = [Data(node_attr = data.x, edge_index=data.edge_index, y=data.y) for data in dataset]

     # Initialize an empty adjacency matrix
    for i in range(len(dataset)):
        data = dataset[i]
        edge_index = data.edge_index
        adjacency_matrix = torch.zeros(max_num_nodes, max_num_nodes, dtype=torch.float)
        # adjacency_matrix = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float)

        # Fill the adjacency matrix based on the edge indices
        for j in range(edge_index.size(1)):
            src = edge_index[0, j]
            tgt = edge_index[1, j]
            edge_type = torch.argmax(data.edge_attr[j]) + 1
            adjacency_matrix[src, tgt] = edge_type
        edge_attr = torch.argmax(data.edge_attr, dim=1) + torch.ones(data.edge_attr.shape[0])
        edge_dataset[i].edge_attr = edge_attr.reshape(-1,1)
        edge_dataset[i].x = adjacency_matrix


    return edge_dataset, max_num_nodes, node_labels_list
    # return dataset, max_num_nodes, node_labels_list


def get_baseline_data(baseline_dataset_name, edge = False):
    if(edge == True):
        dataset, num_nodes, node_labels_list = get_edge_dataset(baseline_dataset_name)
        num_features = dataset[0].x.shape[1]
    else:
        dataset, num_nodes, node_labels_list = get_dataset(baseline_dataset_name)
        num_features = dataset.num_features

    return dataset, num_nodes, num_features, node_labels_list

def mutag_functional_groups_label(dataset, edge = False):
    # get ture label of functional groups
    # NO2
    func_true = []  # initial the true label set
    if (edge == False):
        for i in range(len(dataset)):
            # func_true.append([0] * dataset[i].num_nodes)
            # Set all elements in the first column to 0
            graph = dataset[i].x
            graph[:, 0] = 0
            graph_int = np.argmax(graph, axis=1)
            func_true.append(graph_int)
    else:
        for i in range(len(dataset)):
            graph = dataset[i].edge_attr.T
            # graph_convert = ((graph != 1) & (graph != 2)).int().tolist()
            graph_convert = (graph != 1).int().tolist()
            func_true.append(graph_convert)
    return func_true



if __name__ == '__main__':

    names = ['MUTAG', 'PROTEINS', 'DD', 'COLLAB']
    # dataset, num_nodes, num_features = get_dataset(names[0])
    dataset, num_nodes, dataset.num_features, node_labels_list = get_baseline_data(names[0])















