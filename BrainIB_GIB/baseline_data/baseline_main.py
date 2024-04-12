import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
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
    dataset.data.edge_attr = None
    if dataset.data.x is None:

        max_degree = 0
        degs = [dataset.data.x is None]
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            # Assume degs is your list which may contain a mix of tensor and possibly boolean values
            for i, item in enumerate(degs):
                if isinstance(item, bool):  # Check if the item is a boolean
                    del degs[i]  # Remove the item
                    break  # Exit the loop after removing the first boolean value

            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)


    num_nodes = max_num_nodes = 0
    for data in dataset:
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)
    num_nodes = max_num_nodes

    if not sparse:
        if dataset.transform is None:

            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset, num_nodes

def get_baseline_data(baseline_dataset_name):
    dataset, num_nodes = get_dataset(baseline_dataset_name)
    return dataset, num_nodes, dataset.num_features

if __name__ == '__main__':

    names = ['MUTAG', 'PROTEINS', 'DD', 'COLLAB']
    dataset, num_nodes, num_features = get_dataset(names[0])
















