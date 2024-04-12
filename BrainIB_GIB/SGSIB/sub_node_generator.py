import argparse

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

# from BrainIB_V2.real_data.create_dataset import read_test_dataset

from torch_geometric.loader import DataLoader


class GIB(torch.nn.Module):
    def __init__(self, args, number_of_features, device):
        super(GIB, self).__init__()
        self.device = device
        self.args = args
        self.number_of_features = number_of_features
        self._setup()
        self.mseloss = torch.nn.MSELoss()
        self.relu = torch.nn.ReLU()
        # self.subgraph_const = self.args.subgraph_const
    def _setup(self):
        self.graph_convolution_1 = GCNConv(self.number_of_features, self.args.first_gcn_dimensions)
        self.graph_convolution_2 = GCNConv(self.args.first_gcn_dimensions, self.args.second_gcn_dimensions)
        self.fully_connected_1 = torch.nn.Linear(self.args.second_gcn_dimensions, self.args.first_dense_neurons)
        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons, self.args.second_dense_neurons)

    def forward(self, data):
        # edges = real_data["edges"]
        # features = real_data["features"]
        subgraph = data.to(self.device)
        # print(type(real_data))
        edges = data['edge_index']
        features = data['x']
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
        assignment = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)

        subgraph.attr = assignment
        pos_penalty = assignment.var()
        return subgraph, pos_penalty


"""
调用的时候 self.model = Subgraph(self.args, self.dataset_generator.number_of_features)
"""
# def parameter_parser():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("--real_data", nargs="?", default= "../input/qed/", help="Folder with training graph jsons.")
#     parser.add_argument("--first-gcn-dimensions", type=int, default=16, help="Filters (neurons) in 1st convolution. Default is 16.")
#     parser.add_argument("--second-gcn-dimensions", type=int, default=8, help="Filters (neurons) in 2nd convolution. Default is 8.")
#     parser.add_argument("--first-dense-neurons", type=int, default=16, help="Neurons in SAGE aggregator layer. Default is 16.")
#     parser.add_argument("--second-dense-neurons", type=int, default=2, help="assignment. Default is 2.")
#     parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 5)')
#     return parser.parse_args()
#
# # if __name__ == '__main__':
# #     args = parameter_parser()
# #
# #     dataset = read_test_dataset(10)
# #     print(len(dataset))
# #     indices = range(0, len(dataset), args.batch_size)
# #     for i in indices:
# #         graphs = dataset[i: i + args.batch_size]
# #         batch_graph = next(iter(DataLoader(graphs, batch_size=len(graphs))))
# #         # embeddings, original_output = model(batch_graph)
# #         # subgraphs = copy.deepcopy(graphs)
# #         number_of_features = 116
# #         # positive_penalty = torch.Tensor([0.0]).float().to(device)
# #         sub_model = Subgraph(args, number_of_features)
# #
# #         embeddings, positive, negative, cls_loss, positive_penalty = sub_model.forward(graphs)
# #             # graph = subgraph.to(device)
# #         positive_penalty += positive_penalty
#
#
#     """
#     what I need are: subgraph, pos_penalty
#     """
#
#



