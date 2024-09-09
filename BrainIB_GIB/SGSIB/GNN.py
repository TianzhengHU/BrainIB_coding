from typing import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv


class GNN(nn.Module):
    def __init__(self, num_of_features, encoder, device):
        super(GNN, self).__init__()
        self.encoder = encoder
        self.num_of_features = num_of_features
        self.first_gcn_dimensions = 128
        self.second_gcn_dimensions = 128
        self.heads = 8
        self.SOPOOL_dim_1 = 32
        self.SOPOOL_dim_2 = 32
        self.linear_hidden_dimensions = 32
        self.output_dimensions = 2
        
        self.device = device

        # GCN can be changed_____________
        if(self.encoder == "GCN"):
            self.graph_conv_1 = GCNConv(num_of_features, self.first_gcn_dimensions)
            self.graph_conv_2 = GCNConv(self.first_gcn_dimensions, self.second_gcn_dimensions)

        # GIN
        if(self.encoder == "GIN"):
            self.graph_conv_1 = GINConv(Sequential(Linear(num_of_features, self.first_gcn_dimensions), ReLU(),
                                                   Linear(self.first_gcn_dimensions, self.first_gcn_dimensions)))
            self.graph_conv_2 = GINConv(Sequential(Linear(self.first_gcn_dimensions, self.second_gcn_dimensions), ReLU(),
                                                   Linear(self.second_gcn_dimensions, self.second_gcn_dimensions)))

        self.SOPOOL = nn.Sequential(OrderedDict([
              ("Linear_1", nn.Linear(self.second_gcn_dimensions, self.SOPOOL_dim_1)),
              ("ReLU", nn.ReLU()),
              ("Linear_2", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_1)),
              ("ReLU", nn.ReLU()),
              ("Linear_3", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_2)),
              ("ReLU", nn.ReLU())
        ]))

        # GAT
        if(self.encoder == "GAT"):
            self.graph_conv_1 = GATConv(num_of_features, self.first_gcn_dimensions, heads=self.heads)
            self.graph_conv_2 = GATConv(self.first_gcn_dimensions * self.heads, self.second_gcn_dimensions, heads=self.heads)
            self.SOPOOL = nn.Sequential(OrderedDict([
                ("Linear_1", nn.Linear(self.second_gcn_dimensions * self.heads, self.SOPOOL_dim_1)),
                ("ReLU", nn.ReLU()),
                ("Linear_2", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_1)),
                ("ReLU", nn.ReLU()),
                ("Linear_3", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_2)),
                ("ReLU", nn.ReLU())
            ]))

        
        self.MLP_1 = nn.Sequential(OrderedDict([
              ("Linear_1", nn.Linear(self.SOPOOL_dim_2 ** 2, self.linear_hidden_dimensions)),
              ("ReLU", nn.ReLU()),
              ("Linear_2", nn.Linear(self.linear_hidden_dimensions, self.linear_hidden_dimensions)),
              ("ReLU", nn.ReLU()),
              ("Linear_3", nn.Linear(self.linear_hidden_dimensions, self.output_dimensions)),
              ("ReLU", nn.ReLU()),
        ]))


    def forward(self, graph_batch, edge_weight=None):
        # edge_index, x, edge_attr, y, batch, ptr = graph_batch
        graph_batch = graph_batch.to(self.device)
        encoder = self.encoder
        if(encoder == "GCN"):
            node_features_1 = F.relu(self.graph_conv_1(x=graph_batch.x, edge_index=graph_batch.edge_index, edge_weight=graph_batch.edge_attr))
            node_features_2 = F.relu(self.graph_conv_2(x=node_features_1, edge_index=graph_batch.edge_index, edge_weight=graph_batch.edge_attr))
        if(encoder == "GIN"):
            node_features_1 = F.relu(self.graph_conv_1(x=graph_batch.x, edge_index=graph_batch.edge_index))
            node_features_2 = F.relu(self.graph_conv_2(x=node_features_1, edge_index=graph_batch.edge_index))
        if (encoder == "GAT"):
            node_features_1 = F.relu(self.graph_conv_1(x=graph_batch.x, edge_index=graph_batch.edge_index))
            node_features_2 = F.relu(self.graph_conv_2(x=node_features_1, edge_index=graph_batch.edge_index))


        node_features_ = F.dropout(node_features_2, p=0.5, training=self.training)
        normalized_node_features = F.normalize(node_features_, dim=1)
        
        def sep_graph(node_features, ptr):
            graphs = []
            for i in range(len(ptr)-1):
                temp1 = ptr[i]
                temp2 = ptr[i+1]
                graphs.append(node_features[temp1:temp2])
            return(graphs)

        normalized_node_features = sep_graph(normalized_node_features, graph_batch.ptr)

        HH_tensor = torch.Tensor()

        for graph in normalized_node_features:
            graph = self.SOPOOL(graph)
            temp = torch.mm(graph.t(), graph).view(1, -1)
            if HH_tensor.shape == (0,):
                HH_tensor = temp
            else:
                HH_tensor = torch.cat((HH_tensor, temp), dim=0)

        output = F.dropout(self.MLP_1(HH_tensor), p=0.5, training=self.training)
        
        torch.cuda.empty_cache()
        return HH_tensor, output

