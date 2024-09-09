import copy

import wandb
import argparse
import shap
import networkx as nx

import torch
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Sequential, Linear, ReLU
from typing import OrderedDict

from torch_geometric.data import DataLoader

from BrainIB_V2.real_data.create_dataset import read_Schi_dataset
from BrainIB_V2.real_data.create_dataset import read_dataset
from BrainIB_V2.real_data.create_dataset import read_UCLA_dataset

from BrainIB_V2.SGSIB.utils import separate_data
from BrainIB_V2.SGSIB.GNN import GNN
def parameter_parser():
    parser = argparse.ArgumentParser(description='"traditional Machine Learning Methods"')

    parser.add_argument('--model', type=str, default="GAT",
                        help='input the traditional model name for training (default: GIN) GIN and GAT')
    parser.add_argument('--dataset', type=str, default="UCLA",
                        help='Pick the dataset from ABIDE, BSNIP and some baseline dataset (default: BSNIP)')
    parser.add_argument('--multi_site', type=bool, default=True,
                        help='Decide do the multisite training or not (default: False)')
    parser.add_argument('--training', type=str, default="normal",
                        help='input the type of training, all choice: normal, SHAP (default: normal)')
    parser.add_argument('--iters_per_epoch', type=int, default=1,
                        help='number of iterations per each epoch (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate of graph model (default: 0.0005)')
    return parser.parse_args()

def adapt_dataset(dataset):
    X = dataset[0].x.reshape(1,-1)
    if ('global_values' in dataset[0].keys):
        X = torch.cat((X, dataset[0].global_values.reshape(1, -1)), dim=1)
    y = dataset[0].y
    for i in range(1, len(dataset)):
        y = torch.cat((y, dataset[i].y), dim=0)
        if ('global_values' in dataset[i].keys):
            shap_values = torch.cat((dataset[i].x.reshape(1, -1), dataset[i].global_values.reshape(1, -1)), dim=1)
        else:
            shap_values = dataset[i].x.reshape(1, -1)
        X = torch.cat((X, shap_values), dim=0)
    return X, y


class GIN(torch.nn.Module):
    def __init__(self, num_of_features, device):
        super(GIN, self).__init__()
        self.num_of_features = num_of_features
        self.first_gcn_dimensions = 128
        self.second_gcn_dimensions = 128
        self.heads = 8
        self.SOPOOL_dim_1 = 32
        self.SOPOOL_dim_2 = 32
        self.linear_hidden_dimensions = 32
        self.output_dimensions = 2

        self.device = device
        # GIN
        self.graph_conv_1 = GINConv(Sequential(Linear(num_of_features, self.first_gcn_dimensions), ReLU(), Linear(self.first_gcn_dimensions, self.first_gcn_dimensions)))
        self.graph_conv_2 = GINConv(Sequential(Linear(self.first_gcn_dimensions, self.second_gcn_dimensions), ReLU(), Linear(self.second_gcn_dimensions, self.second_gcn_dimensions)))

        self.SOPOOL = nn.Sequential(OrderedDict([
            ("Linear_1", nn.Linear(self.second_gcn_dimensions, self.SOPOOL_dim_1)),
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

        self.fc1 = Linear(self.second_gcn_dimensions, self.output_dimensions)


    def forward(self, graph_batch):
        node_features_1 = F.relu(self.graph_conv_1(x=graph_batch.x, edge_index=graph_batch.edge_index))
        node_features_2 = F.relu(self.graph_conv_2(x=node_features_1, edge_index=graph_batch.edge_index))

        node_features_ = F.dropout(node_features_2, p=0.5, training=self.training)
        normalized_node_features = F.normalize(node_features_, dim=1)

        def sep_graph(node_features, ptr):
            graphs = []
            for i in range(len(ptr) - 1):
                temp1 = ptr[i]
                temp2 = ptr[i + 1]
                graphs.append(node_features[temp1:temp2])
            return (graphs)

        normalized_node_features = sep_graph(normalized_node_features, graph_batch.ptr)

        HH_tensor = torch.Tensor()

        for graph in normalized_node_features:
            graph = self.SOPOOL(graph)
            temp = torch.mm(graph.t(), graph).view(1, -1)
            if HH_tensor.shape == (0,):
                HH_tensor = temp
            else:
                HH_tensor = torch.cat((HH_tensor, temp), dim=0)

            # x = self.fc1(graph)
        output = F.dropout(self.MLP_1(HH_tensor), p=0.5, training=self.training)
        torch.cuda.empty_cache()

        return HH_tensor, output


class GAT(torch.nn.Module):
    def __init__(self, num_of_features, device):
        super(GAT, self).__init__()
        self.num_of_features = num_of_features
        self.first_gcn_dimensions = 128
        self.second_gcn_dimensions = 128
        self.heads = 8
        self.output_dimensions = 2
        self.SOPOOL_dim_1 = 32
        self.SOPOOL_dim_2 = 32
        self.linear_hidden_dimensions = 32

        self.device = device
        #GAT
        self.graph_conv_1 = GATConv(self.num_of_features, self.first_gcn_dimensions, heads=self.heads)
        self.graph_conv_2 = GATConv(self.first_gcn_dimensions * self.heads, self.second_gcn_dimensions, heads=self.heads)
        self.graph_conv_3 = GATConv(self.second_gcn_dimensions * self.heads, self.output_dimensions, heads=1)


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
        graph_batch = graph_batch.to(self.device)
        node_features_1 = F.relu(self.graph_conv_1(x=graph_batch.x, edge_index=graph_batch.edge_index))
        node_features_2 = F.relu(self.graph_conv_2(x=node_features_1, edge_index=graph_batch.edge_index))

        node_features_ = F.dropout(node_features_2, p=0.5, training=self.training)
        normalized_node_features = F.normalize(node_features_, dim=1)
        def sep_graph(node_features, ptr):
            graphs = []
            for i in range(len(ptr) - 1):
                temp1 = ptr[i]
                temp2 = ptr[i + 1]
                graphs.append(node_features[temp1:temp2])
            return (graphs)

        normalized_node_features = sep_graph(normalized_node_features, graph_batch.ptr)

        HH_tensor = torch.Tensor()

        for graph in normalized_node_features:
            graph = self.SOPOOL(graph)
            temp = torch.mm(graph.t(), graph).view(1, -1)
            if HH_tensor.shape == (0,):
                HH_tensor = temp
            else:
                HH_tensor = torch.cat((HH_tensor, temp), dim=0)

            # x = self.fc1(graph)
        output = F.dropout(self.MLP_1(HH_tensor), p=0.5, training=self.training)
        torch.cuda.empty_cache()

        return HH_tensor, output


def train_GNN(args, epoch, model, train_dataset, optimizer, criterion, device):
    total_loss = 0
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    for pos in pbar:
        indices = range(0, len(train_dataset), args.batch_size)

        for i in indices:
            model.train()
            graphs = train_dataset[i: i + args.batch_size]
            batch_graph = next(iter(DataLoader(graphs, batch_size=len(graphs))))
            _, out = model(batch_graph)
            labels = batch_graph.y.view(-1, ).to(device)

            if(criterion == "CrossEntropyLoss"):
                loss_function = nn.CrossEntropyLoss()
                loss = loss_function(out, labels)
            if(criterion == "nll_loss"):
                loss = F.binary_cross_entropy(out, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = loss.detach().cpu().numpy()

            loss_accum += loss
            pbar.set_description(f'epoch: {epoch}')
    print(loss_accum)
    average_loss = loss_accum / len(indices)
    print(f"Loss Training: {average_loss}")
    return average_loss


# Evaluation function
def test_GNN(args, model, train_dataset, test_dataset, criterion, device):
    model.eval()
    train_dataset = copy.deepcopy(train_dataset)
    test_dataset = copy.deepcopy(test_dataset)

    total_correct_train = 0
    for train_dataset_batch in iter(DataLoader(train_dataset, batch_size=args.batch_size)):
        _, output_train = model(train_dataset_batch)
        _, y_hat_train = torch.max(output_train, dim=1)
        labels_train = train_dataset_batch.y.view(-1).to(device)

        correct = torch.sum(y_hat_train == labels_train)
        total_correct_train += correct
    acc_train = total_correct_train / float(len(train_dataset))
    print(f'train (correct/samples) : ({total_correct_train}/{len(train_dataset)})')


    total_correct_test = 0
    for test_dataset_batch in iter(DataLoader(test_dataset, batch_size=args.batch_size)):
        _, output_test = model(test_dataset_batch)
        _, y_hat_test = torch.max(output_test, dim=1)
        labels_test = test_dataset_batch.y.view(-1,).to(device)
        # test_loss = criterion(output_test, labels_test)
        correct = torch.sum(y_hat_test == labels_test)
        total_correct_test += correct

    acc_test = total_correct_test / float(len(test_dataset))
    print(f'test (correct/samples): ({total_correct_test}/{len(test_dataset)})')

    print("accuracy (train/test): (%f/%f)" % (acc_train, acc_test))

    return acc_train, acc_test

def GNN_training_main(args, model, train_dataset, test_dataset, criterion):
    num_of_fold = 1
    acc_train_list = torch.zeros((num_of_fold,))
    acc_test_list = torch.zeros((num_of_fold,))
    best_epoch = 0
    max_acc_train = 0.0
    max_acc_test = 0.0


    # optimizer = torch.opti m.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_GNN(args,epoch, model, train_dataset, optimizer, criterion, device)
        acc_train, acc_test = test_GNN(args, model, train_dataset, test_dataset, criterion, device)

        # print info and save models
        max_acc_train = max(max_acc_train, acc_train)
        acc_train_list[0] = max_acc_train
        if (acc_test > max_acc_test):
            best_epoch = epoch
        max_acc_test = max(max_acc_test, acc_test)
        acc_test_list[0] = max_acc_test
        scheduler.step()
        print(f'Accuracy in epoch {epoch} (this acc / best acc in best epoch): ({acc_test} / {max_acc_test})')



if __name__ == '__main__':
    args = parameter_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = args.dataset
    multi_site = args.multi_site
    if (multi_site == False):
        if (dataset_name == "ABIDE"):
            dataset = read_dataset()
        elif (dataset_name == "BSNIP"):
            dataset = read_Schi_dataset()
        elif (dataset_name == "UCLA"):
            dataset = read_UCLA_dataset()
        feature_num = dataset[0].num_node_features

        train_dataset, test_dataset = separate_data(dataset, 42, 0)
    elif(multi_site == True):
        dataset_BSNIP = read_Schi_dataset()
        dataset_UCLA = read_UCLA_dataset()
        train_dataset_BSNIP, test_dataset_BSNIP = separate_data(dataset_BSNIP, 42, 0)
        train_dataset_UCLA, test_dataset_UCLA = separate_data(dataset_UCLA, 42, 0)
        if (dataset_name == "BSNIP"):
            train_dataset = train_dataset_BSNIP
            test_dataset = test_dataset_UCLA
        elif (dataset_name == "UCLA"):
            train_dataset = train_dataset_UCLA
            test_dataset = test_dataset_BSNIP
        feature_num = dataset_BSNIP[0].num_node_features
    model = args.model

    if(model == "GIN"):
        # train_dataset, test_dataset = separate_data(dataset, 42, 0)
        model = GIN(feature_num, device)
        criterion = "CrossEntropyLoss"
        GNN_training_main(args, model, train_dataset, test_dataset, criterion)

    elif(model == "GAT"):
        # train_dataset, test_dataset = separate_data(dataset, 42, 0)
        model = GAT(feature_num, device)
        criterion = "CrossEntropyLoss"
        GNN_training_main(args, model, train_dataset, test_dataset, criterion)
















