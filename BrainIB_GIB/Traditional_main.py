import copy

import wandb
import argparse
import shap
import networkx as nx
import joblib
import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
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
    parser.add_argument('--model', type=str, default="AdaBoost",
                        help='input the traditional model name for training (default: SVM), all choice: SVM, Decision Tree, KNN, AdaBoost')
    parser.add_argument('--multisite', type=str, default=True,
                        help='The training is multisite or not(default: False)')
    parser.add_argument('--dataset_name', type=str, default="UCLA",
                        help='The dataset use for training (default: BSNIP), others: UCLA, ABIDE')
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

def SVM_training(args, X_train, X_test, y_train, y_test, feature_names, target_names):
    batch_size = args.batch_size
    epochs = args.epochs
    best_acc = 0
    best_model = None
    # Initialize progress bar
    # Initialize SVM classifier with Stochastic Gradient Descent (SGD)
    svm_classifier = SGDClassifier(loss='log', alpha=0.001, max_iter=1000, tol=1e-3, random_state=42)
    # Standardize features

    # 对训练数据进行标准化（只在训练集上进行fit）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 在训练集上fit
    X_test_scaled = scaler.transform(X_test)  # 在测试集上transform

    for epoch in range(epochs):
        pbar_epoch = tqdm(total=len(X_train), desc=f'Epoch {epoch + 1}/{epochs}', unit=' samples')

        # 以批次为单位进行训练
        for batch_start in range(0, len(X_train_scaled), batch_size):
            batch_end = min(batch_start + batch_size, len(X_train_scaled))
            X_batch = X_train_scaled[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            svm_classifier.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
            pbar_epoch.update(len(X_batch))

        # 计算训练集的预测结果并计算准确率
        train_pred = svm_classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)

        # Predict the labels for the test set after each epoch
        y_pred = svm_classifier.predict(X_test_scaled)        # Calculate accuracy after each epoch
        accuracy = accuracy_score(y_test, y_pred)
        # best_acc = max(best_acc, accuracy)

        # Update the best model if the current model is better
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = svm_classifier

        # Update progress bar description with accuracy
        # progress_bar.set_postfix(loss=svm_classifier.loss, accuracy=f'{accuracy:.4f}')
        # Update progress bar description with accuracy
        pbar_epoch.set_postfix(loss=svm_classifier.loss, accuracy=f'{accuracy:.4f}')
        pbar_epoch.close()

        print(f"SVM Best_Acc:'{best_acc}', Accuracy:'{accuracy}'")
        # Close progress bar

    # Save the best model
    if best_model is not None:
        joblib.dump(best_model, 'best_svm_model.pkl')
        print("Best model saved to 'best_svm_model.pkl'")

    # Reduce the background dataset using shap.kmeans
    background = shap.kmeans(X_train_scaled, 10)  # Use 10 clusters as background
    explainer = shap.KernelExplainer(best_model.predict_proba, background)
    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)
    # Ensure JavaScript is enabled for visualizations
    shap.initjs()
    # Adjust figure size for summary plot
    plt.figure(figsize=(16, 16))  # Set figure size (width, height)
    # Plot SHAP values for a single prediction (first test sample)
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test_scaled[0], feature_names=feature_names)
    # Summary plot of SHAP values for all predictions
    plt.figure(figsize=(16, 16))  # Set figure size (width, height)
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, class_names=target_names)
    # Dependency plot for a single feature
    # plt.figure(figsize=(16, 16))  # Set figure size (width, height)
    # shap.dependence_plot(0, shap_values[0], X_test_scaled, feature_names=feature_names)
    plt.savefig("figs/SHAP_values.pdf", format="PDF", dpi=300)

    # Show the plots
    plt.show()





def Tree_training(args, X_train, X_test, y_train, y_test, feature_names, target_names):
    batch_size = args.batch_size
    epochs = args.epochs
    # Initialize Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    # Fit the classifier on the training data
    dt_classifier.fit(X_train, y_train)
    # Predict the labels for the test set after each epoch
    y_pred = dt_classifier.predict(X_test)

    # Calculate accuracy after each epoch
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Tree: Accuracy:'{accuracy}'")

    # 可视化决策树
    plt.figure(figsize=(60, 12))
    plot_tree(dt_classifier, filled=True, feature_names=feature_names, class_names=target_names, fontsize=10, max_depth=5)
    plt.savefig('figs/Traditional/decision_tree_plot.pdf', dpi=300)  # Set DPI to 300 for higher resolution
    plt.show()

def KNN_training(args, X_train, X_test, y_train, y_test):
    # Initialize KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Fit the classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("KNN Accuracy:", accuracy)


def AdaBoost_training(args, X_train, X_test, y_train, y_test):
    batch_size = args.batch_size
    epochs = args.epochs
    best_acc = 0
    # Initialize AdaBoost classifier with Decision Trees as base classifiers
    ada_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (decision stump)
            n_estimators=50,  # Number of weak learners
            random_state=42
        )

    # Number of estimators (number of weak learners)
    num_estimators = ada_classifier.n_estimators

    # Initialize progress bar
    pbar = tqdm(total=num_estimators, desc='Training Progress')

    # Train the classifier and update progress bar
    for _ in range(num_estimators):
        # Fit the classifier on the entire training data
        ada_classifier.fit(X_train, y_train)

        # Update progress bar
        pbar.update(1)

    # Predict the labels for the test set after each epoch
    y_pred = ada_classifier.predict(X_test)

    # Calculate accuracy after each epoch
    accuracy = accuracy_score(y_test, y_pred)
    best_acc = max(accuracy, best_acc)
    # Close progress bar
    pbar.close()
    print(f"AdaBoost Accuracy: {accuracy:.4f} - Best Acc:  {best_acc:.4f}")

def SHAP_data(dataset):
    shap_dataset = [Data(node_attr=data.x, edge_index=data.edge_index, y=data.y) for data in dataset]

    for i in range(len(dataset)):
        # for i in range(200):
        # Set your threshold value
        threshold = 0.4
        # Apply the threshold
        data = dataset[i]
        number_of_nodes = data.num_nodes
        adj = (data.x > threshold).int()  # Converts boolean mask to int (1 and 0)
        edges = torch.nonzero(adj)
        # Convert binary tensor to NetworkX graph
        G = nx.Graph()
        for edge in edges:
            a, b = edge.tolist()
            G.add_edge(a, b)

        # LOCALLY SHAP VALUES
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G)
        # Calculate betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(G)
        # Calculate closeness centrality
        closeness_centrality = nx.closeness_centrality(G)
        load_centrality = nx.load_centrality(G)
        # Calculate shortest path lengths from each node to every other node
        shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        # Calculate the degree of each node
        degree_dict = dict(G.degree())
        # Calculate the average neighbor degree of each node
        avg_neighbor_degree_dict = nx.average_neighbor_degree(G)

        # Assuming the node indices range from 0 to 104
        degree_centrality_values = torch.tensor([degree_centrality.get(i, 0) for i in range(105)])
        betweenness_centrality_values = torch.tensor([betweenness_centrality.get(i, 0) for i in range(105)])
        closeness_centrality_values = torch.tensor([closeness_centrality.get(i, 0) for i in range(105)])
        load_centrality_values = torch.tensor([load_centrality.get(i, 0) for i in range(105)])
        degree_values = torch.tensor([degree_dict.get(i, 0) for i in range(105)])
        avg_neighbor_degree_values = torch.tensor([avg_neighbor_degree_dict.get(i, 0) for i in range(105)])

        # Convert shortest path lengths to a tensor
        shortest_path_lengths_values = torch.zeros((105, 105), dtype=torch.float)
        for m in range(number_of_nodes):
            for n in range(number_of_nodes):
                if n in shortest_path_lengths[m]:
                    shortest_path_lengths_values[m, n] = shortest_path_lengths[m][n]
        average_shortest_path_lengths_values = torch.mean(shortest_path_lengths_values, dim=0)

        # concat node shape values
        edges_values = torch.sum(adj, dim=1)
        shap_values = edges_values
        shap_values = torch.cat((shap_values, degree_centrality_values), dim=-1)
        shap_values = torch.cat((shap_values, betweenness_centrality_values), dim=-1)
        shap_values = torch.cat((shap_values, closeness_centrality_values), dim=-1)
        shap_values = torch.cat((shap_values, load_centrality_values), dim=-1)
        shap_values = torch.cat((shap_values, average_shortest_path_lengths_values), dim=-1)
        shap_values = torch.cat((shap_values, degree_values), dim=-1)
        shap_values = torch.cat((shap_values, avg_neighbor_degree_values), dim=-1)

        shap_dataset[i].x = shap_values.reshape(-1, 105)

        # GLOBAL SHAP VALUES
        Average_Shortest_Path_lengths = average_shortest_path_lengths_values.mean().item()
        Average_Degree = edges.shape[0] / number_of_nodes

        other_values = torch.tensor([Average_Shortest_Path_lengths, Average_Degree])
        other_values_names = ["Average_Shortest_Path_lengths", "Average_Degree"]
        shap_dataset[i].global_values = other_values
        if (i % 50 == 1):
            print(i)

    X, y = adapt_dataset(shap_dataset)

    feature_names = []
    for i in range(dataset[0].num_nodes):
        names = f"{i}_edge", f"{i}_degree_centrality", f"{i}_betweenness_centrality", f"{i}_closeness_centrality", f"{i}_load_centrality", f"{i}_avg_shortest_lengths_path", f"{i}_degree", f"{i}_avg_neighbor_degree"
        feature_names.extend(names)
    feature_names.extend(other_values_names)

    target_names = np.array(['normal control', 'patients'])

    return X, y, feature_names, target_names

def normal_shap_fit(dataset):
    if (args.training == "normal"):
        X, y = adapt_dataset(dataset)
        # 生成矩阵上所有单元格的特征名称
        feature_names = []
        for i in range(feature_num):  # 行数
            for j in range(feature_num):  # 列数
                feature_name = f"{i}_{j}"
                feature_names.append(feature_name)
        target_names = np.array(['normal control', 'patients'])
    if (args.training == "SHAP"):
        # Split the dataset into training and testing sets
        X, y, feature_names, target_names = SHAP_data(dataset)

    # train_graph_list, test_graph_list = separate_data(dataset, 42, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, feature_names, target_names




if __name__ == '__main__':
    args = parameter_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    multisite = args.multisite
    dataset_name = args.dataset_name



    if(multisite==False):
        if(dataset_name =="ABIDE"):
            dataset = read_dataset()
        elif(dataset_name =="BSNIP"):
            dataset = read_Schi_dataset()
        elif(dataset_name =="UCLA"):
            dataset = read_UCLA_dataset()
        feature_num = dataset[0].num_node_features
        X_train, X_test, y_train, y_test, feature_names, target_names = normal_shap_fit(dataset)

    elif(multisite==True):
        dataset_BSNIP = read_Schi_dataset()
        feature_num = dataset_BSNIP[0].num_node_features
        X_train_BSNIP, X_test_BSNIP, y_train_BSNIP, y_test_BSNIP, feature_names_BSNIP, target_names_BSNIP = normal_shap_fit(dataset_BSNIP)

        dataset_UCLA = read_UCLA_dataset()
        X_train_UCLA, X_test_UCLA, y_train_UCLA, y_test_UCLA, feature_names_UCLA, target_names_UCLA = normal_shap_fit(dataset_UCLA)


    model = args.model

    if(model=="SVM"):
        if(multisite==False):
            SVM_training(args, X_train, X_test, y_train, y_test, feature_names, target_names)
        elif(multisite==True):
            if (dataset_name == "BSNIP"):
                SVM_training(args, X_train_BSNIP, X_test_UCLA, y_train_BSNIP, y_test_UCLA, feature_names_BSNIP, target_names_BSNIP)
            elif (dataset_name == "UCLA"):
                SVM_training(args, X_train_UCLA, X_test_BSNIP, y_train_UCLA, y_test_BSNIP, feature_names_BSNIP, target_names_BSNIP)
    elif(model=="Decision Tree"):
        if (multisite == False):
            Tree_training(args, X_train, X_test, y_train, y_test, feature_names, target_names)
        elif (multisite == True):
            if (dataset_name == "BSNIP"):
                Tree_training(args, X_train_BSNIP, X_test_UCLA, y_train_BSNIP, y_test_UCLA, feature_names_BSNIP, target_names_BSNIP)
            elif (dataset_name == "UCLA"):
                Tree_training(args, X_train_UCLA, X_test_BSNIP, y_train_UCLA, y_test_BSNIP, feature_names_BSNIP, target_names_BSNIP)

    elif(model == "KNN"):
        if (multisite == False):
            KNN_training(args, X_train, X_test, y_train, y_test)
        elif (multisite == True):
            if (dataset_name == "BSNIP"):
                KNN_training(args, X_train_BSNIP, X_test_UCLA, y_train_BSNIP, y_test_UCLA)
            elif (dataset_name == "UCLA"):
                KNN_training(args, X_train_UCLA, X_test_BSNIP, y_train_UCLA, y_test_BSNIP)
    elif(model=="AdaBoost"):
        if (multisite == False):
            AdaBoost_training(args, X_train, X_test, y_train, y_test)
        elif(multisite == True):
            if (dataset_name == "BSNIP"):
                AdaBoost_training(args, X_train_BSNIP, X_test_UCLA, y_train_BSNIP, y_test_UCLA)
            elif (dataset_name == "UCLA"):
                AdaBoost_training(args, X_train_UCLA, X_test_BSNIP, y_train_UCLA, y_test_BSNIP)
















