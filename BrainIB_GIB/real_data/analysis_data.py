import os
import scipy.io as scio
from scipy.sparse import coo_matrix
from scipy.io import loadmat
from create_dataset import read_Schi_dataset
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path
import scipy.io
def get_nc_and_sz(dataset, thresholds_for_adj):
    sz = torch.zeros(105, 105)
    nc = torch.zeros(105, 105)
    for i in range(len(dataset)):
        if(dataset[i].y.item() == 1):
            # patients
            abs_data = abs(dataset[i].x)
            abs_data[abs_data < thresholds_for_adj] = 0
            abs_data[abs_data == thresholds_for_adj] = 1
            abs_data[abs_data > thresholds_for_adj] = 1
            sz = sz + abs_data
            # sz_num = sz_num  + 1
        if(dataset[i].y.item() == 0):
            # normal control
            abs_data = abs(dataset[i].x)
            abs_data[abs_data < thresholds_for_adj] = 0
            abs_data[abs_data == thresholds_for_adj] = 1
            abs_data[abs_data > thresholds_for_adj] = 1
            nc = nc + abs_data
    sz_num = sz[0][0].item()
    nc_num = nc[0][0].item()
    # then normalization all the sum number
    # sz_norm = torch.nn.functional.normalize(sz, p=2.0, dim=1, eps=1e-12, out=None)
    # nc_norm = torch.nn.functional.normalize(nc, p=2.0, dim=1, eps=1e-12, out=None)
    sz_norm = sz / sz_num
    nc_norm = nc / nc_num
    return sz_norm, nc_norm



def save_positive_nc_ans_sz(sz_norm, nc_norm):
    # count some real_data for the result tensor
    # Count non-zero elements along rows (dimension 1)
    sz_norm[sz_norm < 0] = 0
    sz_norm[sz_norm == 1] = 0
    nc_norm[nc_norm < 0] = 0
    nc_norm[nc_norm == 1] = 0

    # Specify the file path
    sz_norm_array = sz_norm.numpy()
    file_path = "sz_norm.pth"  # Change to your desired file path
    # Save the tensor to file
    np.savetxt("sz_norm_data.csv", sz_norm_array, delimiter=",", fmt='%.3f')

    nc_norm_array = nc_norm.numpy()
    file_path = "nc_norm.pth"  # Change to your desired file path
    # Save the tensor to file
    np.savetxt("nc_norm_data.csv", nc_norm_array, delimiter=",", fmt='%.3f')
    # Specify the desired CSV file path
def graw_network(sz_norm, save_path, draw):
    # Assuming 'adjacency_matrix' is your tensor adjacency matrix
    adjacency_matrix = sz_norm  # Example random adjacency matrix
    # Create an empty graph
    G = nx.Graph()
    # Add nodes with labels as indices
    for i in range(adjacency_matrix.shape[0]):
        G.add_node(i)
    # Add edges based on the adjacency matrix and their weights
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            weight = adjacency_matrix[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight.item())  # Convert weight tensor to scalar
    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(G)
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(G)

    # Print degree centrality for each node
    # for node, centrality in degree_centrality.items():
    #     print(f"Node {node}: Degree Centrality = {centrality}")
    if(draw == True):
        # Draw the graph with edges weighted by the matrix values
        # Manually specify positions for each node
        fixed_positions = {node: (node % 10, node // 10) for node in G.nodes()}  # Example positions, adjust as needed
         # Draw the graph with fixed node positions
        pos = fixed_positions
        weights = [edata['weight'] for _, _, edata in G.edges(data=True)]
        # Set figure size for higher resolution
        plt.figure(figsize=(50, 50))
        nx.draw(G, pos, with_labels=True, labels={node: node for node in G.nodes()}, node_color='skyblue', node_size=20,
                edge_color='k', width=weights, font_size=15, font_color='skyblue')
        # Increase spacing between nodes for better readability
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # Save the plot as a high-resolution image
        plt.savefig(save_path, dpi=300)
        # plt.show()
    return degree_centrality, betweenness_centrality, closeness_centrality


    # Plot the heatmap
def draw_heatmap(sz_norm, threshold, file_path):
    sz_norm[sz_norm < threshold] = 0
    # sz_norm[sz_norm == 1] = 0
    # 创建自定义颜色映射
    colors = [(0, 'yellow'), (0.5, 'red'), (1, 'red')]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_red_yellow', colors)

    plt.imshow(sz_norm, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()   # Add color bar to show the scale
    # Set tick labels
    # plt.xticks(np.arange(0, 105, 5), np.arange(0, 105, 5))
    plt.xticks(np.arange(0, 105, 10), np.arange(0, 105, 10))

    plt.yticks(np.arange(0, 105, 5), np.arange(0, 105, 5))
    # plt.yticks(np.arange(0, 105, 10), np.arange(0, 105, 10))
    # Add grid
    plt.grid(True, color='white')
    plt.savefig(file_path)
    plt.show()

def draw_heat_map_directly(save_path, norm_array):
    plt.figure(figsize=(24, 18))  # Adjust the width and height as needed
    # Create the heatmap
    plt.imshow(norm_array, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add color bar to show the mapping of values to colors
    # Add index labels along the axes
    plt.xticks(np.arange(len(norm_array)), labels=np.arange(len(norm_array)))  # Set x-axis tick labels
    plt.yticks(np.arange(len(norm_array)), labels=np.arange(len(norm_array)))  # Set y-axis tick labels
    plt.savefig(save_path, dpi = 300)
    plt.show()

def draw_cluster_directly(save_path, norm_array, cluster_1):
    plt.figure(figsize=(24, 18))  # Adjust the width and height as needed
    # Create the heatmap
    plt.imshow(norm_array, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add color bar to show the mapping of values to colors
    # Add index labels along the axes
    plt.xticks(np.arange(len(norm_array)), labels=cluster_1)  # Set x-axis tick labels
    plt.yticks(np.arange(len(norm_array)), labels=cluster_1)  # Set y-axis tick labels
    plt.savefig(save_path, dpi = 300)
    plt.show()
def compare_dict(dict1, dict2):
    """Compare two dictionaries and return keys where the values are different."""
    different = []
    keys = []
    for key in dict1.keys() | dict2.keys():
        if dict1.get(key) != dict2.get(key):
            different.append(key)
    return different
def save_centrality(thresholds, sz_degree_centrality, sz_betweenness_centrality, sz_closeness_centrality, nc_degree_centrality, nc_betweenness_centrality, nc_closeness_centrality):
    # Specify the file path
    file_path = str(thresholds) + "sz_and_nc_degree_centrality.csv"  # Change to your desired file path

    with open(file_path, 'w') as file:
        concatenated_tensor = torch.cat((sz_degree_centrality, nc_degree_centrality), dim=0).reshape(2, 105).numpy()
        df = pd.DataFrame(concatenated_tensor).T# 在行维度上连接
        df.to_csv(file_path, index=False)

    file_path = str(thresholds) + "sz_and_nc_betweenness_centrality.csv"  # Change to your desired file path
    # Open the file in write mode and write the list elements to the file
    with open(file_path, 'w') as file:
        concatenated_tensor = torch.cat((sz_betweenness_centrality, nc_betweenness_centrality), dim=0).reshape(2, 105).numpy()
        df = pd.DataFrame(concatenated_tensor).T# 在行维度上连接
        df.to_csv(file_path, index=False)

    file_path = str(thresholds) + "sz_and_nc_closeness_centrality.csv"  # Change to your desired file path
    # Open the file in write mode and write the list elements to the file
    with open(file_path, 'w') as file:
        concatenated_tensor = torch.cat((sz_closeness_centrality, nc_closeness_centrality), dim=0).reshape(2, 105).numpy()
        df = pd.DataFrame(concatenated_tensor).T# 在行维度上连接
        df.to_csv(file_path, index=False)
def get_adj(edges_tensor):
    # Initialize an empty adjacency matrix with shape (num_nodes, num_nodes)
    adj_matrix = np.zeros((105, 105))

    # Iterate over each edge in the tensor and update the adjacency matrix
    for i in range(edges_tensor.shape[1]):
        src_node = edges_tensor[0, i]
        tgt_node = edges_tensor[1, i]
        adj_matrix[src_node, tgt_node] = 1  # Assuming it's a binary adjacency matrix (1 if edge exists, 0 otherwise)

    return torch.tensor(adj_matrix)
def get_centrality_collects(dataset, thresholds):
    sz_num = 0
    nc_num = 0
    draw_pic = False
    sz_degree_centrality_collect = torch.zeros(105)
    sz_betweenness_centrality_collect = torch.zeros(105)
    sz_closeness_centrality_collect = torch.zeros(105)

    nc_degree_centrality_collect = torch.zeros(105)
    nc_betweenness_centrality_collect = torch.zeros(105)
    nc_closeness_centrality_collect = torch.zeros(105)
    save_path = "../figs/nodes_connections/sz_larger_than_" + str(thresholds) + ".png"

    for i in range(len(dataset)):
        if (i % 100 == 0): print(i)
        if (dataset[i].y.item() == 1):
            # patients
            abs_data = get_adj(dataset[i].edge_index)
            sz_degree_centrality, sz_betweenness_centrality, sz_closeness_centrality = graw_network(abs_data, save_path,
                                                                                                    draw_pic)
            sz_degree_centrality_collect = sz_degree_centrality_collect + torch.tensor(
                list(sz_degree_centrality.values()))
            sz_betweenness_centrality_collect = sz_betweenness_centrality_collect + torch.tensor(
                list(sz_betweenness_centrality.values()))
            sz_closeness_centrality_collect = sz_closeness_centrality_collect + torch.tensor(
                list(sz_closeness_centrality.values()))
            sz_num = sz_num + 1
        if (dataset[i].y.item() == 0):
            # normal control
            abs_data = get_adj(dataset[i].edge_index)
            nc_degree_centrality, nc_betweenness_centrality, nc_closeness_centrality = graw_network(abs_data, save_path,
                                                                                                    draw_pic)
            nc_degree_centrality_collect = nc_degree_centrality_collect + torch.tensor(
                list(nc_degree_centrality.values()))
            nc_betweenness_centrality_collect = nc_betweenness_centrality_collect + torch.tensor(
                list(nc_betweenness_centrality.values()))
            nc_closeness_centrality_collect = nc_closeness_centrality_collect + torch.tensor(
                list(nc_closeness_centrality.values()))
            nc_num = nc_num + 1
    return sz_degree_centrality_collect, sz_betweenness_centrality_collect, sz_closeness_centrality_collect, nc_degree_centrality_collect, nc_betweenness_centrality_collect, nc_closeness_centrality_collect
def get_star_node(target):
    sz_num = 0
    nc_num = 0
    sz_adj_matrix = torch.zeros(105, 105)
    nc_adj_matrix = torch.zeros(105, 105)
    for i in range(len(dataset)):
        # if (i % 100 == 0): print(i)
        if (dataset[i].y.item() == 1):
            # patients
            edge_index = dataset[i].edge_index
            # 找到所有包含数字 4 的列的索引
            indices = torch.where(edge_index == target)[1]
            # 使用索引提取这些列，并生成一个新的 tensor
            edges = edge_index[:, indices]
            # Get the number of nodes
            num_nodes = 105
            # Create the adjacency matrix
            adj_matrix_dense = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (num_nodes, num_nodes))
            sz_adj_matrix = sz_adj_matrix + adj_matrix_dense
            sz_num = sz_num + 1
        if (dataset[i].y.item() == 0):
            # normal control
            edge_index = dataset[i].edge_index
            # 找到所有包含数字 4 的列的索引
            indices = torch.where(edge_index == target)[1]
            # 使用索引提取这些列，并生成一个新的 tensor
            edges = edge_index[:, indices]
            # Get the number of nodes
            num_nodes = 105
            # Create the adjacency matrix
            adj_matrix_dense = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (num_nodes, num_nodes))
            nc_adj_matrix = nc_adj_matrix + adj_matrix_dense
            nc_num = nc_num + 1

    sz_adj_norm = sz_adj_matrix / sz_num
    nc_adj_norm = nc_adj_matrix / nc_num
    return sz_adj_norm, nc_adj_norm
def draw_node_central_graph(sz_adj_norm, fig_path):
    # Create an empty NetworkX graph
    G = nx.Graph()
    # Add edges with weights
    for i in range(sz_adj_norm.shape[0]):
        for j in range(i + 1, sz_adj_norm.shape[1]):
            if sz_adj_norm[i][j] != 0:
                G.add_edge(i, j, weight=sz_adj_norm[i][j])
    # Draw the graph
    # 生成边的粗细列表，根据权重值设置
    edge_widths = [d['weight'] * 10 for u, v, d in G.edges(data=True)]
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=50,
            width=edge_widths, edge_color='k', linewidths=1, font_size=10, font_color = "red", arrows=True)
    # Show the graph
    plt.savefig(fig_path)
    plt.show()

def get_target_nodes_network(target_nodes):
    sz_num = 0
    nc_num = 0
    sz_adj_matrix = torch.zeros(105, 105)
    nc_adj_matrix = torch.zeros(105, 105)
    for i in range(len(dataset)):
        # if (i % 100 == 0): print(i)
        if (dataset[i].y.item() == 1):
            # patients
            edge_index = dataset[i].edge_index
            num_nodes = 105
            # Create the adjacency matrix
            adj_matrix_dense = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes))
            sz_adj_matrix = sz_adj_matrix + adj_matrix_dense
            sz_num = sz_num + 1
        if (dataset[i].y.item() == 0):
            # normal control
            edge_index = dataset[i].edge_index
            num_nodes = 105
            # Create the adjacency matrix
            adj_matrix_dense = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes))
            nc_adj_matrix = nc_adj_matrix + adj_matrix_dense
            nc_num = nc_num + 1

    # filter all target nodes
    # 生成一个 105x105 的全 False 的 mask
    mask = torch.zeros(105, 105, dtype=torch.bool)
    # 指定要设置为 True 的行和列
    # 将特定的行和列设置为 True
    mask[target_nodes, :] = True
    mask[:, target_nodes] = True
    # mask[target_nodes, target_nodes] = True
    sz_adj_matrix = sz_adj_matrix * mask
    nc_adj_matrix = nc_adj_matrix * mask

    sz_adj_norm = sz_adj_matrix / sz_num
    nc_adj_norm = nc_adj_matrix / nc_num

    return sz_adj_norm, nc_adj_norm

def get_cluster_network(cluster_name):
    sz_num = 0
    nc_num = 0
    sz_adj_matrix = torch.zeros(105, 105)
    nc_adj_matrix = torch.zeros(105, 105)
    for i in range(len(dataset)):
        # if (i % 100 == 0): print(i)
        if (dataset[i].y.item() == 1):
            # patients
            edge_index = dataset[i].edge_index
            num_nodes = 105
            # Create the adjacency matrix
            adj_matrix_dense = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes))
            sz_adj_matrix = sz_adj_matrix + adj_matrix_dense
            sz_num = sz_num + 1
        if (dataset[i].y.item() == 0):
            # normal control
            edge_index = dataset[i].edge_index
            num_nodes = 105
            # Create the adjacency matrix
            adj_matrix_dense = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes))
            nc_adj_matrix = nc_adj_matrix + adj_matrix_dense
            nc_num = nc_num + 1

    # filter all target nodes
    # 生成一个 105x105 的全 False 的 mask
    mask = torch.zeros(105, 105, dtype=torch.bool)
    # 指定要设置为 True 的行和列
    # 将特定的行和列设置为 True
    # Filter the target nodes from the larger matrix
    sz_adj_matrix = sz_adj_matrix[cluster_name][:, cluster_name]
    nc_adj_matrix = nc_adj_matrix[cluster_name][:, cluster_name]

    sz_adj_norm = sz_adj_matrix / sz_num
    nc_adj_norm = nc_adj_matrix / nc_num
    return sz_adj_norm, nc_adj_norm

def draw_target_nodes_network(sz_adj_norm, fig_path):
    G = nx.Graph()
    # Add edges with weights
    for i in range(sz_adj_norm.shape[0]):
        for j in range(i + 1, sz_adj_norm.shape[1]):
            if sz_adj_norm[i][j] != 0:
                G.add_edge(i, j, weight=sz_adj_norm[i][j])
    # Draw the graph
    # 生成边的粗细列表，根据权重值设置
    edge_widths = [d['weight'] * 2 for u, v, d in G.edges(data=True)]
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=50,
            width=edge_widths, edge_color='k', linewidths=1, font_size=10, font_color = "red", arrows=True)
    # Show the graph
    plt.savefig(fig_path)
    plt.show()
def target_nodes_network(target_nodes, bar_list):
    for i in range(len(bar_list)):
        bar = bar_list[i]
        sz_adj_norm, nc_adj_norm = get_target_nodes_network(target_nodes)
        # 计算两个矩阵之间的差值矩阵
        diff_matrix = torch.abs(sz_adj_norm - nc_adj_norm)
        # 找到差值矩阵中的最大值及其索引坐标
        max_value, max_indices = torch.max(diff_matrix.view(-1), dim=0)
        max_row = max_indices // diff_matrix.shape[1]
        max_col = max_indices % diff_matrix.shape[1]
        print(bar, max_value.item(), (max_row.item(), max_col.item()))

        sz_adj_norm[sz_adj_norm < bar] = 0
        nc_adj_norm[nc_adj_norm < bar] = 0

        fig_path = "../figs/target_nodes_network/sz"+str(bar)+".png"
        draw_target_nodes_network(sz_adj_norm, fig_path)
        fig_path = "../figs/target_nodes_network/nc"+str(bar)+".png"
        draw_target_nodes_network(nc_adj_norm, fig_path)

def star_node_network(target_nodes):
    for i in range(len(target_nodes)):
        target = target_nodes[i]
        sz_adj_norm, nc_adj_norm = get_star_node(target)
        bar = 0.1
        sz_adj_norm[sz_adj_norm < bar] = 0
        nc_adj_norm[nc_adj_norm < bar] = 0

        fig_path = "../figs/nodes_stars/" + str(target) + "_sz.png"
        draw_node_central_graph(sz_adj_norm, fig_path)
        fig_path = "../figs/nodes_stars/" + str(target) + "_nc.png"
        draw_node_central_graph(nc_adj_norm, fig_path)
        diff = sz_adj_norm[target] - nc_adj_norm[target]
        print(target, abs(diff).max())

def draw_cricle_graph(types, values, names, indices, sz_norm, edge_thresd, path):
    color_map = mpl.colormaps['Set2']
    # sz_norm[sz_norm < 0.8] = 0
    adjacency_matrix = sz_norm.numpy()
    np.fill_diagonal(adjacency_matrix, 0)

    # Create a graph
    G = nx.Graph()
    # Add nodes with attributes from the lists
    for node_type, value, name, index in zip(types, values, names, indices):
        G.add_node(name, type=node_type, value=value, index=index)
    # Add edges from the adjacency matrix

    for i, row in enumerate(adjacency_matrix):
        for j, weight in enumerate(row):
            if weight > edge_thresd:
                zoom_weight = (weight) * 3# Assuming a threshold for edge creation, change as needed
                G.add_edge(names[i], names[j], weight= zoom_weight)
            # else:
            #     G.add_edge(names[i], names[j], weight=0)

    pos = nx.circular_layout(G)
    # Draw nodes with different colors based on type, using Set3 colormap
    node_colors = [color_map(G.nodes[node]['type'] - 1) for node in G.nodes]  # -1 because types are 1-indexed
    node_sizes = [value * 800 for value in values]  # Scale values for visualization
    plt.figure(figsize=(30, 30))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    # Draw edges

    edge_widths = [G[u][v]['weight'] * 1 for u, v in G.edges()]  # Multiplying by 2 for better visibility
    edge_color = [color_map(G.nodes[node]['type'] - 1) for node in G.nodes]
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_widths)

    # Draw node labels radiating out from the center
    label_factor = 1.08  # Adjust this value as needed to move labels further away from nodes
    for node, (x, y) in pos.items():
        angle = np.arctan2(y, x) * (180 / np.pi)  # Calculate angle (in degrees)
        rotation = (angle + 360) % 360  # Ensure rotation is between 0-360 degrees
        alignment = 'right' if 90 < rotation <= 270 else 'left'
        x, y = x * label_factor, y * label_factor  # Move labels outwards
        color = color_map(G.nodes[node]['type'] - 1)  # -1 because types are 1-indexed
        plt.text(x, y, node, fontsize=25, rotation=rotation, ha=alignment, va='center',
                 color=color, bbox=dict(facecolor='none', edgecolor='none', pad=0))

    # Show the plot
    plt.axis('off')
    plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    dataset = read_Schi_dataset()
    print(len(dataset))
    thresholds_for_adj = 0.4
    sz_norm, nc_norm = get_nc_and_sz(dataset, thresholds_for_adj)
    thresholds = 0
    draw_heatmap(sz_norm, thresholds, "../figs/sz_norm_"+str(thresholds)+".png")
    draw_heatmap(nc_norm, thresholds, "../figs/nc_norm_"+str(thresholds)+".png")



    # Assuming 'tensor' is your tensor
    scipy.io.savemat('sz_norm.mat', {'sz_norm': sz_norm})
    scipy.io.savemat('nc_norm.mat', {'nc_norm': nc_norm})

    # Assuming 'tensor' is your tensor
    sz_norm_np = sz_norm.numpy()  # Convert PyTorch tensor to NumPy array
    nc_norm_np = nc_norm.numpy()
    np.savetxt('sz_norm_full.csv', sz_norm_np, delimiter=',', fmt='%.3f')
    np.savetxt('nc_norm_full.csv', nc_norm_np, delimiter=',', fmt='%.3f')

    sz_norm[sz_norm < 0.4] = 0
    sz_norm[sz_norm < 0.4] = 0
    sz_norm_np = sz_norm.numpy()  # Convert PyTorch tensor to NumPy array
    nc_norm_np = nc_norm.numpy()
    np.savetxt('sz_norm_04.csv', sz_norm_np, delimiter=',', fmt='%.3f')
    np.savetxt('nc_norm_04.csv', nc_norm_np, delimiter=',', fmt='%.3f')

    sz_norm[sz_norm < 0.8] = 0
    sz_norm[sz_norm < 0.8] = 0
    sz_norm_np = sz_norm.numpy()  # Convert PyTorch tensor to NumPy array
    nc_norm_np = nc_norm.numpy()
    np.savetxt('sz_norm_08.csv', sz_norm_np, delimiter=',', fmt='%.3f')
    np.savetxt('nc_norm_08.csv', nc_norm_np, delimiter=',', fmt='%.3f')


    # thresholds = 0.4
    # # sz_degree_centrality_collect, sz_betweenness_centrality_collect, sz_closeness_centrality_collect, nc_degree_centrality_collect, nc_betweenness_centrality_collect, nc_closeness_centrality_collect = get_centrality_collects(dataset, thresholds)
    #
    # # save_centrality(thresholds, sz_degree_centrality_collect, sz_betweenness_centrality_collect, sz_closeness_centrality_collect,
    # #                 nc_degree_centrality_collect, nc_betweenness_centrality_collect, nc_closeness_centrality_collect)
    #
    # target_nodes = [4, 17, 22, 29, 30, 32, 46, 55, 72, 90, 91, 93, 98]
    cluster_1 = [16, 17,19,28,29,56,57,58,59,62,63,68,69,79,80,81,83]
    cluster_2 = [42, 43,44,45,46,47,48,49,50,51,52,53,54,55]
    cluster_3 = [6, 7,8,9,10,11,12,13,14,15,18,60,61,64,65,82,88]
    cluster_4 = [0, 1,2,3,4,5,21,22,23,24,25,26,27,30,31,34,35,66,67,84,85,89]
    cluster_5 = [20,32,33,36,37,38,39,40,41,70,71,72,73,74,75,76,77,86,87]
    cluster_6 = [90,91,92,93,94,95,96,97,98,99,100,101,102,103,104]

    # # now get the target_node_central graph network
    # # star_node_network(target_nodes)
    # # get a network based on all target nodes, look how they are connect with each other
    # bar_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # target_nodes_network(bar_list)
    # print()
    all_nodes = []
    for i in range(105):
        all_nodes.append(i)
    name = "all_nodes"
    sz_adj_norm, nc_adj_norm = get_target_nodes_network(all_nodes)
    norm_array = sz_adj_norm - nc_adj_norm
    np.savetxt(name+"_adjacency_sz-nc.csv", norm_array, delimiter=",", fmt='%.4f')
    save_path = name+"_adjacency_sz-nc.pdf"
    draw_heat_map_directly(save_path, norm_array)
    # norm_array[-0.5<norm_array<0.5] = 0
    # Set the figure size

    # 这里的norm matrix is cluster_nodes to all nodes size (105,105)
    name = "cluster_1"
    sz_adj_norm, nc_adj_norm = get_target_nodes_network(cluster_1)
    norm_array = sz_adj_norm - nc_adj_norm
    np.savetxt(name+"_adjacency_sz-nc.csv", norm_array, delimiter=",", fmt='%.4f')
    save_path = name+"_adjacency_sz-nc.pdf"
    draw_heat_map_directly(save_path, norm_array)


    # here is going to filter the matrix only in cluster's shape
    name = "cluster_1_only"
    cluster_name = cluster_1
    sz_adj_norm, nc_adj_norm = get_cluster_network(cluster_name)
    norm_array = sz_adj_norm - nc_adj_norm
    np.savetxt(name+"_adjacency_sz-nc.csv", norm_array, delimiter=",", fmt='%.4f')
    save_path = name+"_adjacency_sz-nc.pdf"
    draw_cluster_directly(save_path, norm_array, cluster_name)

    name = "cluster_1_only"
    cluster_name = cluster_1
    sz_adj_norm, nc_adj_norm = get_cluster_network(cluster_name)
    np.savetxt(name+"_adjacency_sz_adj_norm.csv", sz_adj_norm, delimiter=",", fmt='%.4f')
    save_path = name+"_adjacency_sz_adj_norm.pdf"
    draw_cluster_directly(save_path, sz_adj_norm, cluster_name)
    np.savetxt(name+"_adjacency_nc_adj_norm.csv", sz_adj_norm, delimiter=",", fmt='%.4f')
    save_path = name+"_adjacency_nc_adj_norm.pdf"
    draw_cluster_directly(save_path, nc_adj_norm, cluster_name)