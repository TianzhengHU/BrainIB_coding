import torch

from BrainIB_V2.baseline_data.baseline_main import get_baseline_data, mutag_functional_groups_label
from SGSIB.GNN import GNN
from SGSIB.sub_node_generator import GIB
from SGSIB.sub_graph_generator import MLP_subgraph
from SGSIB_main import parameter_parser
from BrainIB_V2.real_data.create_dataset import read_dataset, read_Schi_dataset, read_UCLA_dataset
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data


def draw_only_sub(subgraph):
    node_mask_prob = subgraph['attr'].T[0]
    node_mask = torch.as_tensor((node_mask_prob - node_mask_prob.mean()) > 0, dtype=torch.int32)
    subgraph.attr = node_mask

    # Convert the Data object to a NetworkX graph, considering the 'attr' mask
    G = nx.Graph()
    G.add_nodes_from(range(subgraph.num_nodes))
    edge_list = subgraph.edge_index.t().tolist()

    # Filter edges to be drawn based on the 'attr' mask
    filtered_edge_list = [edge for edge in edge_list if subgraph.attr[edge[0]] == 1 and subgraph.attr[edge[1]] == 1]
    G.add_edges_from(filtered_edge_list)

    # Create a visualization of the graph, considering the 'attr' mask
    position = nx.spring_layout(G)
    # Filter nodes to be drawn based on the 'attr' mask
    # 只画出被选中的子图的点
    nodes_to_draw = [node for node, attr_value in enumerate(subgraph.attr) if attr_value == 1]

    labels = {i: str(i) for i in nodes_to_draw}

    plt.figure(figsize=(8, 8))
    nx.draw(G, position, nodelist=nodes_to_draw, with_labels=True, labels=labels, node_size=40, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
    plt.title("Graph Visualization")

    # Save the figure in high resolution
    # plt.savefig("graph_visualization_hd.png", format="PNG", dpi=300)

    plt.show()


def draw_compare(subgraph, i):
    node_mask_prob = subgraph['attr'].T[0]
    node_mask = torch.as_tensor((node_mask_prob - node_mask_prob.mean()) > 0, dtype=torch.int32)
    subgraph.attr = node_mask

    # Convert the Data object to a NetworkX graph, considering the 'attr' mask
    G = nx.Graph()
    G.add_nodes_from(range(subgraph.num_nodes))
    edge_list = subgraph.edge_index.t().tolist()
    G.add_edges_from(edge_list)

    # Create a visualization of the graph, considering the 'attr' mask
    position = nx.kamada_kawai_layout(G)

    # Define node colors based on 'attr' value
    # 红色的点表示被选中的子图，蓝色的点表示没有被选中
    node_colors = ['blue' if subgraph.attr[node] == 0 else 'red' for node in G.nodes()]

    labels = {i: str(i) for i in G.nodes()}

    plt.figure(figsize=(20, 20))
    nx.draw(G, position, with_labels=True, labels=labels, nodelist=G.nodes(), node_color=node_colors, node_size=300, edgelist=edge_list, font_size=10, font_color='black', font_weight='bold')
    plt.title("Graph Visualization")

    # Save the figure in high resolution
    plt.savefig("figs/graph_visualization/graph_visualization_"+str(i)+".png", format="PNG", dpi=300)

    plt.show()

def draw_evaluation(subgraph,i,nodes_number):
    node_mask_prob = subgraph['attr'].T[0]
    node_mask = torch.as_tensor((node_mask_prob - node_mask_prob.mean()) > 0, dtype=torch.int32)
    subgraph.attr = node_mask

    # Convert the Data object to a NetworkX graph, considering the 'attr' mask
    G = nx.Graph()
    filtered_nodes = torch.where(subgraph.attr == 1)[0][:nodes_number]

    # Create a subgraph with the filtered nodes
    filtered_subgraph = subgraph.subgraph(filtered_nodes)

    edge_list = filtered_subgraph.edge_index.t().tolist()
    G.add_edges_from(edge_list)

    # Create a visualization of the graph, considering the 'attr' mask
    position = nx.kamada_kawai_layout(G)

    # Define node colors based on 'attr' value
    # 红色的点表示被选中的子图，蓝色的点表示没有被选中
    node_colors = ['blue' if subgraph.attr[node] == 0 else 'red' for node in G.nodes()]

    labels = {i: str(i) for i in G.nodes()}

    plt.figure(figsize=(16, 16))
    nx.draw(G, position, with_labels=True, labels=labels, nodelist=G.nodes(), node_color=node_colors, node_size=300, edgelist=edge_list, font_size=10, font_color='black', font_weight='bold')
    plt.title("Graph Visualization")

    # Save the figure in high resolution
    plt.savefig("figs/graph_visualization_evaluation/graph_visualization_evaluation_"+str(i)+".png", format="PNG", dpi=300)

    plt.show()



# for i in range(25):
#     subgraph, pos = SG_model(dataset[i])
#     draw_evaluation(subgraph, i, 30)

def statistic_valid(dataset):
    total_num = len(dataset)
    node_masks = []
    for i in range(len(dataset)):
        subgraph, pos = SG_model(dataset[i])
        node_mask_prob = subgraph['attr'].T[0]
        node_mask = torch.as_tensor((node_mask_prob - node_mask_prob.mean()) > 0, dtype=torch.int32)
        node_masks.append(node_mask.tolist())

    node_masks = np.array(node_masks)
    sum = node_masks.sum(axis=0, keepdims=True)/len(node_masks)
    return sum

def visual_graph(G, index):
    # Create a visualization of the graph, considering the 'attr' mask
    position = nx.kamada_kawai_layout(G)
    # Define node colors based on 'attr' value
    labels = G.node_labels()
    # 红色的点表示被选中的子图，蓝色的点表示没有被选中
    node_colors = []
    for i in range(len(labels)):
        if(labels[i] == 0):
            node_colors.append('black')
        elif(labels[i] == 1):
            node_colors.append('red')
        elif(labels[i] == 2):
            node_colors.append('salmon')
        else:
            node_colors.append('skyblue')

    plt.figure(figsize=(20, 20))
    nx.draw(G, position, with_labels=True, node_color=node_colors, node_size=500, font_size=10, font_color='lightgreen', font_weight='bold')
    plt.title("Graph Visualization")
    # Save the figure in high resolution
    # plt.savefig("figs/graph_visualization/"+str(index)+".pdf", format="PNG", dpi=300)
    # Display the plot
    plt.show()

def get_avg_zs_and_nc_barplot(dataset):
    normal_sample = []
    neg_patients = []
    for i in range(len(dataset)):
        if (dataset[i]['y'].item() == 1):
            neg_patients.append(dataset[i])
        else:
            normal_sample.append(dataset[i])

    patient_sum = statistic_valid(neg_patients)
    normal_sum = statistic_valid(normal_sample)
    patient_sum = patient_sum.flatten()
    normal_sum = normal_sum.flatten()

    patient_sum_list = list(patient_sum)
    normal_sum_list = list(normal_sum)

    # Determine the width of the bars and the number of categories
    bar_width = 0.35
    num_categories = len(normal_sum)  # Assuming both lists have the same length

    # Create a larger figure for the plot
    plt.figure(figsize=(24, 8))  # Adjust the width and height as needed

    # Create the grouped bar chart
    index = np.arange(num_categories)
    plt.bar(index, patient_sum, bar_width, label='Patients', color='b')
    plt.bar(index + bar_width, normal_sum, bar_width, label='normal people', color='r')

    # Set x-axis labels and add a legend
    plt.xticks(index + bar_width / 2, index)
    plt.legend()

    # Add a title and labels
    plt.title('nodes probs')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.savefig("positive and negative patients nodes probs (BSNIP_GIB).png", format="PNG", dpi=300)
    # Show the bar graph
    plt.show()

def visual_MUTAG_graph(data, node_labels, name, node_mask, index):
    # Create a visualization of the graph, considering the 'attr' mask
    # Create a networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.T.tolist())
    position = nx.kamada_kawai_layout(G)
    # Define node colors based on 'attr' value
    # 红色的点表示被选中的子图，蓝色的点表示没有被选中
    node_colors = []
    for i in range(len(node_mask)):
        if(node_mask[i] == 0):
            node_colors.append('black')
        elif(node_mask[i] == 1):
            node_colors.append('red')
        elif(node_mask[i] == 2):
            node_colors.append('salmon')
        else:
            node_colors.append('skyblue')

    labels_dict = {node: label for node, label in zip(G.nodes(), node_labels)}
    plt.figure(figsize=(12, 12))
    nx.draw(G, position, with_labels=True, labels=labels_dict, node_color=node_colors, node_size=5000, font_size=50,
            font_color='lightgreen', font_weight='bold')
    plt.title("Graph Visualization")
    # Save the figure in high resolution
    plt.savefig("figs/"+name+"/"+str(index)+".pdf", format="PDF", dpi=300)
    # Display the plot
    plt.show()


def visual_MUTAG_graph_edge(data, node_labels, edge_labels, name, edge_mask, index):
    # Create a visualization of the graph, considering the 'attr' mask
    # Create a networkx graph
    node_int_to_label = {integer: label for integer, label in
                         zip([0, 1, 2, 3, 4, 5, 6], ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br'])}
    edge_int_to_label = {integer: label for integer, label in
                         zip([0, 1, 2, 3], ['aromatic', 'single', 'double', 'triple'])}

    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.T.tolist())
    position = nx.kamada_kawai_layout(G)
    # Define node colors based on 'attr' value
    # 红色的点表示被选中的子图，蓝色的点表示没有被选中
    node_colors = []
    for i in range(len(node_labels)):
        if(node_labels[i] == 0):
            node_colors.append('black')
        elif(node_labels[i] == 1):
            node_colors.append('green')
        elif(node_labels[i] == 2):
            node_colors.append('skyblue')
        elif (node_labels[i] == 3):
            node_colors.append('pink')
        else:
            node_colors.append('salmon')

    edge_colors = []
    for i in range(len(edge_mask)):
        if(edge_labels[i] == 0):
            edge_colors.append('black')
        elif(edge_labels[i] == 1):
            edge_colors.append('red')

    edge_type = []

    labels_dict = {node: label for node, label in zip(G.nodes(), node_labels)}
    node_labels = [node_int_to_label[integer] for integer in node_labels]

    plt.figure(figsize=(12, 12))
    nx.draw(G, position, with_labels=True, labels=node_int_to_label, node_color=node_colors, node_size=5000, font_size=50,
            font_color='lightgreen', font_weight='bold', edge_color=edge_colors)
    plt.title("Graph Visualization")
    # Save the figure in high resolution
    plt.savefig("figs/"+name+"/"+str(index)+".pdf", format="PDF", dpi=300)
    # Display the plot
    plt.show()



def count_list_layers(lst):
    if not isinstance(lst, list):
        return 0
    max_depth = 0
    for item in lst:
        if isinstance(item, list):
            depth = count_list_layers(item) + 1
            if depth > max_depth:
                max_depth = depth
    return max_depth



    # baseline data
def Baseline_data_analysis(sub_model):
    names = ['MUTAG', 'PROTEINS', 'DD', 'OHSU', 'github_stargazers', 'REDDIT-BINARY', 'IMDB-BINARY', 'NCI-1']
    name = names[0]
    num_edge_features = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (sub_model == 'MLP_subgraph'):
        dataset, num_nodes, num_node_features, node_labels_list = get_baseline_data(name)
        mutag_node_functional_groups_label = mutag_functional_groups_label(dataset)

        dataset, num_nodes, num_node_features, _ = get_baseline_data(name, True)
        mutag_edge_functional_groups = mutag_functional_groups_label(dataset, True)
        model = GNN(num_of_features=num_node_features,encoder="GIN", device=device).to(device)
        SG_model = MLP_subgraph(node_features_num=num_node_features, num_nodes=num_nodes,
                                edge_features_num=num_edge_features, device=device)
        name = 'MUTAG_BrainIB'
    else:
        dataset, num_nodes, num_node_features, node_labels_list = get_baseline_data(name)
        mutag_node_functional_groups_label = mutag_functional_groups_label(dataset)

        model = GNN(num_of_features=num_node_features,encoder="GIN", device=device).to(device)
        # Instantiate the subgraph generator
        SG_model = GIB(args, number_of_features=num_node_features, device=device).to(device)
        name = 'MUTAG_GIB'


    best_epoch_number = 47
    with open("./SGSIB/model/"+name+"/Best_Epoch.txt", 'r') as f:
        # Read the entire contents of the file
        line = f.read()
        best_epoch_number = int(float(line.strip()))

    dir_number = 0
    savedir = "./SGSIB/model/"+name+"/GNN_model" + str(dir_number)
    sub_path = savedir + "/subgraph" + "_" + str(best_epoch_number) + ".tar"
    para_dict = torch.load(sub_path)['state_dict']
    SG_model.load_state_dict(para_dict)
    SG_model.eval()

    if (sub_model == 'MLP_subgraph'):
        edge_masks = []

        for i in range(len(dataset)):  # len(dataset)
            subgraph, pos = SG_model(dataset[i])
            edge_mask_prob = subgraph['attr']
            edge_mask = torch.as_tensor((edge_mask_prob - edge_mask_prob.mean()) > 0, dtype=torch.int32)
            edge_masks.append(edge_mask.tolist())

            node_labels = mutag_node_functional_groups_label[i]
            edge_labels = mutag_edge_functional_groups[i][0]

            visual_MUTAG_graph_edge(dataset[i], node_labels, edge_labels, name, edge_mask, i)



    #     start to visual the graph
        edge_masks

    else:
        node_masks = []    # Create a dictionary mapping integers to node labels
        int_to_label = {integer: label for integer, label in zip([0, 1, 2, 3, 4, 5, 6], ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br'])}
        for i in range(len(dataset)):  #len(dataset)
            subgraph, pos = SG_model(dataset[i])
            node_mask_prob = subgraph['attr'].T[1]
            node_mask = torch.as_tensor((node_mask_prob - node_mask_prob.mean()) > 0, dtype=torch.int32)
            node_masks.append(node_mask.tolist())
            # Visualization
            # Create a new list by mapping integers to node labels
            node_labels = [int_to_label[integer] for integer in mutag_node_functional_groups_label[i].tolist()]
            visual_MUTAG_graph(dataset[i], node_labels, name, node_mask, i)





    # Initialize counter for common numbers
    # Predicted Positive, Actual Positive
    TP = 0
    # Predicted Negative, Actual Negative
    TN = 0

    # Predicted Positive, Actual Negative
    FP = 0
    # Predicted Negative, Actual Positive
    FN = 0

    sum_of_common_numbers = 0
    if (sub_model == 'MLP_subgraph'):
        masks = edge_masks
    else:
        masks = node_masks
    # Iterate through corresponding small lists in both lists
    for mask, groups_label in zip(masks, mutag_edge_functional_groups):
        # Count common numbers in the two small lists
        if(count_list_layers(groups_label)>0):
            groups_label = groups_label[0]
        for x1, x2 in zip(mask, groups_label):
            if x1 == x2 == 1:
                TP += 1
            elif x1 == x2 == 0:
                TN += 1
            elif (x1 == 1 and x2 == 0):
                FP += 1
            elif (x1 == 0 and x2 == 1):
                FN += 1
        sum_of_common_numbers += sum(groups_label)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(f"Accuracy:{accuracy:.4f},Precision:{precision:.4f},Recall:{recall:.4f},F1 score:{f1_score:.4f}")


if __name__ == '__main__':
    args = parameter_parser()
    num_node_features = 105
    num_edge_features = 1

    # Quantativity analysis
    # sub_model = 'GIB'
    sub_model = "MLP_subgraph"
    Baseline_data_analysis(sub_model)

    # num_node_features = 140
    # dataset = read_dataset()
    # dataset = read_Schi_dataset()
    # name = "Synthetic"
    # dataset = torch.load('./synthetic_data/Synthetic_dataset.pt')

#     Start analysis the multi_site dataset in UCLA
#     dataset = read_UCLA_dataset()
    dataset = read_Schi_dataset()
    encoder = "GIN"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(num_of_features=num_node_features, encoder=args.encoder, device=device).to(device)
    SG_model = GIB(args, number_of_features=num_node_features, device=device).to(device)

    name = "BSNIP_multi_site_GIB_32_alpha2_GIN_0.6385in32"
    # name = "BSNIP"
    best_epoch_number = 32
    # with open("./SGSIB/model/"+name+"/Best_Epoch.txt", 'r') as f:
    #     # Read the entire contents of the file
    #     line = f.read()
    #     best_epoch_number = int(float(line.strip()))

    dir_number = 0
    savedir = "./SGSIB/model/"+name+"/GNN_model" + str(dir_number)
    sub_path = savedir + "/subgraph" + "_" + str(best_epoch_number) + ".tar"
    para_dict = torch.load(sub_path)['state_dict']
    SG_model.load_state_dict(para_dict)
    SG_model.eval()

    # node_masks = []  # Create a dictionary mapping integers to node labels
    # for i in range(len(dataset)):  # len(dataset)
    #     subgraph, pos = SG_model(dataset[i])
    #     node_mask_prob = subgraph['attr'].T[0]
    #     node_mask = torch.as_tensor((node_mask_prob - node_mask_prob.mean()) > 0, dtype=torch.int32)
    #     node_masks.append(node_mask.tolist())
    #
    # node_masks = np.array(node_masks)
    # sum = node_masks.sum(axis=0, keepdims=True)/len(node_masks)

    probabilities = statistic_valid(dataset)
    n = 20

def top_n_largest_with_indices(probabilities, n=20):
    # Enumerate the list to keep track of original indices
    indexed_numbers = list(enumerate(probabilities[0]))

    # Sort the list based on the numbers in descending order to get the largest numbers first
    sorted_indexed_numbers = sorted(indexed_numbers, key=lambda x: x[1], reverse=True)

    # Extract the top n largest numbers and their original indices
    top_n_numbers = [num for idx, num in sorted_indexed_numbers[:n]]
    top_n_indices = [idx for idx, num in sorted_indexed_numbers[:n]]

    return top_n_numbers, top_n_indices

    top_n_numbers, top_n_indices = top_n_largest_with_indices(probabilities, n=20)
    print(top_n_numbers, top_n_indices)


top_n_largest_with_indices(probabilities, n=20)