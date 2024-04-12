import torch
from SGSIB.GNN import GNN
from SGSIB.sub_node_generator import GIB
from SGSIB.sub_graph_generator import MLP_subgraph
from SGSIB_main import parameter_parser
from BrainIB_V2.real_data.create_dataset import read_dataset
from BrainIB_V2.real_data.create_dataset import read_Schi_dataset
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data


args = parameter_parser()
num_node_features = 105
num_edge_features = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = GNN(num_of_features=num_node_features, device=device).to(device)
# Instantiate the subgraph generator
# SG_model = MLP_subgraph(node_features_num=num_node_features, edge_features_num=num_edge_features, device=device)
SG_model = GIB(args, number_of_features=num_node_features, device=device).to(device)


dir_number = 0
savedir = "./SGSIB/model/GNN_model" + str(dir_number)

# best_epoch_number = 34

best_epoch_number = 47

# path = savedir + "/GNN" + "_" + str(epoch_number) + ".tar"
# origin_path = savedir + "/GNN" + "_" + str(epoch_number) + ".tar"
# para_dict = torch.load(origin_path)['state_dict']
# model.load_state_dict(para_dict)
# model.eval()

sub_path = savedir + "/subgraph" + "_" + str(best_epoch_number) + ".tar"
para_dict = torch.load(sub_path)['state_dict']
SG_model.load_state_dict(para_dict)
SG_model.eval()

# dataset = read_dataset()
dataset = read_Schi_dataset()
normal_sample=[]
neg_patients=[]
for i in range(len(dataset)):
    if(dataset[i]['y'].item()==1):
        neg_patients.append(dataset[i])
    else:
        normal_sample.append(dataset[i])



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


def draw_compare(subgraph,i):
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

    plt.figure(figsize=(16, 16))
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
plt.bar(index, patient_sum, bar_width, label='Patients',color='b')
plt.bar(index + bar_width, normal_sum, bar_width, label='normal people',color='r')

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



