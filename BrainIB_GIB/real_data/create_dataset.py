import os
import scipy.io as scio
from scipy.sparse import coo_matrix
from scipy.io import loadmat
import torch
from torch_geometric.data import Data


def read_dataset():
    """
    read real_data from PATH+'/real_data/md_AAL_0.4.mat' and reconstruct real_data as 'torch_geometric.real_data.Data'
    """
    PATH = os.getcwd()
    data = scio.loadmat(PATH+'/md_AAL_0.4.mat')
    # real_data = scio.loadmat(PATH+'/real_data/md_AAL_0.4.mat') # Data is available at google drive (https://drive.google.com/drive/folders/1EkvBOoXF0MB2Kva9l4GQbuWX25Yp81a8?usp=sharing).
    dataset = []
    for graph_index in range(len(data['label'])):
        label = data['label']

        graph_struct = data['graph_struct'][0]

        edge = torch.Tensor(graph_struct[graph_index]['edge'])

        ROI = torch.Tensor(graph_struct[graph_index]['ROI'])

        node_tags = torch.Tensor(graph_struct[graph_index]['node_tags'])
        adj = torch.Tensor(graph_struct[graph_index]['adj'])
        neighbor = graph_struct[graph_index]['neighbor']
        y = torch.Tensor(label[graph_index])
        # indices = edge[:, :2]
        # values = edge[:, -1]
        A = torch.sparse_coo_tensor(indices = edge[:, :2].t().long(),values = edge[:, -1].reshape(-1,).float(),size = (116, 116))
        G = (A.t() + A).coalesce()

        graph = Data(x=ROI.reshape(-1,116).float(),edge_index=G.indices().reshape(2,-1).long(),edge_attr=G.values().reshape(-1,1).float(), y=y.long())
        dataset.append(graph)
    print("finish create dataset")
    return dataset

def read_test_dataset(num_graphs):
    """
        read real_data from PATH+'/real_data/md_AAL_0.4.mat' and reconstruct real_data as 'torch_geometric.real_data.Data'
        """
    PATH = os.getcwd()
    data = scio.loadmat(PATH+'/md_AAL_0.4.mat')
    # real_data = scio.loadmat('BrainIB_V2/md_AAL_0.4.mat') # Data is available at google drive (https://drive.google.com/drive/folders/1EkvBOoXF0MB2Kva9l4GQbuWX25Yp81a8?usp=sharing).
    dataset = []
    label = data['label']
    graph_struct = data['graph_struct'][0]

    for graph_index in range(num_graphs):
        edge = torch.Tensor(graph_struct[graph_index]['edge'])

        ROI = torch.Tensor(graph_struct[graph_index]['ROI'])

        # node_tags = torch.Tensor(graph_struct[graph_index]['node_tags'])
        adj = torch.Tensor(graph_struct[graph_index]['adj'])
        neighbor = graph_struct[graph_index]['neighbor']
        y = torch.Tensor(label[graph_index])
        edge_weak_connet = edge[:, :2].t().long()
        edge_weight = edge[:, -1]
        A = torch.sparse_coo_tensor(indices=edge[:, :2].t().long(), values=edge[:, -1].reshape(-1, ).float(),
                                    size=(116, 116))
        G = (A.t() + A).coalesce()

        graph = Data(x=ROI.reshape(-1, 116).float(), edge_index=G.indices().reshape(2, -1).long(), edge_attr=G.values().reshape(-1, 1).float(),
                     # edge_weak_connet=edge_weak_connet, edge_weight=edge_weight,
                     y=y.long())
        dataset.append(graph)
    print("finish create dataset")
    return dataset


def read_Schi_dataset():
    PATH = os.getcwd()
    BSNIP = loadmat(PATH + '/BSNIP.mat')
    dataset = []
    data = BSNIP['BSNIP']
    group = data['group']

    num = len(group)
    # for graph_index in range(len(group)):
    for graph_index in range(num):
        group_label = group[graph_index][0][0]
        if(group_label=='SZ' or group_label=='NC'):
            ROI = torch.Tensor(data[graph_index]['FC'][0])
            edge = torch.Tensor(data[graph_index]['edges'][0])
            label = 1 if group_label == 'SZ' else 0
            y = torch.Tensor([label])
            A = torch.sparse_coo_tensor(indices=edge[:, :2].t().long(), values=edge[:, -1].reshape(-1, ).float(),
                                        size=(105, 105))
            G = (A.t() + A).coalesce()

            graph = Data(x=ROI.reshape(-1, 105).float(), edge_index=G.indices().reshape(2, -1).long(),
                         edge_attr=G.values().reshape(-1, 1).float(),
                         # edge_weak_connet=edge_weak_connet, edge_weight=edge_weight,
                         y=y.long())
            dataset.append(graph)
    print("finish read Schizophrenia dataset")
    return dataset



if __name__ == '__main__':
    # here is some test code, for the dataset reader loader
    # sample_data = read_test_dataset(1)
    dataset = read_Schi_dataset()
    print(len(dataset))













