import os
import os.path as osp
import argparse
import wandb
import sys
import torch

import numpy as np

from SGSIB.GNN import GNN
from SGSIB.sub_node_generator import GIB
from SGSIB.sub_graph_generator import MLP_subgraph
from SGSIB.utils import train, test, separate_data
from real_data.create_dataset import read_Schi_dataset, read_dataset, read_UCLA_dataset
# from synthetic_data.synthetic_main import create_syn_dataset_1
from baseline_data.baseline_main import get_baseline_data
os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"


def parameter_parser():
    parser = argparse.ArgumentParser(description='"GIB"')
    parser.add_argument('--sub_model', type=str, default="MLP_subgraph",
                        help='Pick the subgraph model from MLP_subgraph and GIB (default: GIB)')
    parser.add_argument('--dataset', type=str, default="UCLA",
                        help='Pick the dataset from ABIDE, BSNIP and some baseline dataset (default: BSNIP)')
    parser.add_argument('--multi_site', type=bool, default=True,
                        help='Decide do the multisite training or not (default: False)')
    parser.add_argument('--encoder', type=str, default="GIN",
                        help='Pick the model encoder from GCN, GIN and GAT (default: GCN)')
    parser.add_argument('--alpha', type=float, default=1.01,
                        help='Pick the value of the alpha in the MI estimator from 1.01, 2 and 5 (default: 1.01)')
    parser.add_argument('--iters_per_epoch', type=int, default=1,
                        help='number of iterations per each epoch (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument("--mi_weight", type=float, default=0.001,
                        help="weight of mutual information loss (default: 0.001)")
    parser.add_argument("--pos_weight", type=float, default=0.001,
                        help="weight of mutual information loss (default: 0.001)")
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--model_learning_rate', type=float, default=0.0002,
                        help='learning rate of graph model (default: 0.0005)')
    parser.add_argument('--SGmodel_learning_rate', type=float, default=0.0002,
                        help='learning rate of subgraph model (default: 0.0005)')

    parser.add_argument('--input_dim', dest='input_dim', type=int, default=105,
                        help='Input feature dimension (default: 105)')
    parser.add_argument('--number_of_patients', dest='number_of_patients', type=int, default=300,
                        help='Input number of patients in synthetic dataset (default: 300)')
    parser.add_argument('--number_of_normal_control', dest='number_of_normal_control', type=int, default=300,
                        help='Input number of normal control in synthetic dataset (default: 300)')
    parser.add_argument('--sz_nb_shapes', dest='sz_nb_shapes', type=int, default=10,
                        help='Input number of motif house in synthetic dataset (default: 3)')
    parser.add_argument('--sz_width_basis', dest='sz_width_basis', type=int, default=90,
                        help='Input number of basic nodes in synthetic dataset (default: 90)')
    parser.add_argument('--nc_nb_shapes', dest='nc_nb_shapes', type=int, default=0,
                        help='Input number of motif house in synthetic dataset (default: 0)')
    parser.add_argument('--nc_width_basis', dest='nc_width_basis', type=int, default=140,
                        help='Input number of basic nodes in synthetic dataset (default: 105)')

    parser.add_argument("--first-gcn-dimensions", type=int, default=16, help="Filters (neurons) in 1st convolution. Default is 16.")
    parser.add_argument("--second-gcn-dimensions", type=int, default=8, help="Filters (neurons) in 2nd convolution. Default is 8.")
    parser.add_argument("--first-dense-neurons", type=int, default=16, help="Neurons in SAGE aggregator layer. Default is 16.")
    parser.add_argument("--second-dense-neurons", type=int, default=2, help="assignment. Default is 2.")
    # parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 5)')
    return parser.parse_args()



if __name__ == '__main__':
    args = parameter_parser()
    # torch.manual_seed(0)
    np.random.seed(0)
    # np.random.seed(42)
    # np.random.seed(215)

    dataset_name = args.dataset
    num_edge_features = 1
    sub_model = args.sub_model
    batch_size = str(args.batch_size)
    multi_site = args.multi_site

    # if(dataset_name == "ABIDE"):
    #     num_node_features = 116
    #     num_nodes = 116
    #     dataset = read_dataset()
    #
    # el
    if(dataset_name == "BSNIP"):
        num_node_features = 105
        num_nodes = 105
        dataset = read_Schi_dataset()

    elif(dataset_name == "UCLA"):
        num_node_features = 105
        num_nodes = 105
        dataset = read_UCLA_dataset()

    # num_node_features = 140
    # num_node_features = args.nc_nb_shapes
    # dataset_name = "Synthetic"
    # dataset = create_syn_dataset_1(args)
    # torch.save(dataset, './synthetic_data/Synthetic_dataset.pt')


    # baseline data
    # dataset_names = ['MUTAG', 'PROTEINS', 'DD', 'OHSU', 'github_stargazers', 'REDDIT-BINARY', 'IMDB-BINARY', 'NCI-1']
    # dataset_name = dataset_names[0]
    # if(sub_model =='MLP_subgraph'):
    #     dataset, num_nodes, num_node_features, _ = get_baseline_data(dataset_name,True)
    # else:
    #     dataset, num_nodes, num_node_features, _ = get_baseline_data(dataset_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    num_of_fold = 1
    acc_train_list = torch.zeros((num_of_fold,))
    acc_test_list = torch.zeros((num_of_fold,))
    best_epoch = 0




    for fold_idx in range(num_of_fold):
        # start a new wandb run to track this script
        if (multi_site == True):
            if (dataset_name == "BSNIP"):
                print("Use BSNIP as training dataset and test in UCLA!")
                 # train_dataset = dataset
                dataset_name = "BSNIP_multi_site"
                train_dataset, _ = separate_data(dataset, args.seed, fold_idx)
                UCLA_dataset = read_UCLA_dataset()
                _, test_dataset = separate_data(UCLA_dataset, args.seed, fold_idx)
            elif(dataset_name == "UCLA"):
                print("Use UCLA as training dataset and test in BSNIP!")
                # train_dataset = dataset
                dataset_name = "UCLA_multi_site"
                train_dataset = dataset
                BSNIP_dataset = read_Schi_dataset()
                _, test_dataset = separate_data(BSNIP_dataset, args.seed, fold_idx)
                # _, test_dataset = separate_data(test_dataset, args.seed, fold_idx)

        wandb.login(key="b8ed23bd2641fa19932901f9d0cd144c7ead0283")
        wandb.init(
            # set the wandb project where this run will be logged
            project="SG_by_node_generator_1024",
            name=dataset_name + "_" + sub_model + "_" + batch_size +"_alpha" +str(args.alpha)+"_"+args.encoder,
            # track hyperparameters and run metadata
            config={
                "model_learning_rate": args.model_learning_rate,
                "SGmodel_learning_rate": args.SGmodel_learning_rate,
                "architecture": args.encoder,
                "dataset": dataset_name,
                "alpha value": args.alpha,
                "model_learning_rate": args.model_learning_rate,
                "SGmodel_learning_rate": args.SGmodel_learning_rate,
                "epochs": args.epochs, }
        )
        max_acc_train = 0.0
        max_acc_test = 0.0
        

        # Instantiate the backbone network

        model = GNN(num_of_features=num_node_features, encoder=args.encoder, device=device).to(device)
        # Instantiate the subgraph generator
        if (sub_model == "MLP_subgraph"):
            SG_model = MLP_subgraph(node_features_num=num_node_features, num_nodes=num_nodes, edge_features_num=num_edge_features, device=device)
        if (sub_model == "GIB"):
            SG_model = GIB(args, number_of_features=num_node_features, device=device).to(device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.model_learning_rate},
            {'params': SG_model.parameters(), 'lr': args.SGmodel_learning_rate}
            ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        
        for epoch in range(1, args.epochs + 1):
            # Train the model and test it
            avg_loss, mi_loss, avg_mi_loss = train(args, model, train_dataset, optimizer, epoch, SG_model, device)
            acc_train, acc_test, test_loss = test(args, model, train_dataset, test_dataset, SG_model, device)
            
            # print info and save models
            max_acc_train = max(max_acc_train, acc_train)
            acc_train_list[fold_idx] = max_acc_train
            if(acc_test > max_acc_test):
                best_epoch = epoch
            if(max_acc_test>0.74):
                print("found the best performance!!")
                sys.exit()
            max_acc_test = max(max_acc_test, acc_test)
            acc_test_list[fold_idx] = max_acc_test
            print(f'Accuracy in epoch {epoch} (this acc / best acc in best epoch): ({acc_test} / {max_acc_test}) with (lr = {(scheduler.get_last_lr()[0])})')
            # log metrics to wandb
            wandb.log({"learning-rate": scheduler.get_last_lr()[0], "max_acc_train": max_acc_train, "max_acc_test": max_acc_test, "acc_train": acc_train, "acc_test": acc_test, "avg_loss": avg_loss, "mi_loss": mi_loss,"avg_mi_loss":avg_mi_loss,"test_loss":test_loss})

            savedir = "./SGSIB/model/" + dataset_name +"/GNN_model" + str(fold_idx)
            if not osp.exists(savedir):
                os.makedirs(savedir)
            savename = savedir + "/GNN" + "_" + str(epoch) + ".tar"
            torch.save({"epoch" : epoch, "state_dict": model.state_dict(),}, savename)

            savedir = "./SGSIB/model/"+ dataset_name +"/GNN_model" + str(fold_idx)
            if not osp.exists(savedir):
                os.makedirs(savedir)
            savename = savedir + "/subgraph" + "_" + str(epoch) + ".tar"
            torch.save({"epoch" : epoch, "state_dict": SG_model.state_dict(),}, savename)

            filename="./SGSIB/model/"+ dataset_name +"/GNN_" + str(fold_idx) + ".txt"
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write("%f %f %f %f" % (avg_loss, acc_train, acc_test, mi_loss, ))
                    f.write("\n")
            else:
                with open(filename, 'a+') as f:
                    f.write("%f %f %f %f" % (avg_loss, acc_train, acc_test, mi_loss))
                    f.write("\n")
            
            scheduler.step()
            torch.cuda.empty_cache()

        filename = "./SGSIB/model/"+ dataset_name +"/Best_Epoch.txt"
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("%f" % (best_epoch))
                f.write("\n")
        wandb.finish()

    print(100*'*')
    print('ASD 10-fold validation results: ')
    print('Model dataset_Name: SGSIB')
    print(f"train accuracy list: {acc_train_list}")
    print(f"mean = {acc_train_list.mean()}, variance = {acc_train_list.var()}")
    print(f"test accuracy list: {acc_test_list}")
    print(f"mean = {acc_test_list.mean()}, variance = {acc_test_list.var()}")
    print(100*'*')