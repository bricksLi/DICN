import numpy as np
import torch
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid


def get_dataset(dataset):
    datasets = Planetoid('./dataset', dataset)  
    return datasets


def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def vGraph_loss(c, recon_c, prior, q):  

    BCE_loss = F.cross_entropy(recon_c, c) / c.shape[
        0]  ### Normalization is necessary or the dimension of c is too large and it will be the most weighted
    # KL_div_loss = F.kl_div(torch.log(prior + 1e-20), q, reduction='batchmean')
    KL_div_loss = torch.sum(q * (torch.log(q + 1e-20) - torch.log(prior)),
                            -1).mean()  ## As such main use is of just mean()

    loss = BCE_loss + KL_div_loss
    return loss


def cuda(xs, gpu_id):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda(int(gpu_id[0]))
        else:
            return [x.cuda(int(gpu_id[0])) for x in xs]
    return xs


def similarity_measure(edge_index, w, c, gpu_id):  
    '''
    Used for calculating the coefficient alpha in the case of community smoothness loss
    Parameters:
    edge_index: edge matrix of the graph
    w: the starting node values of an edge
    c: the ending node values of an edge
    '''

    alpha = torch.zeros(w.shape[0], 1)
    alpha = cuda(alpha, gpu_id)
    for i in range(w.shape[0]):
        l1 = edge_index[1, :][edge_index[0, :] == w[i]].tolist()
        l2 = edge_index[1, :][edge_index[0, :] == c[i]].tolist()

        common_neighbors = [value for value in l1 if value in l2]
        common_neighbors = len(common_neighbors)
        all_neighbors = len(l1) + len(l2)
        similarity = (float)(common_neighbors / all_neighbors)
        alpha[i, 0] = similarity

    return alpha  # （21112,1）
