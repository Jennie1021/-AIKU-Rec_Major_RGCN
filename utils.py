#!/usr/bin/env python
# coding: utf-8

############################################################################################################################
#
#Note : entities.dict must in an order that "course" entities always in first to calculate filtered MRR among only courses
#
############################################################################################################################


import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################



def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def load_data(file_path):
    '''
        Loading data
        argument:
            file_path: ./data/
        
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'sec_entities.dict')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'sec_relation.dict')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triplets = read_triplets(os.path.join(file_path, 'sec_train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'sec_valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'sec_test2.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data

def sort_and_rank(score, target): 
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#######################################################################
#
# Utility functions for evaluations (raw) -- not use
#
#######################################################################

# def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=100):
#     """ Perturb one element in the triplets
#     """
#     n_batch = (test_size + batch_size - 1) // batch_size
#     ranks = []
#     for idx in range(n_batch):
#         print("batch {} / {}".format(idx, n_batch))
#         batch_start = idx * batch_size
#         batch_end = min(test_size, (idx + 1) * batch_size)
#         batch_a = a[batch_start: batch_end]
#         batch_r = r[batch_start: batch_end]
#         emb_ar = embedding[batch_a] * w[batch_r]
#         emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
#         emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
#         # out-prod and reduce sum
#         out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
#         score = torch.sum(out_prod, dim=0) # size E x V
#         score = torch.sigmoid(score)
#         target = b[batch_start: batch_end]
#         ranks.append(sort_and_rank(score, target))
#     return torch.cat(ranks)


# # return MRR (raw), and Hits @ (1, 3, 10)
# def calc_raw_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
#     with torch.no_grad():
#         s = test_triplets[:, 0]
#         r = test_triplets[:, 1]
#         o = test_triplets[:, 2]
#         test_size = test_triplets.shape[0]

#         # perturb subject
#         ranks_s = perturb_and_get_raw_rank(embedding, w, o, r, s, test_size, eval_bz)
#         # perturb object
#         ranks_o = perturb_and_get_raw_rank(embedding, w, s, r, o, test_size, eval_bz)

#         ranks = torch.cat([ranks_s, ranks_o])
#         ranks += 1 # change to 1-indexed

#         mrr = torch.mean(1.0 / ranks.float())
#         print("MRR (raw): {:.6f}".format(mrr.item()))

#         for hit in hits:
#             avg_count = torch.mean((ranks <= hit).float())
#             print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
#     return mrr.item()


#
#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider an object if it is part of a triplet to filter
    for o in range(num_entities):
        if (target_s, target_r, o) not in triplets_to_filter:
            filtered_o.append(o)
    return torch.LongTensor(filtered_o)

def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider a subject if it is part of a triplet to filter
    for s in range(num_entities):
        if (s, target_r, target_o) not in triplets_to_filter:
            filtered_s.append(s)
    return torch.LongTensor(filtered_s)

def perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb object in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_o_idx = int((filtered_o == target_o).nonzero())
        emb_s = embedding[target_s]
        emb_r = w[target_r]
        emb_o = embedding[filtered_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        indices_f = [x for x in indices.numpy() if x < 120] # 120 is the number of major entities. Select only major entities
        for i in range(len(indices_f)):
            if indices_f[i] == target_o_idx:
                rank = i
                ranks.append(rank)
    return torch.LongTensor(ranks)

def perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb subject in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_s_idx = int((filtered_s == target_s).nonzero())
        emb_s = embedding[filtered_s]
        emb_r = w[target_r]
        emb_o = embedding[target_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        indices_f = [x for x in indices.numpy() if x < 120] # 120 is the number of major entities. Select only major entities
        for i in range(len(indices_f)):
            if indices_f[i] == target_s_idx:
                rank = i
                ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(embedding, w, train_triplets, other_triplets, test_triplets, hits=[]):
    """
    note : don't get confused valid, test triplets. test_triplets in this function. These are the triplets that you want to predict(could be valid or test)
    """
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = torch.cat([torch.tensor(train_triplets), torch.tensor(other_triplets), torch.tensor(test_triplets)]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        print('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)
        print('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

#######################################################################
#
# Main evaluation function
#
#######################################################################

def calc_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[], eval_bz=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits)
    else:
        mrr = calc_raw_mrr(embedding, w, test_triplets, hits, eval_bz)
    return mrr

#######################################################################
#
# Prediction function
#
#######################################################################


def get_score_all(embedding, w, train_triplets, other_triplets, test_triplets, entity2id):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]
        
        triplets_to_filter = torch.cat([torch.tensor(train_triplets), torch.tensor(other_triplets), torch.tensor(test_triplets)]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        
        num_entities = embedding.shape[0]
        score_list_idx = []
        list_of_key = list(entity2id.keys())
        
        for idx in range(test_size):
            if idx % 100 == 0:
                print("test triplet {} / {}".format(idx, test_size))
            target_s = s[idx]
            target_r = r[idx]
            target_o = o[idx]
            filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities)
            target_s_idx = int((filtered_s == target_s).nonzero())
            emb_s = embedding[filtered_s]
            emb_r = w[target_r]
            emb_o = embedding[target_o]
            emb_triplet = emb_s * emb_r * emb_o
            scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
            _, indices = torch.sort(scores, descending=True)
            
            score_list = []
            for i in range(len(_)):
                score = _.detach().numpy()[i]
                ind = indices.detach().numpy()[i]
                rec_ = list_of_key[ind]
                score_ind = [ind,rec_,score]
                if ind < 120:
                    score_list.append(score_ind)

            score_list_idx.append([target_s, score_list])
        return score_list_idx

