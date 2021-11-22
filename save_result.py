#RGCN result save to dw


#Version update
############################################################################################################################
# 0.0.1 : June 24th 2021 : save_result.py created
#
#
############################################################################################################################


__author__: Jinsook Jennie Lee
import torch as th
import numpy as np
import pandas as pd

import sys
import argparse

from model import RGCN, RGCNConv
from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, sort_and_rank, filter_o
from main import *
    
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
            filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities)
            target_o_idx = int((filtered_o == target_o).nonzero())
            emb_s = embedding[target_s]
            emb_r = w[target_r]
            emb_o = embedding[filtered_o]
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

            score_list_idx.append([list_of_key[int(target_s)], score_list])
        return score_list_idx
    
    
def main(args):
    #Data Load

    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data('./data')
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))

    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)

    model = RGCN(len(entity2id), len(relation2id), num_bases=4, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    device = torch.device("cuda:1")
    #model.state_dict() 확인
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        
    #Model Load
    model.eval()
    checkpoint = th.load('best_mrr_model.pth',map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    valid_mrr = checkpoint['MRR']

    print("Epoch:", epoch)
    print("Loss:", loss)
    print("Valid MRR:",valid_mrr)
    
    #embedding weight load
    model.to("cpu") 
    embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    
    final_score_list = get_score_all(embedding, model.relation_embedding, train_triplets, valid_triplets, test_triplets, entity2id)

    result_f = []
    for i in tqdm(range(len(final_score_list))):
        result = pd.DataFrame(final_score_list[i][1],columns = ['th_idx','mmajor_nm','score'])
        result['std_id'] = final_score_list[i][0]
        result_f.append(result)
        
    result_ff = pd.concat(result_f, ignore_index=True)
    result_ff = result_ff[result_ff['score']>0.1]
    result_ff.to_csv("./rgcn_sec_major_rec_result.csv", encoding = 'utf8', sep='\t')
    return result_ff

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='save_result_to_dw')
    args = parser.parse_args()
    print(args)
    main(args)
