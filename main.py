#!/usr/bin/env python
# coding: utf-8

# In[2]:

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm, trange
#from torch.utils.tensorboard import SummaryWriter

from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
from model import RGCN

import sys


def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    if use_cuda:
        device = torch.device('cuda:1')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)

    return loss

def valid(valid_triplets, model,train_triplets, test_graph, test_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, train_triplets, test_triplets, valid_triplets, hits=[1, 3, 10])

    return mrr

def test(test_triplets, model,train_triplets, test_graph, valid_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, train_triplets, valid_triplets, test_triplets, hits=[1, 3, 10])

    return mrr

def main(args):

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    best_mrr = 0

    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data('./data')
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))

    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)
    
    model = RGCN(len(entity2id), len(relation2id), num_bases=args.n_bases, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(model)

    if use_cuda:
        model.cuda()
    

    history = pd.DataFrame(columns=['train_loss','epoch'])
               
    for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):

        model.train()
        optimizer.zero_grad()

        loss = train(train_triplets, model, use_cuda, batch_size=args.graph_batch_size, split_size=args.graph_split_size, 
            negative_sample=args.negative_sample, reg_ratio = args.regularization, num_entities=len(entity2id), num_relations=len(relation2id))
        
        tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        
        history = history.append(pd.DataFrame([[loss, epoch]],columns=['train_loss','epoch']))
          
        if epoch % args.evaluate_every == 0:

            if use_cuda:
                model.cpu()

            model.eval()
            valid_mrr = valid(valid_triplets, model,train_triplets, test_graph, test_triplets)
            
            tqdm.write("Valid MRR {}, at epoch {}".format(valid_mrr, epoch))
            #history.append(pd.DataFrame([[valid_loss]], columns = ['valid_loss']))
            
            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'epoch': epoch,
                           'loss':loss,'MRR':valid_mrr},'best_mrr_model.pth')

            if use_cuda:
                model.cuda()
 
    if use_cuda:
        model.cpu()
    
    history.to_csv("history_0713.txt") 

    model.eval()

    checkpoint = torch.load('best_mrr_model.pth')
    model.load_state_dict(checkpoint['state_dict'])

    test_mrr = test(test_triplets, model,train_triplets, test_graph, valid_triplets)

if __name__ == '__main__':
    '''
    graph_batch_size = sys.argv[1]
    graph_split_size = sys.argv[2]
    negative_sample = sys.argv[3]
    n_epochs = sys.argv[4]
    evaluate_every = sys.argv[5]
    '''
    
    parser = argparse.ArgumentParser(description='RGCN')
    
    parser.add_argument("--graph-batch-size", type=int, default=40000)
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=10000)
    parser.add_argument("--evaluate-every", type=int, default=500)
    
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=4)
    
    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    main(args)

