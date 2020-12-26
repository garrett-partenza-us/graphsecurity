import torch
import torch_geometric
import os
import javalang
from javaprep import *
from edge_index import edges
import torch
from torch_geometric.data import Data, DataLoader
import random
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import copy

def get_word_embeddings(treedict, vocabdict):
    sentences = []
    for key in treedict:
        doc = []
        flow = treedict[key]
        vocabulary = flow[0][0]
        for word in vocabulary:
            doc.append(vocabdict[word[0]])
        sentences.append(doc)
    model = Word2Vec(sentences, min_count=1,size= 50,workers=3, window =3, sg = 1)
    return model
    
def generate_graphlist_owasp(treedict, vocabdict, model, cat_res, type_res, cwe):
    graphlist = []
    for key in treedict:
        filename = (key.split('/')[2]).split('.')[0]
        if type_res[filename].item()!=int(cwe):
            continue
        flow = treedict[key]
        vocabulary = flow[0][0]
        srcs = flow[0][1][0]
        trgs = flow[0][1][1]
        edges = flow[0][2]
        x = []
        for i in range(len(set(srcs))):
            x.append(model[(vocabdict[vocabulary[i][0]])].tolist())
        edge_index = torch.tensor([srcs, trgs], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        if cat_res[filename] == True:
            y = 1
        else:
            y = 0
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphlist.append(graph)
    return graphlist

def balance_bad(graphlist, good, bad):
    target = 1
    for i in range(len(graphlist)):
            if graphlist[i].y == target:
                cp = copy.deepcopy(graphlist[i])
                graphlist.append(cp)
                bad+=1
            if good == bad:
                break
    return graphlist, good, bad


def balance_good(graphlist, good, bad):
    target = 0
    for i in range(len(graphlist)):
            if graphlist[i].y == target:
                cp = copy.deepcopy(graphlist[i])
                graphlist.append(cp)
                good+=1
            if good == bad:
                break
    return graphlist, good, bad

def generate_graphlist_juliet(treedict, vocabdict, model, mydict, cwe):
    print("generating graphs...")
    graphlist = []
    count = 0
    good = 0
    bad = 0
    for key in treedict:
        file = (key.split('/')[4]).split('.')[0]
        if mydict[int(file)] == [True]:
            y = 1
            bad+=1
        else:
            good+=1
            y = 0
        flow = treedict[key]
        vocabulary = flow[0][0]
        srcs = flow[0][1][0]
        trgs = flow[0][1][1]
        edges = flow[0][2]
        x = []
        for i in range(len(set(srcs))):
            x.append(model[(vocabdict[vocabulary[i][0]])].tolist())
        edge_index = torch.tensor([srcs, trgs], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphlist.append(graph)
        
    passes = 1
    while good != bad:
        print("balancing dataset (pass "+str(passes)+")")
        if good > bad:
            graphlist, good, bad = balance_bad(graphlist, good, bad)
        elif bad > good:
            graphlist, good, bad = balance_good(graphlist, good, bad)
        passes+=1
            
    print("Number of good methods: ", good, " Number of bad methods: ", bad)
    random.shuffle(graphlist)
   
        
    
    return graphlist

def split_dataset(graphlist):
    print("splitting dataset")
    loader_train = DataLoader(graphlist[:int(len(graphlist)*0.8)], batch_size=32, shuffle=True)
    loader_test = DataLoader(graphlist[int(len(graphlist)*0.8):], batch_size=32, shuffle=False)
    return loader_train, loader_test

def get_loaders(cwe, mode):
    device = torch.device('cuda:1')
    print(cwe)
    if mode == "juliet":
        print("generating asts...")
        (astdict, vocablen, vocabdict) = createast(dirname = cwe, mode = mode)
        print("generating trees...")
        treedict = createseparategraph(
        astdict,
        vocablen,
        vocabdict,
        device,
        mode='astandnext',
        nextsib=True,
        ifedge=True,
        whileedge=True,
        foredge=True,
        blockedge=True,
        nexttoken=True,
        nextuse=True,
        )
        vocabdict = {v: k for k, v in vocabdict.items()}
        print("training word2vec...")
        model = get_word_embeddings(treedict, vocabdict)
        dfname = (cwe.replace('prep', 'classifications'))+'.csv'
        df = pd.read_csv(dfname)
        mydict = {k: g["Classification"].tolist() for k,g in df.groupby("Filename")}
        model = get_word_embeddings(treedict, vocabdict)
        graphlist = generate_graphlist_juliet(treedict, vocabdict, model, mydict, cwe)
        
 
 
    elif mode == "owasp":
        (astdict, vocablen, vocabdict) = createast(dirname='../data', mode = mode)
        dfname = '../expectedresults-1.1.csv'
        df = pd.read_csv(dfname)
        filenames = df['# test name']
        classifications = df[' real vulnerability']
        cwes = df[' cwe']
        treedict = createseparategraph(
        astdict,
        vocablen,
        vocabdict,
        device,
        mode='astandnext',
        nextsib=True,
        ifedge=True,
        whileedge=True,
        foredge=True,
        blockedge=True,
        nexttoken=True,
        nextuse=True,
        )
        vocabdict = {v: k for k, v in vocabdict.items()}
        model = get_word_embeddings(treedict, vocabdict)
        cat_res = {filenames[i]: classifications[i] for i in range(len(filenames))} 
        type_res = {filenames[i]: cwes[i] for i in range(len(filenames))} 
        graphlist = generate_graphlist_owasp(treedict, vocabdict, model, cat_res, type_res, cwe)
 
    return(split_dataset(graphlist))