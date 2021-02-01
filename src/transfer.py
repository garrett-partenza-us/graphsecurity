#Script to evaluate a Juliet model performance on the OWASP Benchmark CWE89 testcases (transfer learning?). 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import os
import javalang
from javaprep import *
from edge_index import edges
import os
from loaders import get_loaders, get_trees
import matplotlib.pyplot as plt
import random
from datetime import datetime
from datetime import date
import csv
import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn import metrics
from model import GCN

#Prepare OWASP Dataset
vocabdict, treedict = get_trees()
train, test = get_loaders(89, "owasp", vocabdict, treedict)

#Load pretrained Juliet CWE89 model
seed = 1234
model = GCN(seed)
model.load_state_dict(torch.load('../models/Juliet089ForOWASP.pt'))
model.eval()

#Make predictions on OWASP Data with Juliet model
track_a = []
for quiz in test:
    out = model(quiz.x, quiz.edge_index, quiz.batch) 
    pred = out.argmax(dim=1)
    track_a.append(metrics.accuracy_score(quiz.y.tolist(), pred.tolist()))
for quiz in train:
    out = model(quiz.x, quiz.edge_index, quiz.batch) 
    pred = out.argmax(dim=1)
    track_a.append(metrics.accuracy_score(quiz.y.tolist(), pred.tolist()))
    
#Print mean accuracy
print(statistics.mean(track_a))





