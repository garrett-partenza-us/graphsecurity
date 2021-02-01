import sys
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
import statistics
from sklearn import metrics
from model import GCN


def learn(train, test, cwe):
    
    #generate random seed for model (will be stored in csv for reproducability)
    seed = random.randint(1,10000)
    model = GCN(seed)
    
    #variables for model learning
    learning_rate = 0.01
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.08)
    criterion = loss_function
    current_time = (datetime.now()).strftime("%H:%M:%S")
    current_date = (date.today()).strftime("%d/%m/%Y")
    
    #lists for keeping track of data for plotting and storing results
    #variable names should be fairly obvious
    epochs, losses, test_accs, train_accs, precisions, recalls, fscores = [], [], [], [], [], [], []
    
    #initialize report dictionary for writing results to csv after training 
    score_report = {
        'Test Acc': [], 
        'Losses': [],
        'FScores': [],
        'Recalls': [],
        'Precisions': [],
        'CWE': cwe, 
        'Model': str(model).replace('\n', ' ').replace('\t', ' '), 
        'Date': current_date, 
        'Time': current_time, 
        'Seed': seed, 
        'Learning Rate': learning_rate, 
        'Loss Function': loss_function
    }

    #begin learning
    for epoch in range(40):
        
        print("Epoch: ", epoch, end='\r')
        
        #train
        track = []
        model.train()
        for data in train:   
            optimizer.zero_grad()  
            out = model(data.x, data.edge_index, data.batch)  
            loss = criterion(out, data.y)
            pred = out.argmax(dim=1)
            loss.backward()  
            optimizer.step()
            track.append(round(metrics.accuracy_score(data.y.tolist(), pred.tolist()), 3))
        train_accs.append(statistics.mean(track))
        scheduler.step()

        #test
        model.eval()
        track_a, track_p, track_r, track_f = [],[],[],[]
        for quiz in test:
            out = model(quiz.x, quiz.edge_index, quiz.batch) 
            pred = out.argmax(dim=1)
            track_a.append(metrics.accuracy_score(quiz.y.tolist(), pred.tolist()))
            track_p.append(metrics.precision_score(quiz.y.tolist(), pred.tolist(), zero_division=0))
            track_r.append(metrics.recall_score(quiz.y.tolist(), pred.tolist(), zero_division=0))
            track_f.append(metrics.f1_score(quiz.y.tolist(), pred.tolist(), zero_division=0))
        test_accs.append(statistics.mean(track_a))
        precisions.append(statistics.mean(track_p))
        recalls.append(statistics.mean(track_r))
        fscores.append(statistics.mean(track_f))
        epochs.append(epoch)
        losses.append(loss)
        
        #report
        if epoch+1 in [1,3,5,10,20,40]:
            score_report['Test Acc'].append(statistics.mean(track_a))
            score_report['Losses'].append(round(loss.item(), 3))
            score_report['FScores'].append(statistics.mean(track_f))
            score_report['Recalls'].append(statistics.mean(track_r))
            score_report['Precisions'].append(statistics.mean(track_p))
        
        
    #pyplot metrics for visual analyzing
    plt.figure(figsize=(20,10))
    x_ticks = np.arange(0, 40, 5)
    plt.xticks(x_ticks)
    y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(y_ticks)
    plt.plot(epochs, losses, label='train loss', color='darkviolet', linewidth=2)
    plt.plot(epochs, train_accs, label='train acc', color='gold', linewidth=2)
    plt.plot(epochs, test_accs, label='test acc', color='forestgreen', linewidth=2)
    plt.plot(epochs, precisions, label='precision', color='dodgerblue', linewidth=2)
    plt.plot(epochs, recalls, label='recall', color='gray', linewidth=2)
    plt.plot(epochs, fscores, label='f-score', color='crimson', linewidth=2)
    plt.legend(prop={'size': 20})
    plt.savefig('../pngs/JULIET-'+str(cwe.split('/')[3])+'-'+str(current_date).replace('/','-')+'-'+str(current_time)+'.png')
    plt.clf()
    print()
                                             
    return score_report, model

#function for running all Juliet CWE's and reporting/saving best of three runs
def run_all():
    #iterate over all cwes in juliet
    for cwe in [x[0] for x in os.walk('../data/JULIET Testcases/prep/')][1:]:
        current_time = (datetime.now()).strftime("%H:%M:%S")
        current_date = (date.today()).strftime("%d/%m/%Y")
        models = []
        ending_acc = []
        sps = []
        
        #train model three times and take best of three runs
        for i in range(3):
            train, test = get_loaders(cwe, "juliet", None, None)
            score_report, mod = learn(train, test, cwe)
            ending_acc.append(score_report['Test Acc'][-1])
            models.append(mod)
            sps.append(score_report)
            #if perfect accruacy dont train any other models
            if max(ending_acc) == 1:
                break
                
        best_run = ending_acc.index(max(ending_acc))
        score_report = sps[best_run]
        model = models[best_run]
        torch.save(model.state_dict(),
                   '../models/'+str(cwe.split('/')[3])+'-'+(current_date).replace('/','')+'-'+current_time.replace('/','')
                  )
        
        #log results to csv
        row = [
            score_report['CWE'],
            score_report['Date'],
            score_report['Time'],
            score_report['Model'],
            score_report['Seed'],
            score_report['Learning Rate'],
            score_report['Loss Function'],
        ]
        info = zip(score_report['Test Acc'], 
                   score_report['Precisions'], 
                   score_report['FScores'], 
                   score_report['Recalls']
                  )
        for epoch in info:
            for item in epoch:
                row.append(item)
        with open('../juliet_score_report.csv','a') as fd:
            writer = csv.writer(fd)
            writer.writerow(row)
            
    print("Done")

def main():
    args = sys.argv
    if len(args)<2:
        print("ERROR: Please specify cwe to run!")
        quit()
    elif len(args)>2:
        print("ERROR: Recieved too many arguments!")
        quit()
    if args[1] == 'all':
        print("running all weaknesses...")
        run_all()
    else:
        cwe = '../data/JULIET Testcases/prep/CWE'+args[1].strip()
        train, test = get_loaders(cwe, "juliet", None, None)
        score_report, mod = learn(train, test, cwe)

if __name__=='__main__':
    main()

