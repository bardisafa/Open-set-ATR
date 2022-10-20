"""
Adapted from https://github.com/rshaojimmy/OSAD
"""


import pdb
import sys
sys.path.append('../')

import os
import os.path as osp
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from utils.functions import lab_conv
from torch.nn import DataParallel
import numpy as np
import h5py
import torch.nn.functional as F
import libmr
from models import *

from pdb import set_trace as st


WEIBULL_TAIL_SIZE = 20



def openmax(args, kdataloader_trn, kdataloader_tst, ukdataloader_tst, knownclass, model, model_mc):

    model.eval()
    model_mc.eval()

    activation_vectors, mean_activation_vectors, weibulls = precalc_weibull(args, kdataloader_trn, knownclass, model, model_mc)
    

    open_acc, known_acc, known_scores = openset_weibull(args, kdataloader_tst, knownclass, model, model_mc,
                        activation_vectors, mean_activation_vectors, weibulls, mode='closeset')

    unknown_scores = openset_weibull(args, ukdataloader_tst, knownclass, model, model_mc,
                        activation_vectors, mean_activation_vectors, weibulls)
   
    auc = plot_roc(known_scores, unknown_scores)
    
    # SaveEvaluation(args, known_acc, auc)
    return auc, known_acc, open_acc


def precalc_weibull(args, dataloader_train, knownclass, model, model_mc):
    # First generate pre-softmax 'activation vectors' for all training examples
    print("Weibull: computing features for all correctly-classified training data")
    activation_vectors = {}

    for _, (images, labels) in enumerate(dataloader_train):

        labels = lab_conv(knownclass, labels)

        images, labels = images.cuda(), labels.long().cuda()

        with torch.no_grad():
            _, emb = model(images, with_emb=True)
            logits = model_mc(emb)

        correctly_labeled = (logits.data.max(1)[1] == labels)
        labels_np = labels.cpu().numpy()
        logits_np = logits.data.cpu().numpy()
        for i, label in enumerate(labels_np):
            if not correctly_labeled[i]:
                continue
            # If correctly labeled, add this to the list of activation_vectors for this class
            if label not in activation_vectors:
                activation_vectors[label] = []
            activation_vectors[label].append(logits_np[i])
    print("Computed activation_vectors for {} known classes".format(len(activation_vectors)))
    for class_idx in activation_vectors:
        print("Class {}: {} images".format(class_idx, len(activation_vectors[class_idx])))

    # Compute a mean activation vector for each class
    print("Weibull computing mean activation vectors...")
    mean_activation_vectors = {}
    for class_idx in activation_vectors:
        mean_activation_vectors[class_idx] = np.array(activation_vectors[class_idx]).mean(axis=0)

    # Initialize one libMR Wiebull object for each class
    print("Fitting Weibull to distance distribution of each class")
    weibulls = {}
    for class_idx in activation_vectors:
        distances = []
        mav = mean_activation_vectors[class_idx]
        for v in activation_vectors[class_idx]:
            distances.append(np.linalg.norm(v - mav))
        mr = libmr.MR()
        tail_size = min(len(distances), WEIBULL_TAIL_SIZE)
        mr.fit_high(distances, tail_size)
        weibulls[class_idx] = mr
        print("Weibull params for class {}: {}".format(class_idx, mr.get_params()))

    return activation_vectors, mean_activation_vectors, weibulls      


def openset_weibull(args, dataloader_test, knownclass, model, model_mc, activation_vectors, mean_activation_vectors, weibulls, mode='openset'):


    # Apply Weibull score to every logit
    weibull_scores = []
    logits = []
    classes = activation_vectors.keys()

    running_corrects = 0.0

    epoch_size = 0.0
    
    correct=0
    # reclosslist = []
    for steps, (images, labels) in enumerate(dataloader_test):

        labels = lab_conv(knownclass, labels)
        images, labels = images.cuda(), labels.long().cuda()

        print("Calculate weibull_scores in step {}/{}".format(steps, len(dataloader_test)))
        
        with torch.no_grad():
            _, emb = model(images, with_emb=True)
            batch_logits_torch = model_mc(emb)


        batch_logits = batch_logits_torch.data.cpu().numpy()
        batch_weibull = np.zeros(shape=batch_logits.shape)

        for activation_vector in batch_logits:
            weibull_row = np.ones(len(knownclass))
            for class_idx in classes:
                mav = mean_activation_vectors[class_idx]
                dist = np.linalg.norm(activation_vector - mav)
                weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
            weibull_scores.append(weibull_row)
            logits.append(activation_vector)


        if mode == 'closeset':
            _, preds = torch.max(batch_logits_torch, 1)
            ind_unk = np.where(preds.data.cpu().numpy() < 0.95)[0]
            y_pred = batch_logits_torch.data.max(1, keepdim=True)[1]
            y_pred[ind_unk] = len(knownclass)
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            # statistics
            running_corrects += torch.sum(preds == labels.data)
            epoch_size += images.size(0)

    if mode == 'closeset':
        open_acc = correct.double()/epoch_size        
        running_corrects =  running_corrects.double() / epoch_size
        print('Test Acc: {:.4f}'.format(running_corrects))
        print('Open Test Acc: {:.4f}'.format(open_acc))

    weibull_scores = np.array(weibull_scores)
    logits = np.array(logits)

    openmax_scores = -np.log(np.sum(np.exp(logits * weibull_scores), axis=1))
    if mode == 'closeset':
        return open_acc, running_corrects, np.array(openmax_scores)
    else:
        return np.array(openmax_scores)


def plot_roc(known_scores, unknown_scores):
    from sklearn.metrics import roc_curve, roc_auc_score
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    print('AUC {:.03f}'.format(auc_score))
    
    return auc_score



def SaveEvaluation(args, known_acc, auc):

    # filefolder = osp.join('results', 'Test', 'accuracy', args.datasetname+'-'+args.split)
    # mkdir(filefolder)

    filepath = 'results/openmax_results/dataset_' + str(args.dataset) + '.txt'
    output_file = open(filepath,'w')
    output_file.write('Close-set Accuracy:\n'+str(np.array(known_acc.cpu())))
    output_file.write('\nOpen-set AUROC:\n'+str(auc))
    output_file.close()

