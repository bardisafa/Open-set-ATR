import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pdb
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
from torch.utils.data import DataLoader, ConcatDataset
import random
import argparse
import torch.nn as nn
import math
import time
from preprocess import ATRDataset, SquarePad
from models.Nets import classifier32, ModerateCNN, ConvNet, ResNet_DomainClassifier, ResClassifier_MME
from utils.functions import gradcon_score, AverageMeter, lab_conv, test_img, cat_split, get_grad_rep, get_data_loader, train, get_subsets, get_embedding
from models.task_net import ResNet18
from Openset_methods.OpenMax import openmax
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report

from torch.distributions import Categorical

import copy
from tqdm import tqdm


random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

CLASSES = ['cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'cls_unknw']

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--dataset', type=str, default='atr_visible', help="name of dataset")

    args = parser.parse_args()

    return args

def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print(f'AUROC: {auroc}')

    return auroc
def bc_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()  ##### torch.Size([36, 20]) - zeros
    label_range = torch.range(0, out_open.size(0) - 1).long()  ##### label_range - batch size
    label_p[label_range, label] = 1  ###### set label to 1 - [0,0,0,0,....,1,0,0,0]
    label_n = 1 - label_p  ###### label_n - reamining all 1 - [1,1,1,1,,,,,0,1,1,1]
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    # open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
    #                                             1e-8) * label_n, 1)[0]) ##### take max negative alone
    open_loss_neg = torch.mean(torch.sum(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open

def main():
    args = args_parser()

    knownclass = [2,6,5,3,0]
    kk_data = [2,6,5]
    ku_data = [3,0]
    unknownclass = [1,4,7]
    '''
    Transforms:
    each dimension of images divided by two due to memory constraints
    '''
    transform=transforms.Compose([
    SquarePad(),
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    valid_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

    '''
    Loading data
    '''
    train_dataset_1000 = ATRDataset(args, csv_path='../atr_dataset_old/range/1000/ir_2003_train.csv', transform=transform)
    train_dataset_2000 = ATRDataset(args, csv_path='../atr_dataset_old/range/2000/ir_2007_train.csv', transform=transform)    
    train_dataset_3000 = ATRDataset(args, csv_path='../atr_dataset_old/range/3000/ir_2011_train.csv', transform=transform)
    train_dataset = ConcatDataset([train_dataset_1000, train_dataset_2000, train_dataset_3000])
    idx = cat_split(train_dataset, knownclass=knownclass) # to identify knownclass indices
    known_dataset = [train_dataset[x] for x in idx]
    
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCELoss()
    train_loader = DataLoader(known_dataset,\
                batch_size=args.bs, shuffle=True)
    mls_val = 0
    opmax_val = 0
    for n in range(3):
        print("N = ", n)
        # closed-set classifier    
        model = ResNet18(num_cls=len(knownclass))
        model.cuda()
        model.train() 

        model_mc = ResClassifier_MME(num_classes=len(knownclass),
                            norm=False, input_size=512)

        model_bc = ResClassifier_MME(num_classes=2 * len(knownclass),
                            norm=False, input_size=512)
        model_mc.cuda()
        model_bc.cuda() 
        model_mc.train()
        model_bc.train() 

        # xxx = get_grad_rep(known_dataset,5,model,criterion = nn.BCELoss())
        


        optim_encoder = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=0.0005, nesterov=True)

        params = list(model_bc.parameters()) + list(model_mc.parameters())
        optim_classifier = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                        weight_decay=0.0005, nesterov=True)

        

        for iteration in range(args.epochs):

            grad_losses = AverageMeter()
            for batch_idx, data in enumerate(tqdm(train_loader)):
                images, labels = data
                labels = lab_conv(knownclass, labels)
                images, labels = images.cuda(), labels.long().cuda()

                optim_encoder.zero_grad()
                optim_classifier.zero_grad()
                
                _, emb = model(images, with_emb=True)
                log_logits = model_mc(emb)
                loss_multiclass = criterion(log_logits, labels)

                        

                # out_open = model_bc(emb.detach().clone())
                out_open = model_bc(emb)

                out_open = out_open.view(log_logits.size(0), 2, -1) 
                open_loss_pos, open_loss_neg = bc_loss(out_open, labels)
                loss_open = 0.5 * (open_loss_pos + open_loss_neg) 
                #net_loss = loss_multiclass + loss_open

                out_open = out_open.view(images.size(0), 2, -1)
                ent_open = open_entropy(out_open)
                #net_loss = loss_multiclass + loss_open + 0.1*ent_open

                ent_closed = 0.1*Categorical(logits = log_logits).entropy().mean(0)
                pdb.set_trace()
                #net_loss = loss_multiclass + loss_open + 0.1*ent_closed

                net_loss = loss_multiclass + 1*loss_open + 0*ent_closed + 0*ent_open 
                net_loss.backward()

        

                optim_encoder.step()
                optim_classifier.step()

                if batch_idx % 50 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tnet_Loss: {:.6f} \tmulti_Loss: {:.6f} \topen_Loss: {:.6f}'.format(
                        iteration, batch_idx * len(images), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), net_loss.item(), loss_multiclass.item(), loss_open.item()))

        '''
        Testing
        '''
    
        model.eval()
        model_mc.eval()
        model_bc.eval()
        
        valid_dataset1000 = ATRDataset(args, csv_path='../atr_dataset_old/range/1000/ir_2003_test.csv', transform=valid_transform)
        valid_dataset2000 = ATRDataset(args, csv_path='../atr_dataset_old/range/2000/ir_2007_test.csv', transform=valid_transform)
        valid_dataset3000 = ATRDataset(args, csv_path='../atr_dataset_old/range/3000/ir_2011_test.csv', transform=valid_transform)
        valid_dataset = ConcatDataset([valid_dataset1000,valid_dataset2000,valid_dataset3000])
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        # testing
        total_preds = []
        total_label = []
        total_score = []
        total_ent = []
        correct = 0
        
        pred_unk_total = []
        pred_arr = []
        gt_arr = []

        softmax = False
        mls = False
        bc = True
        # opmax = True
        for idx, (data, target) in enumerate(valid_loader):
            target = lab_conv(knownclass, target)
            data, target = data.cuda(), target.long().cuda()

            with torch.no_grad():
                _, emb = model(data, with_emb=True)
                before_softmax = model_mc(emb)
            
            predict_prob = F.softmax(before_softmax, dim=1)
            predict_ent = -torch.sum(predict_prob*torch.log(predict_prob+1e-20), 1)
    
            pred = before_softmax.data.max(1)[1]

            if softmax:
                pred_thr = 0.9
                pred_score, pred_idx = predict_prob.data.max(1, keepdim=True)
                pred_unk = pred_score
                ind_unk = np.where(pred_unk.cpu().numpy() < pred_thr)[0]
            elif mls:
                mls_thr = 3
                mls_score, mls_idx = before_softmax.data.max(1, keepdim=True)
                pred_unk = mls_score
                ind_unk = np.where(pred_unk.cpu().numpy() < mls_thr)[0]
            elif bc:
                out_open = model_bc(emb)
                # pdb.set_trace()
                out_open = F.softmax(out_open.view(before_softmax.size(0), 2, -1),1) 
                tmp_range = torch.range(0, before_softmax.size(0)-1).long().cuda()  
                pred_unk = out_open[tmp_range, 1, pred] 
                ind_unk = np.where(pred_unk.data.cpu().numpy() < 0.9)[0]
                
                

            

        
            # get the index of the max log-probability
            y_pred = before_softmax.data.max(1, keepdim=True)[1]
            y_pred[ind_unk] = len(knownclass)
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            if len(gt_arr) == 0:
                pred_unk_total = pred_unk.detach().cpu().numpy()
                pred_arr = y_pred.cpu().numpy()
                gt_arr = target.cpu().numpy()
            else:
                pred_unk_total = np.append(pred_unk_total, pred_unk.detach().cpu().numpy())
                pred_arr = np.append(pred_arr, y_pred.cpu().numpy())
                gt_arr = np.append(gt_arr, target.cpu().numpy())
            
            if(idx==0):
                total_preds = y_pred
                total_label = target
                total_ent   = predict_ent.view(target.size(0), 1)
                total_score = before_softmax.max(1)[0].view(target.size(0), 1)
            else:
                total_preds = torch.cat([total_preds, y_pred], 0)
                total_label = torch.cat([total_label, target], 0)
                total_ent   = torch.cat([total_ent, predict_ent.view(target.size(0), 1)], 0)
                total_score = torch.cat([total_score, before_softmax.max(1)[0].view(target.size(0), 1)], 0)

        print(classification_report(gt_arr, pred_arr, target_names=CLASSES))
        aa = [softmax, mls, bc]
        bb = ['softmax', 'mls', 'bc']
        for i in range(len(aa)):
            if aa[i]:
                print(bb[i])
                auroc = compute_auroc(pred_unk_total, (total_label.data.cpu().numpy()<int(len(knownclass))))
                mls_val += auroc
                if n==2:
                    mls_val = mls_val/3
                print(bb[i]+" average AUROC="+ str(mls_val))
                

        if softmax:
            
            print("OpenMax results")
            k_idx, u_idx = cat_split(valid_dataset, knownclass, unknownclass) # to identify knownclass indices
            k_ds_test = [valid_dataset[x] for x in k_idx]
            u_ds_test = [valid_dataset[x] for x in u_idx]

            k_loader_test = DataLoader(k_ds_test,\
                        batch_size=args.bs, shuffle=False)
            u_loader_test = DataLoader(u_ds_test,\
                        batch_size=args.bs, shuffle=False)

            auc = openmax(args, train_loader, k_loader_test, u_loader_test, knownclass, model, model_mc)
            opmax_val += auc
            if n==2:
                opmax_val = opmax_val/3
                print("OpenMax average AUROC="+ str(opmax_val))
        

if __name__ == '__main__':
	main()
