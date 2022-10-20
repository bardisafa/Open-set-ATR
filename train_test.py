import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import argparse
import torch.nn as nn
from models.Nets import ResClassifier_MME, ResNet18
from utils.functions import lab_conv, cat_split
from Openset_methods.OpenMax import openmax
from sklearn.metrics import roc_auc_score
from torch.distributions import Categorical
from tqdm import tqdm



random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

CLASSES = ['cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'cls_6', 'cls_unknw']

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--dataset', type=str, default='atr_visible', help="name of dataset")
    parser.add_argument('--method', type=str, default='cbc', help="method")

    args = parser.parse_args()

    return args

def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print(f'AUROC score: {auroc}')

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
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0]) ##### take max negative alone
    return open_loss_pos, open_loss_neg


def main():
    args = args_parser()

    '''
    Transforms:
    '''
    cifar_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          transforms.RandomRotation(30)])
    cifar_transform_valid = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ])


    '''
    Loading data
    '''
    # download and create datasets
    train_dataset = datasets.CIFAR10(root='cifar_data',
                                           train=True,
                                           transform=cifar_transform,
                                           download=True)
    valid_dataset = datasets.CIFAR10(root='cifar_data',
                                           train=False,
                                           transform=cifar_transform_valid,
                                           download=True)
    

    #known/unknown split
    knownclass = [1, 8, 0, 2, 7, 6]
    unknownclass = [3,4,5,9]
    print("Known Class=", knownclass)
    print("Unknown Class=", unknownclass)
    idx = cat_split(train_dataset, knownclass=knownclass) # to identify knownclass indices
    known_dataset = [train_dataset[x] for x in idx]
    
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(known_dataset,\
                batch_size=args.bs, shuffle=True)
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
    
    optim_encoder = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=0.0005, nesterov=True)

    params = list(model_bc.parameters()) + list(model_mc.parameters())
    optim_classifier = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                    weight_decay=0.0005, nesterov=True)

    

    for iteration in range(args.epochs):
        for batch_idx, data in enumerate(tqdm(train_loader)):
            images, labels = data
            labels = lab_conv(knownclass, labels)
            images, labels = images.cuda(), labels.long().cuda()

            optim_encoder.zero_grad()
            optim_classifier.zero_grad()
            
            _, emb = model(images, with_emb=True)
            log_logits = model_mc(emb)
            loss_multiclass = criterion(log_logits, labels)         
            out_open = model_bc(emb)
            out_open = out_open.view(log_logits.size(0), 2, -1) 
            open_loss_pos, open_loss_neg = bc_loss(out_open, labels)
            loss_open = 0.5 * (open_loss_pos + open_loss_neg) 

            out_open = out_open.view(images.size(0), 2, -1)

            ent_closed = 0.1*Categorical(logits = log_logits).entropy().mean(0)

            if args.method == 'cbc':
                net_loss = loss_multiclass + 1*loss_open + 1*ent_closed 
                net_loss.backward()
            else:
                net_loss = loss_multiclass + 0*loss_open + 0*ent_closed 
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
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=False)

    # testing
    total_preds = []
    total_label = []
    total_score = []
    total_ent = []
    correct = 0
    
    pred_unk_total = []
    pred_arr = []
    gt_arr = []

    
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

        if args.method=='softmax' or args.method=='openmax':
            pred_thr = 0.9
            pred_score, pred_idx = predict_prob.data.max(1, keepdim=True)
            pred_unk = pred_score
            ind_unk = np.where(pred_unk.cpu().numpy() < pred_thr)[0]
        elif args.method=='mls':
            mls_thr = 3
            mls_score, mls_idx = before_softmax.data.max(1, keepdim=True)
            pred_unk = mls_score
            ind_unk = np.where(pred_unk.cpu().numpy() < mls_thr)[0]
        elif args.method=='cbc':
            out_open = model_bc(emb)
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

    if args.method=='openmax':   
        print("OpenMax results")
        k_idx, u_idx = cat_split(valid_dataset, knownclass, unknownclass) # to identify knownclass indices
        k_ds_test = [valid_dataset[x] for x in k_idx]
        u_ds_test = [valid_dataset[x] for x in u_idx]

        k_loader_test = DataLoader(k_ds_test,\
                    batch_size=args.bs, shuffle=False)
        u_loader_test = DataLoader(u_ds_test,\
                    batch_size=args.bs, shuffle=False)

        auroc = openmax(args, train_loader, k_loader_test, u_loader_test, knownclass, model, model_mc)
    else:
        auroc = compute_auroc(pred_unk_total, (total_label.data.cpu().numpy()<int(len(knownclass))))

if __name__ == '__main__':
	main()
