from tkinter import FALSE
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import random 


random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)


def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label))
    for j in range(len(label)):
        for i in range(len(knownclass)):

            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))
                break
            else:
                label_convert[j] = len(knownclass)
             
    return label_convert

def lab_conv_binary(knownclass, label):
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label))
    for j in range(len(label)):
        for i in range(len(knownclass)):

            if label[j] == knownclass[i]:
                label_convert[j] = 1
                break
            else:
                label_convert[j] = 0
             
    return label_convert

def test_img(net_g, model_mc, datatest, args, knownclass=None, base=None):
    l = len(knownclass)
    net_g.eval()
    model_mc.eval()
    # testing
    total_preds = []
    total_label = []
    total_score = []
    total_ent = []
    test_loss = 0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)

    for idx, (data, target) in enumerate(data_loader):
        target = lab_conv(knownclass, target)
        data, target = data.cuda(), target.long().cuda()
        # print(target)
        with torch.no_grad():
            _, emb = net_g(data, with_emb=True)
            before_softmax = model_mc(emb)
            # before_softmax = net_g((data))#*255.0
        
        predict_prob = F.softmax(before_softmax, dim=1)
        predict_ent = torch.sum(-predict_prob*torch.log(predict_prob+1e-20), 1)
        
        # get the index of the max log-probability
        y_pred = before_softmax.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        
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

    total_label = total_label.view(-1, 1)
    
    fpr, tpr, _ = roc_curve((total_label.data.cpu().numpy()<int(l)),\
                                total_score.data.cpu().numpy(), pos_label=True)
    auc_test = auc(fpr, tpr)
    fpr_ent, tpr_ent, _ = roc_curve((total_label.data.cpu().numpy()<int(l)),\
                                     total_ent.data.cpu().numpy(), pos_label=False)
    auc_ent = auc(fpr_ent, tpr_ent)
    
    subset_preds = total_preds[total_label<int(l)]
    subset_label = total_label[total_label<int(l)]
    subset_correct = subset_preds.eq(subset_label.data.view_as(subset_preds)).long().cpu().sum()
    subset_accuracy = 100.00 * subset_correct / float(subset_preds.size(0))
    
    # constant for classes
    classes = ('Known 0', 'Known 1', 'Known 2', 'Known 3', 'Known 4',
            'Unknowns')

    # # Build confusion matrix
    # cf_matrix = confusion_matrix(total_label.data.cpu().numpy(), total_preds.data.cpu().numpy())
    # df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
    #                     columns = [i for i in classes])
    # plt.figure(figsize = (12,7))
    # sn.heatmap(df_cm, annot=True)

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, subset_accuracy, auc_test, auc_ent 


def cat_split(train_dataset, knownclass, unknownclass=None):
    '''
    Inputs: knownclasses, dataset
    Output: idxs
    '''
    
    
    '''
    Loading data
    '''
    labels = []
    for i in range(len(train_dataset)):
        labels.append(train_dataset[i][1])
    print(len(train_dataset), set(labels))

    k_idx = [i for i, e in enumerate(labels) if e in knownclass]
    
    if unknownclass:
        u_idx = [i for i, e in enumerate(labels) if e in unknownclass]
        return k_idx, u_idx
    else:
        return k_idx#, u_idx


def get_data_loader(data, labels, batch_size):
    #prepare the datasets for openset classifier
    train_data = []
    for i in range(len(labels)):
        train_data.append([data[i], labels[i]])
    loader = DataLoader(train_data,\
                batch_size=batch_size, shuffle=True)
    return train_data, loader


def get_embedding(model, loader, num_classes, args, with_emb=False, emb_dim=512):
    model.eval()
    lamda = 0
    embedding = torch.zeros([len(loader.sampler), num_classes])
    embedding_pen = torch.zeros([len(loader.sampler), emb_dim])
    # embedding_temp = torch.tensor([]).cuda()
    labels = torch.zeros(len(loader.sampler))
    preds = torch.zeros(len(loader.sampler))
    batch_sz = args.bs
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            batch_sz = data.shape[0]
            if with_emb:
                _, feature = model(x=data, with_emb=True)
                embedding_pen[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feature.shape[0]), :] = feature.cpu()
            else:
                feature = model(data)
                embedding[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feature.shape[0]), :] = feature.cpu()
            
            # embedding[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, embedding_temp.shape[0]), :] = embedding_temp.cpu()
            labels[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feature.shape[0])] = target
            preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feature.shape[0])] = feature.argmax(dim=1, keepdim=True).squeeze()
            #pdb.set_trace()
    return embedding, labels, preds, embedding_pen


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.prev = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


