import os
from collections import OrderedDict
import numpy as np
import torch
from tools.utils import may_mkdirs
import random

def seperate_weight_decay(named_params, lr, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        # if 'bias' in name:
        #     no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'lr': lr, 'weight_decay': 0.},
            {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]

def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)

    # --------------------- dangwei li TIP20 ---------------------
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # --------------------- AAAI ---------------------
    # pos_weights = torch.sqrt(1 / (2 * ratio.sqrt())) * targets
    # neg_weights = torch.sqrt(1 / (2 * (1 - ratio.sqrt()))) * (1 - targets)
    # weights = pos_weights + neg_weights

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

def prob2weight(targets, ratio):
    #ratio = torch.from_numpy(ratio).type_as(targets)

    weights = -0.5 * targets + 1.5
    weights[targets > 1] = 0.0

    return weights
    
def get_model_log_path(root_path, model_name):
    multi_attr_model_dir = os.path.join(root_path, model_name, 'img_model')
    may_mkdirs(multi_attr_model_dir)

    multi_attr_log_dir = os.path.join(root_path, model_name, 'log')
    may_mkdirs(multi_attr_log_dir)

    return multi_attr_model_dir, multi_attr_log_dir

class LogVisual:

    def __init__(self, args):
        self.args = vars(args)
        self.train_loss = []
        self.val_loss = []

        self.ap = []
        self.map = []
        self.acc = []
        self.prec = []
        self.recall = []
        self.f1 = []

        self.error_num = []
        self.fn_num = []
        self.fp_num = []

        self.save = False

    def append(self, **kwargs):
        self.save = False

        if 'result' in kwargs:
            self.ap.append(kwargs['result']['label_acc'])
            self.map.append(np.mean(kwargs['result']['label_acc']))
            self.acc.append(np.mean(kwargs['result']['instance_acc']))
            self.prec.append(np.mean(kwargs['result']['instance_precision']))
            self.recall.append(np.mean(kwargs['result']['instance_recall']))
            self.f1.append(np.mean(kwargs['result']['floatance_F1']))

            self.error_num.append(kwargs['result']['error_num'])
            self.fn_num.append(kwargs['result']['fn_num'])
            self.fp_num.append(kwargs['result']['fp_num'])

        if 'train_loss' in kwargs:
            self.train_loss.append(kwargs['train_loss'])
        if 'val_loss' in kwargs:
            self.val_loss.append(kwargs['val_loss'])

class attributes_Q(object):

    def __init__(self,N,C):

        self.N = N
        self.attri_num = np.zeros([N,2]).astype(np.int)
        self.attri_pointer = np.zeros([N,2]).astype(np.int)
        self.pop_pointer = np.zeros([N,2]).astype(np.int)
        
        self.attri_Q = []
        for i in range(N):
            temp = [[],[]]
            self.attri_Q.append(temp)

        self.max = 1024
        self.C = C

        self.index = []
        for i in range(N):
            temp = []
            index_1 = np.arange(self.max)
            index_2 = np.arange(self.max)
            random.shuffle(index_1)
            random.shuffle(index_2)
            temp.append(index_1)
            temp.append(index_2)
            self.index.append(temp)

        self.pop_weight = np.ones(N) * 0.5

    def update(self, feature, labels):

        feature = feature.cpu()
        labels = labels.cpu().int()

        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                if self.attri_num[j, labels[i, j]] < self.max:
                    self.attri_num[j, labels[i, j]] += 1
                    self.attri_Q[j][labels[i, j]].append(torch.reshape(feature[i, j, :], [1,-1]))
                else:
                    self.attri_Q[j][labels[i, j]][self.attri_pointer[j, labels[i, j]]] = torch.reshape(feature[i, j, :], [1,-1])
                    self.attri_pointer[j, labels[i, j]] = (self.attri_pointer[j, labels[i, j]] + 1)%self.max

    def fullfill(self,):

        count = 0
        for i in range(self.N):
            for j in [0,1]:
                if self.attri_num[i,j] < self.max:
                    res = (self.max//self.attri_num[i,j])+1
                    temp = []
                    for k in range(res):
                        temp = temp + self.attri_Q[i][j]
                    self.attri_Q[i][j] = temp[0:self.max]
                    self.attri_num[i,j] = self.max
                    count+=1

        print(f'{count} attributes are fullfilled!\n')

    def pop(self, batch_size):
        
        pop_features = []
        pop_labels = []
        for j in range(self.N):
            temp_features = torch.zeros([batch_size,self.C])
            temp_labels = torch.zeros([batch_size])
            for i in range(batch_size):
                lam = np.random.beta(1, 1, 1)

                if lam < self.pop_weight[j]:
                    label_index = 1
                else:
                    label_index = 0

                temp_features[i,:] = self.attri_Q[j][label_index][self.index[j][label_index][self.pop_pointer[j,label_index]]]

                self.pop_pointer[j,label_index] += 1
                if self.pop_pointer[j,label_index] >= self.max:
                    self.pop_pointer[j,label_index] = 0
                    random.shuffle(self.index[j][label_index])

                temp_labels[i] = label_index

            pop_features.append(temp_features)
            pop_labels.append(temp_labels)

        pop_features = torch.stack(pop_features,dim=1).cuda()
        pop_labels = torch.stack(pop_labels,dim=1).cuda()

        return pop_features, pop_labels.float()

def get_pkl_rootpath(dataset, zero_shot):
    root = os.path.join("./data", f"{dataset}")
    if zero_shot:
        data_path = os.path.join(root, 'dataset_zs_run0.pkl')
    else:
        data_path = os.path.join(root, 'dataset_all.pkl')  #

    return data_path

def get_reload_weight(model, bb='resnet50', data='PA100k', label = 'eval'):
    
    if bb == 'resnet50':
        if data == 'PA100k':
            model_path = 'YOUR PATH'
        elif data == 'PETA':
            if label == 'eval':
                model_path = 'YOUR PATH'
            elif label == 'all':
                model_path =  'YOUR PATH'
                
        elif data == 'RAP':
            if label == 'eval':
                model_path = 'YOUR PATH'
            elif label == 'all':
                model_path = 'YOUR PATH'
        
    elif bb == 'convnext':
        if data == 'PA100k':
            model_path =  'YOUR PATH'
        elif data == 'PETA':
            if label == 'eval':
                model_path = 'YOUR PATH'
            elif label == 'all':
                model_path =   'YOUR PATH'
        elif data == 'RAP':
            if label == 'eval':
                model_path = 'YOUR PATH'
            elif label == 'all':
                model_path = 'YOUR PATH'
        elif data == 'UPAR':
            model_path = 'YOUR PATH'
    else:
        model_path = None

    load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        pretrain_dict = load_dict['state_dicts']
        #print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")

    model_dict = model.state_dict()

    the_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}

    model_dict.update(the_dict)
    model.load_state_dict(model_dict, strict=True)

    return model
