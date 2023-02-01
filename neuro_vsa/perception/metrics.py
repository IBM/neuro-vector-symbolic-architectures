#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#


import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np

LOG_EPSILON = 1e-39

def log(x):
    return t.log(x + LOG_EPSILON)

def generate_IM(n_in,dims=(5,6,10,4),rng=None):
    '''
    Generate bipolar codebooks and matrix W 

    Parameters
    ----------
    n_in:   int
        vector dimension (d)
    dims:   tupel 
        codebook sizes (type, size, color, pos)
    rng:    numpy random number generator
        
    Return
    ------
    matrix: torch tensor
        Complete matrix W
    im_dict: torch tensor
        Codebooks
    '''
    num_type, num_size, num_color, num_pos = dims
    max_dic = num_pos if num_pos>num_color else num_color
    n_out = num_type*num_pos*num_size*num_color + num_pos # +num_pos for object detection
    num_per_pos = num_color*num_size*num_type+1
    im_dict = ((rng.integers(0,2,(4,max_dic,n_in))*2-1)).astype(np.float32)
    object_dict = ((rng.integers(0,2,(num_pos,n_in))*2-1)).astype(np.float32)
    matrix = np.zeros((n_out, n_in)).astype(np.float32)
    for pos in range(num_pos):
        for col in range(num_color):
            for siz in range(num_size):
                for typ in range(num_type):
                    matrix[pos*num_per_pos+col*num_size*num_type+siz*num_type+typ]=im_dict[0,typ]*im_dict[1,siz]*im_dict[2,col]*im_dict[3,pos]
        matrix[(pos+1)*num_per_pos-1] = object_dict[pos]
    return t.from_numpy(matrix), t.from_numpy(im_dict)

class fixCos(nn.Module):
    '''
    Linear layer with (potentially) fixed weights and 
    cosine similarity metric  

    Parameters
    ----------
    num_features:   int
        input dimensionality
    num_classes:    int
        placeholder
    mat:            torch tensor
        Weight matrix (W)
    fixed_weights:  boolean
        Fix weight matrix (W) if activated
    s:              float
        Inverse softmax temperature
    trainable_s:    boolean
        Trainable inverse softmax temperature if activated    
    '''
    def __init__(self, num_features, num_classes, mat, fixed_weights=True, s=1.0, trainable_s=True):
        super(fixCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = Parameter(t.Tensor([s]), requires_grad=trainable_s)
        self.mat = mat/math.sqrt(num_features)
        self.W = Parameter(mat, requires_grad=False) if fixed_weights else Parameter(mat)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        output = self.s * logits
        return output

    def get_s(self):
        return self.s.data.item()

class marginalization(nn.Module):
    '''
    Maps cosine similarites to object PMFs by marginalization

    Parameters
    ----------
    s:              float 
        inverse softmax temperature
    trainable_s:    boolean
        Trainable inverse softmax temperature if activated    
    in_act:         str
        input activation
    exist_act:      str
        input activation for existence probability
    dims:           tuple
        dimensions of (color, size, type) 
    '''
    def __init__(self,s=1,m=0,trainable_s=False, in_act = "ReLU",exist_act="Identity",dims=(10,6,5)):
        super(marginalization, self).__init__()
        exist_readout, typ_readout, siz_readout, col_readout = get_marginalization_readout(dims)
        self.typ_readout = nn.Parameter(typ_readout,requires_grad=False)
        self.siz_readout = nn.Parameter(siz_readout,requires_grad=False)
        self.col_readout = nn.Parameter(col_readout,requires_grad=False)
        self.exist_readout = nn.Parameter(exist_readout,requires_grad=False)
        if trainable_s: 
            self.s_type = Parameter(t.Tensor([s]), requires_grad=trainable_s)
            self.s_size = Parameter(t.Tensor([s]), requires_grad=trainable_s)
            self.s_color = Parameter(t.Tensor([s]), requires_grad=trainable_s)
            self.s_exist = Parameter(t.Tensor([s]), requires_grad=trainable_s)
        else: 
            self.s_type, self.s_size, self.s_color, self.s_exist = s, s, s, s

        self.m = m
        self.in_act = getattr(nn,in_act)()
        self.exist_act = getattr(nn,exist_act)()

    def forward(self, x):
        x_1 = self.in_act(x-self.m)
        type_prob = F.softmax(self.s_type*F.linear(x_1,self.typ_readout), dim=-1)
        size_prob = F.softmax(self.s_size*F.linear(x_1,self.siz_readout), dim=-1)
        color_prob = F.softmax(self.s_color*F.linear(x_1,self.col_readout), dim=-1)
        exist_prob = F.softmax(self.s_exist*F.linear(self.exist_act(x),self.exist_readout), dim=-1)

        return [log(exist_prob),log(type_prob),log(size_prob),log(color_prob)]

    def get_s(self):
        return self.s_exist.data.item(),self.s_type.data.item(), self.s_size.data.item(), self.s_color.data.item() 

def get_marginalization_readout(dims): 
    '''
    Compute the binary matrix for doing marginalization

    Parameters
    ----------
    dims:           tuple
        dimensions of (color, size, type) 

    Return
    ------
    exist_readout:  torch tensor (2, nmax)
    typ_readout:  torch tensor (n_typ, nmax)
    siz_readout:  torch tensor (n_siz, nmax)
    col_readout:  torch tensor (n_col, nmax)
    '''
    n_col, n_siz, n_typ= dims
    nmax = n_col*n_siz*n_typ +1
    typ_readout = t.zeros(n_typ,nmax); siz_readout = t.zeros(n_siz,nmax)
    col_readout = t.zeros(n_col,nmax); exist_readout = t.zeros(2,nmax)
    
    for col in range(n_col):
        for siz in range(n_siz):
            for typ in range(n_typ): 
                idx = col*n_siz*n_typ + siz*n_typ + typ
                typ_readout[typ,idx] = 1.
                siz_readout[siz,idx] = 1.
                col_readout[col,idx] = 1.

    exist_readout[0,-1] = 1.
    exist_readout[1,:nmax-1] = 1.

    return exist_readout, typ_readout, siz_readout, col_readout


class hd_mult_frontend(nn.Module): 
    '''
    Maps cosine similarites to object PMFs by marginalization

    Parameters
    ----------
    # cosine readout parameters
    num_features:   int
        input dimensionality
    num_classes:    int
        placeholder
    mat:            torch tensor
        Weight matrix (W)
    fixed_weights:  boolean
        Fix weight matrix (W) if activated
    s:              float
        Inverse softmax temperature
    trainable_s:    boolean
        Trainable inverse softmax temperature if activated    

    # Marginalization parameters
    marg_s:              float 
        inverse softmax temperature
    marg_m:             float
        threshold (shift)
    marg_in_act:         str
        input activation
    marg_exist_act:      str
        input activation for existence probability
    num_pos:            int
        number of positions
    '''
    def __init__(self,num_features, num_classes, mat, fixed_weights=True, s=1.0, trainable_s=True,
                marg_s=1,marg_m=0,marg_in_act = "ReLU",marg_exist_act="Identity", num_pos=1):
        super(hd_mult_frontend, self).__init__()
        # init cosine readout 
        self.metric = fixCos(num_features = num_features, num_classes = num_classes, mat=mat,fixed_weights=fixed_weights,s=s,trainable_s=trainable_s)
        # init marginalization (no trainable s here)
        self.marg = marginalization(s=marg_s, m=marg_m, in_act=marg_in_act,exist_act=marg_exist_act)
        self.num_pos = num_pos

    def forward(self,x,B,N): 
        x = self.metric(x)
        x = self.marg(x.view(B,N,self.num_pos,-1))
        return x

    def store_s(self,writer,epoch):
        writer.add_scalar('marg-s/marg-s', self.metric.get_s(), epoch )

class randXentropyloss(nn.Module): 
    '''
    Random cross entropy loss for multi-label classification
    '''
    def __init__(self): 
        super(randXentropyloss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x,target, target_onhot):
        B = target.size(0)
        temp = (target != -1).clone().detach()
        temp = temp.float()
        temp1 = t.multinomial(temp,1).reshape(B,1)
        temp2 = t.arange(B).reshape(B,1).to(temp1.get_device())
        idx = t.cat((temp2,temp1), dim = 1)
        targ= target[idx[:,0],idx[:,1]] 
        loss = self.criterion(x,targ) 
        return loss

class addXentropyloss(nn.Module): 
    '''
    Additive cross entropy loss for multi-label classification (see Supplementary Note 1 Eq.(3))
    '''
    def __init__(self): 
        super(addXentropyloss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x,target, target_onehot):
        B = target.size(0)
        target_logit = t.sum(x*target_onehot,dim=-1, keepdim=True)
        logit = t.cat((target_logit,x),dim=1)
        new_target = t.zeros(B,dtype=t.int64).to(logit.get_device())
        loss = self.criterion(logit, new_target)
        return loss 

######################### PGM related frontend ############################################
class marginalization_line(nn.Module):
    '''
    Maps cosine similarites to object PMFs by marginalization for line constellation

    Parameters
    ----------
    s:              float 
        inverse softmax temperature
    m:              float
        shift/threshold input
    trainable_s:    boolean
        Trainable inverse softmax temperature if activated    
    in_act:         str
        input activation
    exist_act:      str
        input activation for existence probability
    dims:           int
        dimension on line
    '''
    def __init__(self,s=1,m=0,trainable_s=False,in_act="ReLU",exist_act="Identity",dims=10):
        super(marginalization_line, self).__init__()
        self.col_readout = t.eye(dims,dims+1); self.exist_readout = t.zeros(2,dims+1)
        self.exist_readout[0,-1] = 1.; self.exist_readout[1,:dims] = 1.
        self.col_readout = nn.Parameter(self.col_readout,requires_grad=False)
        self.exist_readout = nn.Parameter(self.exist_readout,requires_grad=False)
        if trainable_s: 
            self.s_color = Parameter(t.Tensor([s]), requires_grad=trainable_s)
            self.s_exist = Parameter(t.Tensor([s]), requires_grad=trainable_s)
        else: 
            self.s_color, self.s_exist = s, s
        self.m = m
        self.in_act = getattr(nn,in_act)()
        self.exist_act = getattr(nn,exist_act)()

    def forward(self, x):
        x_1 = self.in_act(x-self.m)
        color_prob = F.softmax(self.s_color*F.linear(x_1,self.col_readout), dim=-1)
        exist_prob = F.softmax(self.s_exist*F.linear(self.exist_act(x),self.exist_readout), dim=-1)

        return [log(exist_prob),log(color_prob)]

    def get_s(self):
        return self.s_exist.data.item(), self.s_color.data.item() 



class hd_mult_frontend_pgm(nn.Module): 
    '''
    Maps cosine similarites to object PMFs by marginalization

    Parameters
    ----------
    # cosine readout parameters
    num_features:   int
        input dimensionality
    num_classes:    int
        placeholder
    mat:            torch tensor
        Weight matrix (W)
    fixed_weights:  boolean
        Fix weight matrix (W) if activated
    s:              float
        Inverse softmax temperature
    trainable_s:    boolean
        Trainable inverse softmax temperature if activated    

    # Marginalization parameters
    marg_s:              float 
        inverse softmax temperature
    marg_m:             float
        threshold (shift)
    marg_in_act:         str
        input activation
    marg_exist_act:      str
        input activation for existence probability
    num_pos:            int
        number of positions
    '''
    def __init__(self,num_features, num_classes, mat, fixed_weights=True, s=1.0, trainable_s=True,
                marg_s=1,marg_trainable_s=True,marg_m=0,marg_in_act = "ReLU",marg_exist_act="Identity"):
        super(hd_mult_frontend_pgm, self).__init__()
        self.nline, self.nshape = 6, 9 #TODO not hardcode
        self.marg_trainable_s = marg_trainable_s
        self.metric = fixCos(num_features = num_features, num_classes = num_classes, mat=mat,fixed_weights=fixed_weights,s=s,trainable_s=trainable_s)

        self.marg_line = marginalization_line(s=marg_s, m=marg_m, trainable_s=marg_trainable_s, in_act=marg_in_act,exist_act=marg_exist_act)
        self.marg_shape = marginalization(s=marg_s, m=marg_m, trainable_s=marg_trainable_s, in_act=marg_in_act,exist_act=marg_exist_act,dims=(10,10,7)) # TODO not hardcode

    def forward(self,x,B,N): 
        x = self.metric(x)
        out_line = self.marg_line(x[:,:self.nline*11].view(B,N,self.nline,-1))
        out_shape = self.marg_shape(x[:,self.nline*11:].view(B,N,self.nshape,-1))
        return out_line, out_shape 

    def store_s(self,writer,epoch):

        # general s from fixCos
        writer.add_scalar('marg-s/marg-s', self.metric.get_s(), epoch )

        # store individual marginalization s if training active
        if self.marg_trainable_s:
            # from shape
            s_exist, s_type, s_size, s_color = self.marg_shape.get_s() 
            writer.add_scalar('marg-s/exist', s_exist, epoch )
            writer.add_scalar('marg-s/type', s_type, epoch )
            writer.add_scalar('marg-s/size', s_size, epoch )
            writer.add_scalar('marg-s/color', s_color, epoch )
            # from line
            s_exist, s_color = self.marg_line.get_s() 
            writer.add_scalar('marg-s/line-exist', s_exist, epoch )
            writer.add_scalar('marg-s/line-color', s_color, epoch )



def generate_IM_pgm(n_in, num_pos_line, num_pos_shape, rng):
    '''
    Generate bipolar codebooks and matrix W for PGM dataset 

    Parameters
    ----------
    n_in:   int
        vector dimension (d)
    num_pos_line: int 
        number of positions for line
    num_pos_shape: int
        number of positions for shape
    rng:    numpy random number generator
        
    Return
    ------
    matrix: torch tensor
        Complete matrix W
    im_dict: torch tensor
        Codebooks
    '''
    # type and size only for shape, color for both shape and line
    num_type, num_size, num_color = 7, 10, 10

    # line
    max_dic = num_color
    n_out = num_pos_line*num_color + num_pos_line # num_pos for object detection
    num_per_pos = num_color+1
    im_dict = ((rng.integers(0,2,(2,max_dic,n_in))*2-1)).astype(np.float32)
    object_dict = ((rng.integers(0,2,(num_pos_line,n_in))*2-1)).astype(np.float32)
    matrix_line = np.zeros((n_out, n_in)).astype(np.float32)
    for pos in range(num_pos_line):
        for col in range(num_color):
            matrix_line[pos*num_per_pos+col]=im_dict[0,col]*im_dict[1,pos]
        matrix_line[(pos+1)*num_per_pos-1] = object_dict[pos]

    # shape
    n_out = num_type*num_pos_shape*num_size*num_color + num_pos_shape # +num_pos for object detection
    num_per_pos = num_color*num_size*num_type+1
    im_dict = ((rng.integers(0,2,(4,max_dic,n_in))*2-1)).astype(np.float32)
    object_dict = ((rng.integers(0,2,(num_pos_shape,n_in))*2-1)).astype(np.float32)
    matrix_shape = np.zeros((n_out, n_in)).astype(np.float32)
    for pos in range(num_pos_shape):
        for col in range(num_color):
            for siz in range(num_size):
                for typ in range(num_type):
                    matrix_shape[pos*num_per_pos+col*num_size*num_type+siz*num_type+typ]=im_dict[0,typ]*im_dict[1,siz]*im_dict[2,col]*im_dict[3,pos]
        matrix_shape[(pos+1)*num_per_pos-1] = object_dict[pos]

    matrix = np.concatenate((matrix_line,matrix_shape),axis=0)    

    return t.from_numpy(matrix), t.from_numpy(im_dict)


