#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch as t
import torch.nn.functional as F
import numpy as np


'''
Sparse block codes toolbox for NVSA backend
'''

def cosine2pmf(sim,act="Softmax",s=40): 
    # Infer PMF from the similarities (e.g., cosine) 
    if act == "Softmax": 
        out = t.nn.functional.softmax(sim.view(1,-1)*s, dim=-1)
    elif act == "Identity": 
        out = t.nn.functional.normalize(sim.view(1,-1), p=1, dim = -1)
    return out

def block_binding2(x,y):
    """
    Bind two vectors together

    Parameters
    ----------
    x: torch FloatTensor (_, _k, _L)
        input vector 1.
    y: torch FloatTensor (_, _k, _L)
        input vector 2.

    Returns
    -------
    res: torch FloatTensor (_, _k, _L)
        bound output vector
    """
    res  = binding_circular(x,y)
    return res

def cyclic_shift(x,n):
    """
    Blockwise cyclic shift 

    Parameters
    ----------
    x: torch FloatTensor (_, _k, _L)
        input vector 1.
    n: int
        number of blocks to be shifted

    Returns
    -------
    out: torch FloatTensor (_, _k, _L)
        shifted output vector
    """
    return t.roll(x,n,dims=1)


def block_unbinding2(x,y): 
    """
    Unbind vector y from x

    Parameters
    ----------
    x: torch FloatTensor (_, _k, _L)
        input vector 1.
    y: torch FloatTensor (_, _k, _L)
        input vector 2.

    Returns
    -------
    res: torch FloatTensor (_, _k, _L)
        unbound output vector
    """
    res  = inv_binding_circular(x,y)
    return res 

def block_binding3(x,y,z): 
    return block_binding2(block_binding2(x,y),z)

def block_unbinding3(x,y,z): 
    return block_unbinding2(block_unbinding2(x,y),z)

def binding_circular(A, B, alpha=1):
    """
    Binds two block codes vectors by blockwise cicular convolution. 

    Parameters
    ----------
    A: torch FloatTensor (_, _k, _l)
        input vector 1
    B: torch FloatTensor  (_, _k, _l)
        input vector 2
    alpha: int, optional
        specifies multiplicative factor for number of shifts (Default value '1')
    Returns
    -------
    C: torch FloatTensor (_k)
        k-dimensional offset vector that is the result of the binding operation.
    """
    ndim = A.dim()
    # add batch dimension (1) if not there yet
    if ndim==2: 
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
    
    batchSize,k,l = A.shape
    
    # prepare inputs
    A = t.unsqueeze(A,1) # input
    B = t.unsqueeze(B,2) # filter weigths
    B = t.flip(B, [3]) # flip input
    B = t.roll(B, 1, dims=3) # roll by one to fit addition

    # reshape for single CONV
    A = t.reshape(A, (1, A.shape[0]*A.shape[2], A.shape[3]))
    B = t.reshape(B, (B.shape[0]*B.shape[1], B.shape[2], B.shape[3]))

    # calculate C = t.remainder(B+A*alpha, self._L)
    C = F.conv1d(F.pad(A, pad=(0,l-1), mode='circular'), B, groups=k*batchSize)
    
    C = t.reshape(C, (batchSize, k, l))

    # Remove batch dimension if it was not there intially
    if ndim==2: 
        C = C.squeeze(0)
    return C

def inv_binding_circular(C,A, alpha=1):
    """
    Inverse binding of two block codes vectors by blockwise cicular correlation. 

    Parameters
    ----------
    A: torch FloatTensor (_, _k, _l)
        input vector 1
    B: torch FloatTensor  (_, _k, _l)
        input vector 2
    alpha: int, optional
        specifies multiplicative factor for number of shifts (Default value '1')
    Returns
    -------
    C: torch FloatTensor (_k)
        k-dimensional offset vector that is the result of the binding operation.
    """

    ndim = A.dim()
    # add batch dimension (1) if not there yet
    if ndim==2: 
        A = A.unsqueeze(0)
        C = C.unsqueeze(0)
    batchSize,k,l = A.shape

    A = t.unsqueeze(A,1) # input
    C = t.unsqueeze(C,2) # filter weigths

    A = t.reshape(A, (1, A.shape[0]*A.shape[2], A.shape[3]))
    C = t.reshape(C, (C.shape[0]*C.shape[1], C.shape[2], C.shape[3]))
        
    B = F.conv1d(F.pad(A, pad=(0,l-1), mode='circular'), C, groups=k*batchSize)
    B = t.reshape(B, (batchSize, k, l))
        
    B = t.flip(B, [2]) # flip input
    B = t.roll(B, 1, dims=2) # roll by one to fit addition

    # Remove batch dimension if it was not there intially
    if ndim==2: 
        B = B.squeeze(0)

    return B



def match_prob(x,y, act=t.nn.Identity()): 
    '''
    Compute similarity between two block codes vectors

    Parameters
    ----------
    x: torch FloatTensor (B,k,l) 
        input vector 1
    y: complex vector (B,k,l)
        input vector 2 
    Output
    ------
    sim: torch FloatTensor (B,)
        output similarity 
    '''
    _,k,l = x.shape
    sim = 1/k*t.sum(x*y, dim=(1,2))
    sim = act(sim)
    return sim

def match_prob_0(x,act=t.nn.Identity()): 
    '''
    Compute similarity between a block codes vectors and the zero vector

    Parameters
    ----------
    x: torch FloatTensor (B,k,l) 
        input vector 1
    Output
    ------
    sim: torch FloatTensor (B,)
        output similarity 
    '''
    _,k,l = x.shape
    # Just add up all 0 elements of every block
    sim = 1/k*t.sum(x[:,:,:1], dim=(1,2))
    sim = act(sim)
    return sim

def match_prob_multi(x,y,act=t.nn.Identity()): 
    '''
    Compute similarity between a dictionary and a query vector

    Parameters
    ----------
    x: torch FloatTensor (scene_dim,k,l) 
        dictionary
    y: torch FloatTensor (k,l) 
        query vector
    Output
    ------
    sim: torch FloatTensor (scene_dim,)
        output similarity 
    '''

    _, k,l= x.shape
    y =y.unsqueeze(0)
    sim = 1/k*t.sum(x*y, dim=(1,2))
    sim = act(sim)
    return sim

def match_prob_multi_batched(dictionary,y,act=t.nn.Identity()): 
    '''
    Compute similarity between a dictionary and a batch of query vectors

    Parameters
    ----------
    x: torch FloatTensor (scene_dim,k,l) 
        dictionary
    y: torch FloatTensor (B,k,l) 
        query vector
    Output
    ------
    sim: torch FloatTensor (B, scene_dim,)
        output similarity 
    '''

    bs, k,l = y.shape
    dictionary =dictionary.unsqueeze(0)
    y = y.unsqueeze(1)
    sim = 1/k*t.sum(dictionary*y, dim=(2,3))
    sim = act(sim)
    return sim

def pmf2vec(dictionary,pmfs): 
    '''
    Map PMF to d-dimensional complex vector

    Parameters
    ----------
    dictionary: torch FloatTensor (scene_dim, k, l)
        codebook 
    pmfs: torch FloatTensor (B,scene_dim)

    Return 
    ------
    out: Tensor (B,k, l)
    '''
    scene_dim , k, l = dictionary.shape
    ndim = pmfs.dim() 
    bs = pmfs.shape[0]

    # batched version
    if ndim == 2:
        pmfs = pmfs.unsqueeze(0)
        bs = 1
    
    superpos = F.linear(pmfs.reshape(-1,scene_dim),dictionary.reshape(scene_dim,-1).transpose(1,0))
    
    # Reshape depending on batched/ no batched versions 
    superpos = superpos.view(-1,k,l) if ndim==2 else superpos.view(bs,-1,k,l)

    return superpos

def check_collision(codebook_block): 
    # check for collisions inside the codebook 
    sum = 0
    for k in range(codebook_block.shape[0]): 
        sum += match_prob_multi(codebook_block,codebook_block[k]).sum()

    return sum>codebook_block.shape[0]


def block_continuous_codebook(device,d=2048, scene_dim=12, k=1, rng = np.random.default_rng(seed=42), fully_orthogonal=True):
    '''
    Create continuous codebook with sparse block codes codewords. 
    

    Parameters
    ----------
    x: torch FloatTensor (scene_dim,k,l) 
        dictionary
    y: torch FloatTensor (B,k,l) 
        query vector
    Output
    ------
    sim: torch FloatTensor (B, scene_dim,)
        output similarity 
    '''

    codebook_block = sample_block_continuous_codebook(device,d,scene_dim,k,rng)
    # resample codebooks if we want to have fully orthogonality
    while fully_orthogonal and check_collision(codebook_block):
        codebook_block = sample_block_continuous_codebook(device,d,scene_dim,k,rng) 

    return codebook_block, codebook_block # TODO remove one of the codebooks

def sample_block_continuous_codebook(device,d=2048,scene_dim=511, k=1, rng = np.random.default_rng(seed=42)):

    l = d//k
    codebook_block = t.zeros((scene_dim,k,l)).to(device)

    # Zero element: in each block the first element is set to 1
    codebook_block[0,:,0] = 1

    # Sample first element
    for k_it in range(k):
        # sample index which is not l/2
        idx = rng.integers(1,l)
        codebook_block[1,k_it,idx]=1

    # Define the remaining codewords by binding 
    for i in range(2,scene_dim): 
        codebook_block[i]=block_binding2(codebook_block[i-1],codebook_block[1])

    return codebook_block

def block_discrete_codebook(device,d=2048,scene_dim=511,k=1,rng = np.random.default_rng(seed=42)):

    l = d//k
    codebook_block = t.zeros((scene_dim,k,l)).to(device)
    for scene in range(scene_dim):
        for k_it in range(k): 
            codebook_block[scene,k_it,rng.integers(0,l)]=1

    return codebook_block, codebook_block

