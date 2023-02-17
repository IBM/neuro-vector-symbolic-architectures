#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import torch as t
from .vsa_block_utils import block_binding2, block_binding3, match_prob, block_unbinding2, block_unbinding3, pmf2vec, match_prob_0, match_prob_multi, cyclic_shift, cosine2pmf 
VSA_WEIGHT= 2
EPS = 10 ** (-20)
# EPS = 10 ** (-38)

######################################### PGM of detectors and executors ########################### 
class vsa_rule_detector_PGM(object):
    def __init__(self,detector_act="threshold",detector_m=-1, detector_s=1,**kwargs):
        
        if detector_act == "threshold": 
            self.match_act = t.nn.Threshold(detector_m,0)
        elif detector_act == "hardtanh": 
            self.match_act = t.nn.Hardtanh(0,1)
        else: 
            self.match_act = getattr(t.nn, detector_act)()
        self.pmf2vec = pmf2vec

    def union(self,p_vsa_d): 
        row1 = block_binding3(p_vsa_d[:,0], p_vsa_d[:,1], p_vsa_d[:,2])
        row2 = block_binding3(p_vsa_d[:,3], p_vsa_d[:,4], p_vsa_d[:,5])
        row_prob = match_prob(row1, row2,self.match_act)

        # Constraint that two rows don't have the same sequence
        row1_shift = block_binding3(p_vsa_d[:,0], cyclic_shift(p_vsa_d[:,1],1), cyclic_shift(p_vsa_d[:,2],2))
        row2_shift = block_binding3(p_vsa_d[:,3], cyclic_shift(p_vsa_d[:,4],1), cyclic_shift(p_vsa_d[:,5],2))
        row_prob_shift = match_prob(row1_shift, row2_shift,self.match_act)

        # Neighboring panels should not have the same value
        flag2 = match_prob(p_vsa_d[:,0],p_vsa_d[:,1],self.match_act)# first column
        flag3 = match_prob(p_vsa_d[:,1],p_vsa_d[:,2],self.match_act) # last column
        flag4 = match_prob(p_vsa_d[:,3],p_vsa_d[:,4],self.match_act)# first column
        flag5 = match_prob(p_vsa_d[:,4],p_vsa_d[:,5],self.match_act) # last column
        flag6 = match_prob(p_vsa_d[:,6],p_vsa_d[:,7],self.match_act) # last column

        # Combine all probabilities
        prob_0 = row_prob*(1-row_prob_shift)*(1-flag2)*(1-flag3)*(1-flag4)*(1-flag5)*(1-flag6)+EPS

        return prob_0, VSA_WEIGHT


    
    def progression_plus(self,p_vsa_c, x_target):
        p1_12 = block_unbinding2(p_vsa_c[:,1],p_vsa_c[:,0])
        p1_23 = block_unbinding2(p_vsa_c[:,2],p_vsa_c[:,1])    
        p1_13 = block_unbinding2(p_vsa_c[:,2],p_vsa_c[:,0])
        p2_12 = block_unbinding2(p_vsa_c[:,4],p_vsa_c[:,3])
        p2_23 = block_unbinding2(p_vsa_c[:,5],p_vsa_c[:,4])    
        p2_13 = block_unbinding2(p_vsa_c[:,5],p_vsa_c[:,3])    
        p3_12 = block_unbinding2(p_vsa_c[:,7],p_vsa_c[:,6]) 
        s1 = match_prob(p1_12,x_target,self.match_act)
        s2 = match_prob(p1_23,x_target,self.match_act)
        s3 = match_prob(p2_12,x_target,self.match_act)
        s4 = match_prob(p2_23,x_target,self.match_act)
        s5 = match_prob(p3_12,x_target,self.match_act)
        s6 = match_prob(p1_13,block_binding2(x_target,x_target),self.match_act)
        s7 = match_prob(p2_13,block_binding2(x_target,x_target),self.match_act)

        s0 = t.clamp(match_prob_0(p1_12,self.match_act),min=0,max=1)
        result = s1*s2*s3*s4*s5*s6*s7*(1-s0)+EPS

        return  result,VSA_WEIGHT

        
class vsa_rule_executor_PGM(object):
    def __init__(self,executor_act="threshold",executor_m=1, executor_s=1, executor_cos2pmf_act="Identity",**kwargs): 
        self.s = executor_s
        self.match_act = t.nn.Threshold(executor_m,0) if executor_act == "threshold" else getattr(t.nn, executor_act)()
        self.cos2pmf_act = executor_cos2pmf_act 
        self.pmf2vec = pmf2vec

    def union(self,vsa_cb_discrete_a,p_vsa_d): 
        temp1 = block_binding3(p_vsa_d[0], p_vsa_d[1], p_vsa_d[2])
        pred = block_unbinding3(temp1, p_vsa_d[6], p_vsa_d[7])
        sim = match_prob_multi(vsa_cb_discrete_a,pred,self.match_act)
        pred = cosine2pmf(sim,self.cos2pmf_act,self.s)
        return pred

    def progression_plus(self,vsa_cb_cont_a,p_vsa_c, x_target):
        pred = block_binding2(p_vsa_c[7],x_target)
        sim = match_prob_multi(vsa_cb_cont_a,pred,self.match_act)
        pred = cosine2pmf(sim,self.cos2pmf_act, self.s)
        return pred
