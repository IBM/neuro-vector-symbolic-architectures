#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import torch as t
from .vsa_block_utils import block_binding2, block_binding3, match_prob, block_unbinding2, block_unbinding3, pmf2vec, match_prob_0, match_prob_multi, match_prob_multi_batched, cosine2pmf 
VSA_WEIGHT= 2
EPS = 10 ** (-20)


######################################### Extended Versions of detectors and executors ########################### 
class vsa_rule_detector_extended(object):
    '''
    VSA backend: rule probability computation

    Parameters
    ----------
    detector_act:   str
        activation function applied after vector similarity computation
    detector_m:     str     
        threshold value if detector_act=threshold
    detector_s:     str
        place holder        # TODO remove 

    '''
    def __init__(self,detector_act="threshold",detector_m=-1, detector_s=1,**kwargs): 

        self.match_act = t.nn.Threshold(detector_m,0) if detector_act == "threshold" else getattr(t.nn, detector_act)()
        self.pmf2vec = pmf2vec

    def distribute_three(self,p_vsa_d):
        # compute binding along first two rows/columns 
        row1 = block_binding3(p_vsa_d[:,0], p_vsa_d[:,1], p_vsa_d[:,2])
        row2 = block_binding3(p_vsa_d[:,3], p_vsa_d[:,4], p_vsa_d[:,5])
        col1 = block_binding3(p_vsa_d[:,0], p_vsa_d[:,3], p_vsa_d[:,6])
        col2 = block_binding3(p_vsa_d[:,1], p_vsa_d[:,4], p_vsa_d[:,7])
        # Compute the row/column probability 
        row_prob = match_prob(row1, row2,self.match_act)
        col_prob = match_prob(col1, col2,self.match_act)

        # Additional constraints that panels should be distinct
        flag2 = match_prob(p_vsa_d[:,0],p_vsa_d[:,1],self.match_act)#  (1,1) and (1,2) 
        flag3 = match_prob(p_vsa_d[:,1],p_vsa_d[:,2],self.match_act) # (1,2) and (1,3)
        flag4 = match_prob(p_vsa_d[:,3],p_vsa_d[:,4],self.match_act)# (2,1) and (2,2)
        flag5 = match_prob(p_vsa_d[:,4],p_vsa_d[:,5],self.match_act) # (2,2) and (2,3)
        flag6 = match_prob(p_vsa_d[:,6],p_vsa_d[:,7],self.match_act) # (3,1) and (3,2)
        
        # Compute overall probability, EPS for numerical stability  
        prob_0 = row_prob*col_prob*(1-flag2)*(1-flag3)*(1-flag4)*(1-flag5)*(1-flag6)+EPS
        return prob_0, VSA_WEIGHT

    def constant(self,p_vsa_d): 
        # Compute similarity between panels within each row
        s1_12 = match_prob(p_vsa_d[:,0],p_vsa_d[:,1],self.match_act) # (1,1) and (1,2)
        s1_13 = match_prob(p_vsa_d[:,0],p_vsa_d[:,2],self.match_act) # (1,1) and (1,3)
        s2_12 = match_prob(p_vsa_d[:,3],p_vsa_d[:,4],self.match_act) # (2,1) and (2,2)
        s2_13 = match_prob(p_vsa_d[:,4],p_vsa_d[:,5],self.match_act) # (2,2) and (2,3)
        s3 = match_prob(p_vsa_d[:,6],p_vsa_d[:,7],self.match_act)    # (3,1) and (3,1)

        # Compute overall probability, EPS for numerical stability  
        result = s1_12*s1_13*s2_12*s2_13*s3+EPS
        return  result, VSA_WEIGHT

    
    def progression_plus(self,p_vsa_c, x_target):
        # Compute difference between neighboring panels 
        p1_12 = block_unbinding2(p_vsa_c[:,1],p_vsa_c[:,0])
        p1_23 = block_unbinding2(p_vsa_c[:,2],p_vsa_c[:,1])    
        p2_12 = block_unbinding2(p_vsa_c[:,4],p_vsa_c[:,3])
        p2_23 = block_unbinding2(p_vsa_c[:,5],p_vsa_c[:,4])    
        p3_12 = block_unbinding2(p_vsa_c[:,7],p_vsa_c[:,6]) 
        # Compute difference between first and last panel in each row
        p1_13 = block_unbinding2(p_vsa_c[:,2],p_vsa_c[:,0])
        p2_13 = block_unbinding2(p_vsa_c[:,5],p_vsa_c[:,3])    

        # Compare the delta vector with target delta vector (1, or 2)
        s1 = match_prob(p1_12,x_target,self.match_act)
        s2 = match_prob(p1_23,x_target,self.match_act)
        s3 = match_prob(p2_12,x_target,self.match_act)
        s4 = match_prob(p2_23,x_target,self.match_act)
        s5 = match_prob(p3_12,x_target,self.match_act)
        # Compare the delta vector with TWICE target delta vector (1, or 2)
        s6 = match_prob(p1_13,block_binding2(x_target,x_target),self.match_act)
        s7 = match_prob(p2_13,block_binding2(x_target,x_target),self.match_act)
        # check if delta is not zero
        s0 = t.clamp(match_prob_0(p1_12,self.match_act),min=0,max=1)

        # Compute overall probability, EPS for numerical stability  
        result = s1*s2*s3*s4*s5*s6*s7*(1-s0)+EPS
        return  result,VSA_WEIGHT

    def progression_minus(self,p_vsa_c, x_target):
        # Same implementation as progression_plus, only the 
        # order of the panels in the unbinding is inverted

        # Compute difference between neighboring panels 
        p1_12 = block_unbinding2(p_vsa_c[:,0],p_vsa_c[:,1])
        p1_23 = block_unbinding2(p_vsa_c[:,1],p_vsa_c[:,2])    
        p2_12 = block_unbinding2(p_vsa_c[:,3],p_vsa_c[:,4])
        p2_23 = block_unbinding2(p_vsa_c[:,4],p_vsa_c[:,5])    
        p3_12 = block_unbinding2(p_vsa_c[:,6],p_vsa_c[:,7]) 
        # Compute difference between first and last panel in each row
        p1_13 = block_unbinding2(p_vsa_c[:,0],p_vsa_c[:,2])    
        p2_13 = block_unbinding2(p_vsa_c[:,3],p_vsa_c[:,5])    

        # Compare the delta vector with target delta vector (1, or 2)
        s1 = match_prob(p1_12,x_target,self.match_act)
        s2 = match_prob(p1_23,x_target,self.match_act)
        s3 = match_prob(p2_12,x_target,self.match_act)
        s4 = match_prob(p2_23,x_target,self.match_act)
        s5 = match_prob(p3_12,x_target,self.match_act)
        # Compare the delta vector with TWICE target delta vector (1, or 2)
        s6 = match_prob(p1_13,block_binding2(x_target,x_target),self.match_act)
        s7 = match_prob(p2_13,block_binding2(x_target,x_target),self.match_act)

        # check if delta is not zero
        s0 = t.clamp(match_prob_0(p1_12,self.match_act),min=0,max=1)

        # Compute overall probability, EPS for numerical stability  
        result = s1*s2*s3*s4*s5*s6*s7*(1-s0)+EPS
        return  result,VSA_WEIGHT

    def arithmetic_plus(self,p_vsa_c,vsa_cb_cont_a): 
        # Bind first two panels in each row
        p1_12 = block_binding2(p_vsa_c[:,0],p_vsa_c[:,1])
        p2_12 = block_binding2(p_vsa_c[:,3],p_vsa_c[:,4])
        p3_12 = block_binding2(p_vsa_c[:,6],p_vsa_c[:,7])
        # Compare last panel 
        s1 = match_prob(p1_12,p_vsa_c[:,2],self.match_act)
        s2 = match_prob(p2_12,p_vsa_c[:,5],self.match_act)
        # for last row, check if bound vector is within the dictionary range 
        flag =  t.clamp(t.sum(match_prob_multi_batched(vsa_cb_cont_a,p3_12,self.match_act),dim=-1) ,min=0,max=1)

        # Compute overall probability, EPS for numerical stability  
        result = s1*s2*flag+EPS
        return  result, 6*VSA_WEIGHT 

    def arithmetic_minus(self,p_vsa_c,vsa_cb_cont_a): 
        # Unbind first two panels in each row
        p1_12 = block_unbinding2(p_vsa_c[:,0],p_vsa_c[:,1])
        p2_12 = block_unbinding2(p_vsa_c[:,3],p_vsa_c[:,4])
        p3_12 = block_unbinding2(p_vsa_c[:,6],p_vsa_c[:,7])
        # Compare with last panel 
        s1 = match_prob(p1_12,p_vsa_c[:,2],self.match_act)
        s2 = match_prob(p2_12,p_vsa_c[:,5],self.match_act)
        # for last row, check if bound vector is within the dictionary range 
        flag = t.clamp(t.sum(match_prob_multi_batched(vsa_cb_cont_a,p3_12,self.match_act),dim=-1) ,min=0,max=1)

        # Compute overall probability, EPS for numerical stability  
        result = s1*s2*flag+EPS
        return  result,6*VSA_WEIGHT
        
class vsa_rule_executor_extended(object):
    '''
    VSA backend: rule execution

    Parameters
    ----------
    executor_act:   str
        Activation after computing the match
    executor_m:     float
        Threshold value if executor_act=threshold
    executor_s:     float 
        Scaling of similarities before executor_cos2pmf_act
    executor_cos2pmf_act: str
        Cosine to pmf function 
    '''
    def __init__(self,executor_act="threshold",executor_m=1, executor_s=1, executor_cos2pmf_act="Identity", **kwargs): 
        self.s = executor_s
        self.match_act = t.nn.Threshold(executor_m,0) if executor_act=="threshold" else getattr(t.nn, executor_act)()
        self.cos2pmf_act = executor_cos2pmf_act 
        self.pmf2vec = pmf2vec

    def distribute_three(self,vsa_cb_discrete_a,p_vsa_d): 
        # Bind all vectors in the first row
        temp1 = block_binding3(p_vsa_d[0], p_vsa_d[1], p_vsa_d[2])
        # Unbind the first row with first two panels in last row => prediction
        pred = block_unbinding3(temp1, p_vsa_d[6], p_vsa_d[7])
        # Compute similarities of predcition and codebooks
        sim = match_prob_multi(vsa_cb_discrete_a,pred,self.match_act)
        # Transform similarities to PMF
        pred = cosine2pmf(sim,self.cos2pmf_act,self.s)
        return pred

    def constant(self,vsa_cb_discrete_a,p_vsa_d): 
        # Prediction is simply the penultimate panel    
        pred =p_vsa_d[7] 
        # Compute similarities of predcition and codebooks
        sim = match_prob_multi(vsa_cb_discrete_a,pred,self.match_act)
        # Transform similarities to PMF
        pred = cosine2pmf(sim,self.cos2pmf_act,10)
        return pred

    def progression_plus(self,vsa_cb_cont_a,p_vsa_c, x_target):
        # prediction is binding of penultimate panel and target delta
        pred = block_binding2(p_vsa_c[7],x_target)
        # Compute similarities of predcition and codebooks
        sim = match_prob_multi(vsa_cb_cont_a,pred,self.match_act)
        # Transform similarities to PMF
        pred = cosine2pmf(sim,self.cos2pmf_act, self.s)
        return pred

    def progression_minus(self,vsa_cb_cont_a,p_vsa_c, x_target):
        # prediction is unbinding of penultimate panel and target delta
        pred = block_unbinding2(p_vsa_c[7],x_target)
        # Compute similarities of predcition and codebooks
        sim = match_prob_multi(vsa_cb_cont_a,pred,self.match_act)
        # Transform similarities to PMF
        pred = cosine2pmf(sim, self.cos2pmf_act,self.s)
        return pred

    def arithmetic_plus(self,vsa_cb_continuous_a, p_vsa_c): 
        # prediction is binding of panels in last row 
        pred = block_binding2(p_vsa_c[6],p_vsa_c[7])
        # Compute similarities of predcition and codebooks
        sim = match_prob_multi(vsa_cb_continuous_a,pred,self.match_act)
        # Transform similarities to PMF
        pred = cosine2pmf(sim, self.cos2pmf_act,self.s)
        return pred 

    def arithmetic_minus(self,vsa_cb_continuous_a, p_vsa_c): 
        # prediction is unbinding of panels in last row 
        pred = block_unbinding2(p_vsa_c[6],p_vsa_c[7])
        # Compute similarities of predcition and codebooks
        sim = match_prob_multi(vsa_cb_continuous_a,pred,self.match_act)
        # Transform similarities to PMF
        pred = cosine2pmf(sim,self.cos2pmf_act,self.s)
        return pred