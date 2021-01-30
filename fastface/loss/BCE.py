import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BinaryCrossEntropy(nn.Module):
    __negative_selection_rules__ = ("none","ohem","random","mix",)
    def __init__(self, negative_selection_rule:str='none', **kwargs):
        super().__init__()
        self.neg_select_rule = negative_selection_rule
        assert self.neg_select_rule in self.__negative_selection_rules__,"given negative selection rule is not defined"

        self.min_num_of_neg_ratio = kwargs.get('min_num_of_neg_ratio', 0.01)
        if self.neg_select_rule in ("ohem","mix"):
            # select 5 negatives for 1 positive using ohem selection
            self.ohem_ratio = kwargs.get('ohem_ratio', 5)
        if self.neg_select_rule in ("random","mix"):
            # select 5 negatives for 1 positive using random selection
            self.random_ratio = kwargs.get('random_ratio', 5)
        if self.neg_select_rule in ("random","ohem","mix"):
            # select 10 negatives for 1 positive
            self.neg_select_ratio = kwargs.get('neg_select_ratio', 10)

    def forward(self, input:torch.Tensor, target:torch.Tensor):
        # input: torch.Tensor(N,)
        # target: torch.Tensor(N,)
        loss = F.binary_cross_entropy_with_logits(input,target, reduction='none')

        if self.neg_select_rule == 'none':
            return loss.mean()

        s_pos, = torch.where(target==1)
        s_neg, = torch.where(target==0)

        num_of_positives = s_pos.size(0)
        num_of_negatives = s_neg.size(0)
        min_num_of_negatives = int(target.size(0)*self.min_num_of_neg_ratio)
        max_num_of_negatives = num_of_negatives

        num_of_negatives = int(num_of_positives*self.neg_select_ratio)

        if self.neg_select_rule == 'random':
            # select negatives randomly
            num_of_negatives = int(num_of_positives*self.random_ratio)
            num_of_negatives = max(min_num_of_negatives, num_of_negatives)
            num_of_negatives = min(max_num_of_negatives, num_of_negatives)
            num_of_negatives = max(num_of_negatives,1) # for safety
            s_neg = random.sample(s_neg.cpu().numpy().tolist(), k=num_of_negatives)
            if len(s_neg) == 0:
                print("zero negatives found! ")
                print("min_num_of_negatives: ",min_num_of_negatives)
                print("max_num_of_negatives: ",max_num_of_negatives)
                print("num_of_negatives: ",num_of_negatives)
                print("s_pos: ",s_pos)
                exit(0)
            if s_pos.size(0) == 0:
                return loss[s_neg].mean()
            return torch.cat([loss[s_neg],loss[s_pos]]).mean()
        elif self.neg_select_rule == 'ohem':
            # select negatives using ohem
            num_of_negatives = min(num_of_negatives, num_of_positives*self.ohem_ratio)
            num_of_negatives = max(min_num_of_negatives, num_of_negatives)
            num_of_negatives = min(max_num_of_negatives, num_of_negatives)
            pos_loss = loss[s_pos]
            neg_loss = loss[s_neg]
            s_neg = neg_loss.argsort(descending=True)
            neg_loss = neg_loss[s_neg][:num_of_negatives]
            if pos_loss.size(0) == 0:
                return neg_loss.mean()
            return torch.cat([pos_loss,neg_loss]).mean()
        elif self.neg_select_rule == 'mix':
            # use ohem and random selection
            raise NotImplementedError("not yet implemented, use another")
        else:
            raise AssertionError("this line should not be show up!!")